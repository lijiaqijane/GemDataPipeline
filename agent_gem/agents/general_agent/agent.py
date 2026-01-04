from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from agent_gem.core.task_schema import TaskPackage
from agent_gem.sandbox import SandboxExecutor
from agent_gem.tools import SearchTool

from ..base import BaseAgent, TaskContext
from .data_pipeline import DataPipelineMixin
from .persist import persist_quadruple_format
from .sandbox import GeneralAgentSandboxFusionExecutor
from .task_builder import TaskBuilderMixin
from .tool_synthesis import ToolSynthesisMixin
from .validation_flow import ValidationMixin

if TYPE_CHECKING:  # pragma: no cover
    from agent_gem.generator import GenerationRequest

logger = logging.getLogger(__name__)


class GeneralAgent(DataPipelineMixin, ToolSynthesisMixin, TaskBuilderMixin, ValidationMixin, BaseAgent):
    agent_type = "general_agent"
    description = "Automatic environment-synthesis agent that creates diverse, verifiable tasks with a growing toolset."

    def _configure_sandbox(self, sandbox: SandboxExecutor):
        try:
            sandbox.register_tool(
                SearchTool(
                    cache_path=sandbox.search_cache_path,
                    bash_runner=sandbox.execute_bash,
                )
            )
        except ValueError as exc:
            logger.error("Search tool configuration failed: %s", exc)
            raise
        sandbox.set_tool_call_callback(self._record_tool_call)
        sandbox.set_tool_call_allowlist(set())

    def generate(self, request: GenerationRequest) -> Optional[TaskPackage]:
        try:
            if not request.topic:
                request.topic = "general task"

            # Generate task_id: use prefix if provided, otherwise use UUID
            if request.task_id_prefix:
                task_id = request.task_id_prefix
            else:
                task_id = str(uuid.uuid4())
            ctx = TaskContext(task_id=task_id, request=request)

            # Set task-specific llm_io.log path
            task_dir = self.writer.task_dir(task_id, self.agent_type)
            task_llm_log = task_dir / "llm_io.log"
            original_llm_log_file = os.environ.get("LLM_LOG_IO_FILE")
            os.environ["LLM_LOG_IO_FILE"] = str(task_llm_log)
            # Reinitialize LLM client with new log path if log_io is enabled
            if os.getenv("LLM_LOG_IO", "0") in {"1", "true", "True"}:
                from agent_gem.llm import LLMClient
                self.llm = LLMClient.from_env()

            sandbox_dir = Path(task_dir, "_sandbox")
            if not request.use_sandbox_fusion:
                raise RuntimeError("SandboxFusion is required; local sandbox execution is disabled.")
            sandbox = GeneralAgentSandboxFusionExecutor(
                sandbox_dir=sandbox_dir,
                base_url=os.getenv("SANDBOX_FUSION_URL", "http://localhost:8080"),
                timeout_s=20,
            )
            self._configure_sandbox(sandbox)
            required_tools = {"bash", "search"}
            available = sandbox.tool_names()
            missing = required_tools - available
            if missing:
                raise RuntimeError(f"Sandbox missing required tools: {sorted(missing)}")

            logger.info(
                "Generating task: %s, topic: %s, path: %s",
                task_id,
                request.topic,
                self.writer.task_dir(task_id, self.agent_type),
            )

            records: list[dict[str, Any]] | None = None
            resume_seed = os.getenv("RESUME_SEEDED_DB", "").lower() in {"1", "true", "yes"}
            if resume_seed:
                records = self._load_seeded_records(ctx)
                if records is not None:
                    logger.info("Step 1/5: resuming from existing seeded database (skip seeding).")
            if records is None:
                logger.info("Step 1/5: seeding database from web search and real pages...")
                records = self._seed_database(request.topic, ctx, sandbox)

            logger.info("Step 2/5: profiling sandbox data sources...")
            data_profile = self._inspect_data_sources(sandbox, ctx)

            logger.info("Step 3/5: synthesizing task-specific tools...")
            task_tool_specs, tools_code, tool_selftest = self._synthesize_task_tools_with_retry(
                request.topic, records, ctx, sandbox, data_profile
            )
            self.writer.record_steps(task_id, self.agent_type, ctx.history)

            logger.info("Step 4/5: proposing initial task and code...")
            package = self._propose_task(
                task_id,
                request,
                records,
                task_tool_specs,
                ctx,
                tools_code=tools_code,
                data_profile=data_profile,
                tool_selftest=tool_selftest,
            )
            # Check if format-based regeneration is needed
            expected_fields = self._expected_fields_from_format(package.task.submit_result_format)
            if expected_fields:
                task_tool_specs, tools_code, tool_selftest = self._ensure_tools_meet_format_requirements(
                    request.topic,
                    records,
                    ctx,
                    sandbox,
                    data_profile,
                    task_tool_specs,
                    tools_code,
                    tool_selftest,
                    expected_fields,
                )
                # Re-propose task after regeneration
                package = self._propose_task(
                    task_id,
                    request,
                    records,
                    task_tool_specs,
                    ctx,
                    tools_code=tools_code,
                    data_profile=data_profile,
                    tool_selftest=tool_selftest,
                )
            if package.metadata is None:
                package.metadata = {}
            package = package.copy(
                update={
                    "metadata": {
                        **(package.metadata or {}),
                        "data_profile": self._safe_metadata_json(data_profile),
                        "tool_selftest": self._safe_metadata_json(tool_selftest),
                    }
                }
            )
            package = self._ensure_substantive_task(task_tool_specs, package, ctx, request)
            logger.info("Step 5/5: running validation and refinement passes...")
            package = self._ensure_valid(
                request,
                package,
                ctx,
                sandbox,
                records,
                tools_code=tools_code,
            )
            task_tool_specs = list(package.task.tool_set)
            self.writer.record_steps(task_id, self.agent_type, ctx.history)

            # Collect all validated packages (all difficulty levels)
            validated_packages = []
            meta = package.metadata or {}
            # Only add if validation passed (no validation errors)
            if not any(meta.get(key) for key in ("validation_error", "verification_error", "repair_failed")):
                validated_packages.append(package)

            effective_refine_rounds = max(1, max(request.max_refine_rounds, int(request.difficulty)))
            for round_idx in range(2, effective_refine_rounds + 1):
                target = min(int(request.difficulty), round_idx)
                ctx.current_difficulty = target
                try:
                    tool_selftest = json.loads((package.metadata or {}).get("tool_selftest", "{}"))
                except Exception:
                    tool_selftest = {}
                refined = self._refine_task(
                    previous=package,
                    records=records,
                    tool_specs=task_tool_specs,
                    ctx=ctx,
                    target_difficulty=target,
                    tool_selftest=tool_selftest if isinstance(tool_selftest, dict) else None,
                )
                refined = self._ensure_substantive_task(task_tool_specs, refined, ctx=ctx, request=request)
                current_tools_code = (package.metadata or {}).get("tools_code", tools_code)
                package = self._ensure_valid(
                    request,
                    refined,
                    ctx,
                    sandbox,
                    records,
                    tools_code=current_tools_code,
                )
                task_tool_specs = list(package.task.tool_set)
                self.writer.record_steps(task_id, self.agent_type, ctx.history)
                # Only add if validation passed (no validation errors)
                meta = package.metadata or {}
                if not any(meta.get(key) for key in ("validation_error", "verification_error", "repair_failed")):
                    validated_packages.append(package)

            if request.persist_result and self.writer is not None:
                self.writer.record_steps(
                    task_id,
                    self.agent_type,
                    [step.to_payload() for step in ctx.history],
                    extra={
                        "topic": request.topic,
                        "difficulty": request.difficulty,
                        "records_count": len(records),
                        "task_tools": [spec.name for spec in task_tool_specs],
                    },
                )
                try:
                    persist_quadruple_format(
                        self.writer,
                        category=request.topic or "general task",
                        records=records,
                        packages=validated_packages,
                    )
                except Exception:
                    logger.debug("Failed to persist quadruple format", exc_info=True)

            return package
        except Exception as e:
            logger.error(f"Failed to generate task package: {e}", exc_info=True)
            return None
        finally:
            # Restore original LLM_LOG_IO_FILE if it was set
            if "original_llm_log_file" in locals():
                if original_llm_log_file is None:
                    os.environ.pop("LLM_LOG_IO_FILE", None)
                else:
                    os.environ["LLM_LOG_IO_FILE"] = original_llm_log_file

    def _record_tool_call(self, record: Any, ctx: TaskContext) -> None:
        try:
            message = {
                "type": "tool_call",
                "tool": getattr(record, "tool", None),
                "input": getattr(record, "tool_input", None),
                "output": getattr(record, "tool_output", None),
                "error": getattr(record, "error", None),
                "duration_s": getattr(record, "duration_s", None),
            }
            ctx.add_step(message, request_id=f"tool_{getattr(record, 'call_id', '')}")
        except Exception:
            logger.debug("Failed to record tool call step", exc_info=True)
