from __future__ import annotations

import json
import os
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from agent_gem.core.task_schema import TaskPackage
from agent_gem.sandbox import SandboxExecutor, SandboxFusionExecutor
from agent_gem.tools import BashTool, PythonRunnerTool, SearchTool

from ..base import BaseAgent, TaskContext
from .data_pipeline import DataPipelineMixin
from .setup_flow import SetupMixin
from .task_builder import TaskBuilderMixin
from .tool_synthesis import ToolSynthesisMixin
from .validation_flow import ValidationMixin

if TYPE_CHECKING:  # pragma: no cover
    from agent_gem.generator import GenerationRequest

logger = logging.getLogger(__name__)


class GeneralAgent(SetupMixin, DataPipelineMixin, ToolSynthesisMixin, TaskBuilderMixin, ValidationMixin, BaseAgent):
    agent_type = "general_agent"
    description = "Automatic environment-synthesis agent that creates diverse, verifiable tasks with a growing toolset."

    def _configure_sandbox(self, sandbox: SandboxExecutor):
        if not isinstance(sandbox, SandboxFusionExecutor):
            sandbox.register_tool(BashTool(workdir=sandbox.sandbox_dir, timeout_s=sandbox.timeout_s))
        try:
            sandbox.register_tool(SearchTool(cache_path=sandbox.search_cache_path))
        except ValueError as exc:
            logger.error("Search tool configuration failed: %s", exc)
            raise
        if not isinstance(sandbox, SandboxFusionExecutor):
            sandbox.register_tool(PythonRunnerTool(workdir=sandbox.sandbox_dir, timeout_s=sandbox.timeout_s))
        sandbox.set_tool_call_callback(self._record_tool_call)

    def generate(self, request: GenerationRequest) -> Optional[TaskPackage]:
        if not request.topic:
            request.topic = "general task"

        task_id = str(uuid.uuid4())
        ctx = TaskContext(task_id=task_id, request=request)

        sandbox_dir = Path(self.writer.task_dir(task_id, self.agent_type), "_sandbox")
        if request.use_sandbox_fusion:
            sandbox = SandboxFusionExecutor(
                sandbox_dir=sandbox_dir,
                base_url=os.getenv("SANDBOX_FUSION_URL", "http://localhost:8080"),
                timeout_s=20,
            )
        else:
            sandbox = SandboxExecutor(sandbox_dir=sandbox_dir)
        self._configure_sandbox(sandbox)

        logger.info(
            "Generating task: %s, topic: %s, path: %s",
            task_id,
            request.topic,
            self.writer.task_dir(task_id, self.agent_type),
        )

        logger.info("Step 1/7: seeding database from web search and real pages...")
        records = self._seed_database(request.topic, ctx, sandbox)
        logger.info("Step 2/7: materializing local data files from harvested records...")
        self._create_data_files_from_records(records, sandbox, ctx)

        logger.info("Step 3/7: generating and running setup_env.py...")
        setup_bundle, setup_snapshot = self._generate_setup_bundle(request.topic, ctx, sandbox, records)
        logger.info("Step 4/7: profiling sandbox data sources...")
        data_profile = self._inspect_data_sources(sandbox, ctx)

        logger.info("Step 5/7: synthesizing task-specific tools...")
        task_tool_specs, tools_code = self._synthesize_task_tools(request.topic, records, ctx, sandbox, data_profile)
        self.writer.record_steps(task_id, self.agent_type, ctx.history)
        self._register_task_tools(task_tool_specs, sandbox, ctx, tools_code=tools_code)
        tool_selftest = self._self_test_tools(task_tool_specs, sandbox, request.topic, ctx)

        regen_needed, regen_reasons = self._needs_tool_regeneration(tool_selftest)
        if regen_needed:
            task_tool_specs, tools_code, tool_selftest = self._regenerate_tools_with_selftest(
                request.topic,
                records,
                ctx,
                sandbox,
                data_profile,
                tool_selftest,
                task_tool_specs,
                tools_code,
            )
            ctx.add_step({"type": "tool_regeneration_triggered", "reasons": regen_reasons})
            logger.info("Tool regeneration completed to cover missing fields: %s", regen_reasons)

        logger.info("Step 6/7: proposing initial task and code...")
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
        expected_fields = self._expected_fields_from_format(package.task.submit_result_format)
        if expected_fields:
            regen_for_format, format_reasons = self._needs_tool_regeneration(tool_selftest, required_fields=expected_fields)
            if regen_for_format:
                task_tool_specs, tools_code, tool_selftest = self._regenerate_tools_with_selftest(
                    request.topic,
                    records,
                    ctx,
                    sandbox,
                    data_profile,
                    tool_selftest,
                    task_tool_specs,
                    tools_code,
                    required_fields=expected_fields,
                )
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
                ctx.add_step(
                    {"type": "tool_regeneration_for_format", "reasons": format_reasons, "expected_fields": sorted(expected_fields)}
                )
                logger.info("Tool regeneration for expected submit_result_format fields: %s", sorted(expected_fields))
        if package.metadata is None:
            package.metadata = {}
        package = package.copy(
            update={
                "metadata": {
                    **(package.metadata or {}),
                    "data_profile": json.dumps(data_profile, ensure_ascii=False)[:4000],
                    "tool_selftest": json.dumps(tool_selftest, ensure_ascii=False)[:4000],
                }
            }
        )
        package = self._ensure_substantive_task(task_tool_specs, package, ctx, request)
        logger.info("Step 7/7: running validation and refinement passes...")
        package = self._ensure_valid(
            request,
            package,
            ctx,
            sandbox,
            records,
            setup_snapshot=setup_snapshot,
            setup_bundle=setup_bundle,
            tools_code=tools_code,
        )
        task_tool_specs = list(package.task.tool_set)
        self.writer.record_steps(task_id, self.agent_type, ctx.history)

        effective_refine_rounds = max(1, max(request.max_refine_rounds, int(request.difficulty)))
        for round_idx in range(2, effective_refine_rounds + 1):
            target = min(int(request.difficulty), round_idx)
            refined = self._refine_task(
                previous=package,
                records=records,
                tool_specs=task_tool_specs,
                ctx=ctx,
                target_difficulty=target,
            )
            refined = self._ensure_substantive_task(task_tool_specs, refined, ctx=ctx, request=request)
            current_tools_code = (package.metadata or {}).get("tools_code", tools_code)
            package = self._ensure_valid(
                request,
                refined,
                ctx,
                sandbox,
                records,
                setup_snapshot=None,
                setup_bundle=None,
                tools_code=current_tools_code,
            )
            task_tool_specs = list(package.task.tool_set)
            self.writer.record_steps(task_id, self.agent_type, ctx.history)

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
                self.writer.persist_quadruple_format(
                    category=request.topic or "general task",
                    records=records,
                    packages=[package],
                    merge=False,
                )
            except Exception:
                logger.debug("Failed to persist quadruple format", exc_info=True)

        return package

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
