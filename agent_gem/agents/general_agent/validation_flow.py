from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Optional, TYPE_CHECKING

from agent_gem.core.task_schema import TaskPackage, ToolSpec
from agent_gem.core.validation import CodeValidator
from agent_gem.sandbox import SandboxFusionExecutor

from ..base import TaskContext
from .sandbox import GeneralAgentSandboxFusionExecutor

SandboxType = GeneralAgentSandboxFusionExecutor

if TYPE_CHECKING:  # pragma: no cover
    from agent_gem.generator import GenerationRequest  # noqa: F401


class ValidationMixin:
    """Validation, repair, and quality gates for generated tasks."""
    logger = logging.getLogger(__name__)

    @staticmethod
    def _hash_code(value: str | None) -> str:
        if not value:
            return ""
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _summarize_for_repair(value: Any, *, limit: int = 1200) -> str:
        if value is None:
            return ""
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = repr(value)
        text = text.replace("\n", " ").strip()
        if len(text) > limit:
            return text[:limit] + "..."
        return text

    @staticmethod
    def _verifier_is_broken(run: Any) -> bool:
        if run.verified is None:
            return True
        msg = (run.verification_error or "").lower()
        if not msg:
            return False
        return any(
            key in msg
            for key in (
                "verification execution failed",
                "verification returned unsupported type",
                "verification returned no boolean result",
                "missing 'def verify",
                "syntax error",
            )
        )

    @staticmethod
    def _tool_selftest_missing_or_empty(tool_selftest: dict[str, Any] | None) -> bool:
        if not tool_selftest:
            return True
        saw_ok = False
        any_ok = False
        any_fields = False
        for info in tool_selftest.values():
            if not isinstance(info, dict):
                continue
            if "ok" in info:
                saw_ok = True
                if info.get("ok") is True:
                    any_ok = True
            if info.get("fields"):
                any_fields = True
        if saw_ok:
            return not any_ok
        return not any_fields

    def _choose_repair_target(
        self,
        *,
        run: Any,
        attempt: int,
        solution_repeat: int,
        verification_repeat: int,
    ) -> str:
        if self._verifier_is_broken(run):
            target = "verification"
        else:
            target = "solution"

        if solution_repeat >= 1 and verification_repeat >= 1:
            return "both"
        if solution_repeat >= 1 and target == "verification":
            return "both"
        if attempt >= 3 and run.verified is False and target == "solution":
            return "both"
        return target

    def _build_verification_failure_error(self, run: Any) -> str:
        parts: list[str] = []
        if run.verification_error:
            parts.append(f"verification_error: {run.verification_error}")
        if run.verification_score is not None:
            parts.append(f"verification_score: {run.verification_score}")
        if run.verification_details is not None:
            details_preview = self._summarize_for_repair(run.verification_details, limit=1200)
            if details_preview:
                parts.append(f"verification_details: {details_preview}")
        if run.answer is not None:
            answer_preview = self._summarize_for_repair(run.answer, limit=1800)
            if answer_preview:
                parts.append(f"answer_preview: {answer_preview}")
        return "\n".join(parts) if parts else "verification returned False"

    @staticmethod
    def _normalize_solution_indentation(code: str) -> str:
        """Ensure lines inside solve() are indented to avoid IndentationError."""
        if not code or not isinstance(code, str):
            return code
        lines = code.splitlines()
        try:
            def_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("def solve"))
        except StopIteration:
            return code
        fixed = lines[: def_idx + 1]
        for line in lines[def_idx + 1 :]:
            if not line.strip():
                fixed.append(line)
                continue
            leading = len(line) - len(line.lstrip(" "))
            if leading < 4:
                fixed.append("    " + line.lstrip())
            else:
                fixed.append(line)
        return "\n".join(fixed) + ("\n" if code.endswith("\n") else "")

    def _ensure_valid(
        self,
        request: "GenerationRequest",
        package: TaskPackage,
        ctx: TaskContext,
        sandbox: SandboxType,
        records: list[dict[str, Any]],
        *,
        tools_code: str | None = None,
    ) -> TaskPackage:
        if not request.validate or sandbox is None:
            return package

        try:
            metadata_profile = json.loads((package.metadata or {}).get("data_profile", "{}"))
        except Exception:
            metadata_profile = {}

        last_error: str | None = None
        augmented_once = False
        last_solution_hash: str | None = None
        last_verification_hash: str | None = None
        solution_repeat = 0
        verification_repeat = 0
        for attempt in range(1, request.max_validation_rounds + 1):
            if attempt > 1:
                self.logger.warning(
                    "Retrying task validation (attempt %s/%s). Previous error: %s",
                    attempt,
                    request.max_validation_rounds,
                    last_error or "unknown error",
                )
            current_solution_hash = self._hash_code(package.solution)
            current_verification_hash = self._hash_code(package.verification)
            if last_solution_hash == current_solution_hash:
                solution_repeat += 1
            else:
                solution_repeat = 0
            if last_verification_hash == current_verification_hash:
                verification_repeat += 1
            else:
                verification_repeat = 0
            last_solution_hash = current_solution_hash
            last_verification_hash = current_verification_hash
            if package.solution:
                sol_ok, sol_err = CodeValidator.validate_solution_code(package.solution)
                if not sol_ok:
                    fixed = self._normalize_solution_indentation(package.solution)
                    fixed_ok, fixed_err = CodeValidator.validate_solution_code(fixed)
                    if fixed_ok:
                        package = package.copy(update={"solution": fixed})
                        ctx.add_step({"type": "solution_indentation_fixed", "error": sol_err})
                    else:
                        last_error = fixed_err or sol_err
                        package = self._repair_package(
                            request, package, ctx, error=last_error, records=records
                        )
                        continue
            allowed_tools = {spec.name for spec in package.task.tool_set}
            submit_tool = "submit_result"
            verify_called_tools = CodeValidator.extract_tool_calls(package.verification or "")
            verify_missing = verify_called_tools - allowed_tools
            if verify_missing:
                last_error = (
                    "verification calls tools not in the declared tool_set; "
                    f"missing={sorted(verify_missing)} allowed={sorted(allowed_tools)}"
                )
                package = self._repair_package(
                    request,
                    package,
                    ctx,
                    error=last_error,
                    records=records,
                    repair_target="verification",
                )
                continue
            called_tools = CodeValidator.extract_tool_calls(package.solution)
            called_data_tools = {name for name in called_tools if name != submit_tool}
            if len(called_data_tools) < 2:
                last_error = (
                    "solution must call at least 2 different tools; "
                    f"called={sorted(called_tools)} allowed={sorted(allowed_tools)}"
                )
                package = self._repair_package(request, package, ctx, error=last_error, records=records)
                continue
            missing_tools = called_tools - allowed_tools
            if missing_tools:
                last_error = (
                    "solution calls tools not in the declared tool_set; "
                    f"missing={sorted(missing_tools)} allowed={sorted(allowed_tools)}"
                )
                package = self._repair_package(request, package, ctx, error=last_error, records=records)
                continue

            pre_snapshot = sandbox.snapshot_fs()

            run_start = time.time()
            difficulty = int(
                getattr(ctx, "current_difficulty", 0)
                or getattr(request, "difficulty", 0)
                or getattr(package.task, "difficulty_level", 1)
                or 1
            )
            group_prefix = f"difficulty_{difficulty}"
            # User preference: only one folder per difficulty; all attempts (including retries) go into it.
            run = sandbox.run_task(package, run_group=group_prefix)
            post_snapshot = sandbox.snapshot_fs()
            used_tools: set[str] = set()
            tool_call_count = 0
            try:
                log_path = getattr(sandbox, "tool_calls_path", None)
                if log_path and log_path.exists():
                    for line in log_path.read_text(encoding="utf-8").splitlines():
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        if rec.get("started_at", 0) >= run_start:
                            tool_call_count += 1
                            if isinstance(rec.get("tool"), str):
                                used_tools.add(rec["tool"])
            except Exception:
                pass
            if not used_tools and isinstance(sandbox, SandboxFusionExecutor):
                # SandboxFusion executes tools remotely without local tool call logs.
                used_tools = set(called_tools)
                tool_call_count = max(tool_call_count, len(used_tools))
            new_files = sorted(set(post_snapshot) - set(pre_snapshot))
            changed_files = sorted(k for k in set(post_snapshot).intersection(pre_snapshot) if post_snapshot[k] != pre_snapshot[k])

            verifier_weak = (
                run.verification_error
                and run.verification_score is None
                and (not run.verification_details)
            )

            snapshot_hash = lambda snap: hashlib.sha256(json.dumps(snap, sort_keys=True).encode("utf-8")).hexdigest()
            meta_update = {
                "pre_state_hash": snapshot_hash(pre_snapshot),
                "post_state_hash": snapshot_hash(post_snapshot),
                "new_artifact_count": str(len(new_files)),
                "changed_artifact_count": str(len(changed_files)),
            }
            if tools_code:
                meta_update |= {
                    "tools_code_preview": tools_code[:200],
                    "tools_code": tools_code[:5000] if len(tools_code) > 2000 else tools_code,
                }
            elif package.metadata and "tools_code" in package.metadata:
                meta_update["tools_code"] = package.metadata.get("tools_code", "")
            meta_update |= {
                "runtime_tool_calls": ",".join(sorted(used_tools)),
                "runtime_tool_call_count": str(tool_call_count),
            }
            package = package.copy(update={"metadata": {**(package.metadata or {}), **meta_update}})
            ctx.add_step(
                {
                    "type": "task_validation",
                    "attempt": attempt,
                    "verified": run.verified,
                    "error": run.error,
                    "verification_error": run.verification_error,
                    "new_files": new_files[:10],
                    "changed_files": changed_files[:10],
                    "runtime_tools_used": sorted(used_tools),
                    "runtime_tool_call_count": tool_call_count,
                    "solution_repeat": solution_repeat,
                    "verification_repeat": verification_repeat,
                }
            )
            if verifier_weak:
                last_error = self._build_verification_failure_error(run)
                target = self._choose_repair_target(
                    run=run,
                    attempt=attempt,
                    solution_repeat=solution_repeat,
                    verification_repeat=verification_repeat,
                )
                if target == "both":
                    last_error = f"{last_error}\nstagnation: solution/verification repeated; rewrite both."
                try:
                    sandbox.annotate_run_record(
                        run_group=group_prefix,
                        run_id=run.run_id,
                        updates={
                            "task_candidate": False,
                            "task_candidate_reason": last_error,
                        },
                    )
                except Exception:
                    pass
                package = self._repair_package(
                    request,
                    package,
                    ctx,
                    error=last_error,
                    records=records,
                    repair_target=target,
                )
                continue
            used_data_tools = {name for name in used_tools if name != submit_tool}
            if len(used_data_tools) < 2:
                last_error = (
                    f"runtime tool calls insufficient; used={sorted(used_tools)} count={tool_call_count}"
                )
                try:
                    sandbox.annotate_run_record(
                        run_group=group_prefix,
                        run_id=run.run_id,
                        updates={
                            "task_candidate": False,
                            "task_candidate_reason": last_error,
                        },
                    )
                except Exception:
                    pass
                package = self._repair_package(request, package, ctx, error=last_error, records=records)
                continue
            if (not run.answer or run.verified is False) and attempt < request.max_validation_rounds:
                try:
                    data_profile = json.loads((package.metadata or {}).get("data_profile", "{}"))
                    tool_selftest = json.loads((package.metadata or {}).get("tool_selftest", "{}"))
                    metadata_profile = data_profile
                except Exception:
                    data_profile, tool_selftest = {}, {}
                if self._tool_selftest_missing_or_empty(tool_selftest):
                    refreshed_selftest: dict[str, Any] = {}
                    try:
                        refreshed_selftest = self._self_test_tools(
                            list(package.task.tool_set),
                            sandbox,
                            request.topic,
                            ctx,
                        )
                    except Exception:
                        refreshed_selftest = tool_selftest if isinstance(tool_selftest, dict) else {}
                    if refreshed_selftest:
                        tool_selftest = refreshed_selftest
                        package = package.copy(
                            update={
                                "metadata": {
                                    **(package.metadata or {}),
                                    "data_profile": json.dumps(data_profile, ensure_ascii=False)[:4000],
                                    "tool_selftest": json.dumps(refreshed_selftest, ensure_ascii=False)[:4000],
                                }
                            }
                        )
                if not augmented_once:
                    new_specs, new_code, augmented = self._augment_toolset(
                        request.topic or package.task.task_title,
                        records,
                        list(package.task.tool_set),
                        data_profile or {},
                        ctx,
                        sandbox,
                    )
                    if augmented:
                        augmented_once = True
                        tools_code = (package.metadata or {}).get("tools_code", "")
                        if new_code and new_code not in tools_code:
                            tools_code = (tools_code + "\n\n" + new_code).strip()
                        try:
                            new_selftest = self._self_test_tools(new_specs, sandbox, request.topic, ctx)
                        except Exception:
                            new_selftest = tool_selftest
                        package = package.copy(
                            update={
                                "task": package.task.copy(update={"tool_set": new_specs}),
                                "metadata": {
                                    **(package.metadata or {}),
                                    "tool_selftest": json.dumps(new_selftest, ensure_ascii=False)[:4000],
                                    "tools_code": tools_code[:5000] if tools_code else "",
                                    "toolset_augmented": "true",
                                },
                            }
                        )
                        package = self._repair_package(
                            request,
                            package,
                            ctx,
                            error="toolset augmented; update solution to use new tools",
                            records=records,
                        )
                        continue
                if self._tool_selftest_missing_or_empty(tool_selftest):
                    last_error = "tool_selftest_missing_or_empty; skipping tool resynthesis; verify/solution must use existing tools"
                    package = self._repair_package(
                        request,
                        package,
                        ctx,
                        error=last_error,
                        records=records,
                        repair_target="solution",
                    )
                    continue
                if run.answer is None or run.verified is None:
                    last_error = run.error or run.verification_error or "missing_answer_or_verdict"
                    package = self._repair_package(request, package, ctx, error=last_error, records=records)
                    continue
            if run.verified is True:
                # Additional check: ensure answer contains meaningful content
                answer_meaningful, meaningful_error = self._check_answer_has_meaningful_content(
                    run.answer, package.task.submit_result_format
                )
                if not answer_meaningful:
                    last_error = f"answer_is_empty_or_meaningless: {meaningful_error}"
                    try:
                        sandbox.annotate_run_record(
                            run_group=group_prefix,
                            run_id=run.run_id,
                            updates={
                                "result_persisted": False,
                                "result_persist_reason": last_error,
                                "task_candidate": False,
                                "task_candidate_reason": last_error,
                                "verification_error": meaningful_error,
                                "verified": False,  # Override verified status
                            },
                        )
                    except Exception:
                        pass
                    package = self._repair_package(request, package, ctx, error=last_error, records=records)
                    continue
                
                meta = package.metadata or {}
                cleaned_meta = {
                    k: v
                    for k, v in meta.items()
                    if k not in {"validation_error", "verification_error", "repair_error", "repair_failed"}
                }
                package = package.copy(update={"metadata": cleaned_meta})
                if new_files:
                    for rel in new_files:
                        try:
                            target = sandbox.sandbox_dir / rel
                            if target.exists():
                                target.unlink(missing_ok=True)
                        except Exception:
                            continue
                    try:
                        # Fold negative checks into the same difficulty folder (user preference).
                        negative = sandbox.run_task(package, run_group=group_prefix)
                        if negative.verified:
                            last_error = "negative_check_failed"
                            try:
                                sandbox.annotate_run_record(
                                    run_group=group_prefix,
                                    run_id=run.run_id,
                                    updates={
                                        "result_persisted": False,
                                        "result_persist_reason": last_error,
                                        "task_candidate": False,
                                        "task_candidate_reason": last_error,
                                    },
                                )
                                sandbox.annotate_run_record(
                                    run_group=group_prefix,
                                    run_id=negative.run_id,
                                    updates={
                                        "result_persisted": False,
                                        "result_persist_reason": last_error,
                                        "task_candidate": False,
                                        "task_candidate_reason": last_error,
                                    },
                                )
                            except Exception:
                                pass
                            package = self._repair_package(request, package, ctx, error=last_error, records=records)
                            continue
                    except Exception:
                        pass
                try:
                    sandbox.annotate_run_record(
                        run_group=group_prefix,
                        run_id=run.run_id,
                        updates={
                            "task_candidate": True,
                            "task_candidate_reason": "verified",
                        },
                    )
                except Exception:
                    pass
                try:
                    sandbox.persist_verified_result(group_prefix)
                except Exception:
                    pass
                return package

            if run.verified is False and not run.error:
                last_error = self._build_verification_failure_error(run)
                target = self._choose_repair_target(
                    run=run,
                    attempt=attempt,
                    solution_repeat=solution_repeat,
                    verification_repeat=verification_repeat,
                )
                if target == "both":
                    last_error = f"{last_error}\nstagnation: solution/verification repeated; rewrite both."
                try:
                    sandbox.annotate_run_record(
                        run_group=group_prefix,
                        run_id=run.run_id,
                        updates={
                            "task_candidate": False,
                            "task_candidate_reason": last_error,
                        },
                    )
                except Exception:
                    pass
                package = self._repair_package(
                    request,
                    package,
                    ctx,
                    error=last_error,
                    records=records,
                    repair_target=target,
                )
                continue

            if run.error or run.verification_error:
                last_error = run.error or run.verification_error or "unknown_error"
                target = self._choose_repair_target(
                    run=run,
                    attempt=attempt,
                    solution_repeat=solution_repeat,
                    verification_repeat=verification_repeat,
                )
                if target == "both":
                    last_error = f"{last_error}\nstagnation: solution/verification repeated; rewrite both."
                try:
                    sandbox.annotate_run_record(
                        run_group=group_prefix,
                        run_id=run.run_id,
                        updates={
                            "task_candidate": False,
                            "task_candidate_reason": last_error,
                        },
                    )
                except Exception:
                    pass
                package = self._repair_package(
                    request,
                    package,
                    ctx,
                    error=last_error,
                    records=records,
                    repair_target=target,
                )
                continue
            if attempt >= request.max_validation_rounds:
                break

        if last_error:
            package = package.copy(
                update={
                    "metadata": {
                        **(package.metadata or {}),
                        "validation_error": last_error,
                        "data_profile": json.dumps(metadata_profile, ensure_ascii=False)[:4000],
                    }
                }
            )
        return package

    def _repair_package(
        self,
        request: "GenerationRequest",
        package: TaskPackage,
        ctx: TaskContext,
        error: str | None = None,
        repair_target: str | None = None,
        records: Optional[list[dict[str, Any]]] = None,
    ) -> TaskPackage:
        # IMPORTANT: repair must NOT rely on legacy plan/builder functions.
        # All solve()/verify() code must be authored by the agent (LLM).
        err = (error or "").lower()
        target = repair_target
        if target is None:
            if "verification returned false" in err:
                target = "solution"
            elif any(key in err for key in ["verification", "verified", "verifier", "verify"]):
                target = "verification"
            elif any(
                key in err
                for key in [
                    "solution",
                    "solve",
                    "runtime",
                    "execution",
                    "syntax",
                    "indent",
                    "tool calls",
                    "missing_answer",
                    "format",
                    "schema",
                    "submit_result_format",
                    "answer_format",
                ]
            ):
                target = "solution"
            else:
                target = "solution"

        task_content = package.task.task_content
        submit_result_format = package.task.submit_result_format
        tool_specs = list(package.task.tool_set)
        meta = package.metadata or {}
        try:
            tool_selftest = json.loads(meta.get("tool_selftest", "{}"))
        except Exception:
            tool_selftest = {}

        ctx.add_step(
            {
                "type": "repair_attempt",
                "error": error,
                "target": target,
                "previous_solution_preview": (package.solution or "")[:400],
                "previous_verification_preview": (package.verification or "")[:400],
            }
        )

        try:
            solution, verification = self._generate_agent_solution_and_verification(
                request=request,
                ctx=ctx,
                task_content=task_content,
                submit_result_format=submit_result_format,
                tool_specs=tool_specs,
                records=records or [],
                tool_selftest=tool_selftest if isinstance(tool_selftest, dict) else {},
                previous_solution=package.solution,
                previous_verification=package.verification,
                repair_error=error or "",
                repair_target=target,
            )
        except Exception as exc:
            ctx.add_step({"type": "repair_failed", "error": str(exc)[:800]})
            return package.copy(
                update={
                    "metadata": {
                        **meta,
                        "repair_error": error or "",
                        "repair_failed": "true",
                    }
                }
            )

        return package.copy(
            update={
                "task": package.task.copy(
                    update={
                        "task_content": task_content,
                        "submit_result_format": submit_result_format,
                    }
                ),
                "solution": solution,
                "verification": verification,
                "metadata": {
                    **meta,
                    "repair_error": error or "",
                    "repaired": "true",
                },
            }
        )

    def _ensure_substantive_task(
        self,
        tool_specs: list[ToolSpec],
        package: TaskPackage,
        ctx: TaskContext,
        request: "GenerationRequest | None" = None,
    ) -> TaskPackage:
        allowed = {spec.name for spec in tool_specs}
        submit_tool = "submit_result"
        called = CodeValidator.extract_tool_calls(package.solution)
        called_data = {name for name in called if name != submit_tool}

        if called_data and called.issubset(allowed) and len(called_data) >= 2:
            return package

        ctx.add_step(
            {
                "type": "task_quality_gate",
                "reason": "insufficient_tool_usage",
                "called_tools": sorted(called),
                "allowed_tools": sorted(allowed),
            }
        )

        if request is None:
            return package

        repaired = self._repair_package(
            request,
            package,
            ctx,
            error="solution must call at least two distinct allowed tools",
            records=[],
        )
        repaired_called = CodeValidator.extract_tool_calls(repaired.solution)
        repaired_data = {name for name in repaired_called if name != submit_tool}
        if repaired_data and repaired_called.issubset(allowed) and len(repaired_data) >= 2:
            return repaired

        return package.copy(
            update={
                "metadata": {
                    **(package.metadata or {}),
                    "repair_failed": "true",
                    "repair_failure_reason": (
                        f"insufficient_tool_usage: called={sorted(called)} "
                        f"allowed={sorted(allowed)}"
                    ),
                }
            }
        )

    @staticmethod
    def _check_answer_has_meaningful_content(answer: Any, submit_result_format: Any = None) -> tuple[bool, str]:
        """Check if answer contains meaningful content (not just empty structures or default values).
        
        Returns:
            tuple[bool, str]: (is_meaningful, error_message)
            - is_meaningful: True if answer contains meaningful content, False otherwise
            - error_message: Detailed error message if not meaningful, empty string otherwise
        """
        if answer is None:
            return False, "Answer is None"
        
        # Extract expected keys from submit_result_format if provided
        expected_keys = set()
        if submit_result_format is not None:
            if isinstance(submit_result_format, str):
                try:
                    submit_result_format = json.loads(submit_result_format)
                except Exception:
                    pass
            if isinstance(submit_result_format, dict):
                # For example format (keys are the contract)
                if submit_result_format.get("type") != "object" or "properties" not in submit_result_format:
                    expected_keys = set(submit_result_format.keys())
                else:
                    # For JSON schema format
                    props = submit_result_format.get("properties", {})
                    required = submit_result_format.get("required", [])
                    expected_keys = set(required) if required else set(props.keys())
        
        # Unwrap submit_result wrapper if present
        # Recursively find a dict that contains all expected keys (if we know them)
        # or find a dict with non-empty meaningful content
        def _find_data_dict(obj: Any, expected_keys: set[str], max_depth: int = 5) -> Any:
            """Recursively find a dict that contains expected keys (if provided) or has meaningful content."""
            if max_depth <= 0 or not isinstance(obj, dict):
                return None
            
            # If we have expected keys, find dict containing all of them
            if expected_keys:
                if expected_keys.issubset(obj.keys()):
                    return obj
            else:
                # No expected keys: check if current dict has meaningful content
                has_content = any(
                    (isinstance(v, list) and len(v) > 0) or
                    (isinstance(v, str) and v.strip()) or
                    (isinstance(v, (int, float)) and v != 0) or
                    (isinstance(v, dict) and len(v) > 0)
                    for v in obj.values()
                )
                if has_content:
                    return obj
            
            # Recursively search nested dicts
            for value in obj.values():
                if isinstance(value, dict):
                    found = _find_data_dict(value, expected_keys, max_depth - 1)
                    if found is not None:
                        return found
                elif isinstance(value, list):
                    for item in value[:3]:  # Check first few items
                        if isinstance(item, dict):
                            found = _find_data_dict(item, expected_keys, max_depth - 1)
                            if found is not None:
                                return found
            
            return None
        
        # Try to find the actual data object
        data = answer
        if isinstance(answer, dict):
            found_data = _find_data_dict(answer, expected_keys, max_depth=5)
            if found_data is not None:
                data = found_data
            # If recursive search fails, use answer as-is
        
        if not isinstance(data, dict):
            return False, f"Answer data is not a dict (got {type(data).__name__})"
        
        def _is_meaningful_value(value: Any, path: str = "") -> bool:
            """Recursively check if value contains meaningful content."""
            if value is None:
                return False
            if isinstance(value, dict):
                if not value:  # Empty dict
                    return False
                # Check if dict has at least one meaningful value
                return any(_is_meaningful_value(v, f"{path}.{k}") for k, v in value.items())
            if isinstance(value, list):
                if not value:  # Empty list
                    return False
                # Check if list has at least one meaningful item
                return any(_is_meaningful_value(item, f"{path}[{i}]") for i, item in enumerate(value))
            if isinstance(value, str):
                # Empty string or only whitespace is not meaningful
                if not value.strip():
                    return False
                # Common non-meaningful values
                if value.strip().lower() in ("", "none", "n/a", "null", "[]", "{}"):
                    return False
                # String "0" might be meaningful for some fields, but "0" as total_count is suspicious
                # We'll be lenient here and allow it
                return True
            if isinstance(value, (int, float)):
                # Zero might be meaningful in some contexts, but we'll flag it as potentially empty
                # However, for counts/totals, zero usually means no data found
                # We'll allow zero but it's a warning sign
                return True
            if isinstance(value, bool):
                return True
            return True
        
        # Check if the root object has meaningful content
        if not data:
            return False, "Answer is an empty dict"
        
        # Check if at least one field has meaningful content
        has_meaningful = False
        for key, value in data.items():
            if _is_meaningful_value(value, key):
                has_meaningful = True
                break
        
        if not has_meaningful:
            return False, "Answer contains no meaningful values (all fields are empty, None, or default values)"
        
        # Additional check: if answer contains common "empty result" patterns, reject it
        # Pattern 1: All arrays are empty
        all_arrays_empty = True
        for key, value in data.items():
            if isinstance(value, list):
                if len(value) > 0:
                    all_arrays_empty = False
                    break
            elif isinstance(value, dict):
                # Check nested arrays
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, list) and len(nested_value) > 0:
                        all_arrays_empty = False
                        break
                if not all_arrays_empty:
                    break
        
        if all_arrays_empty and any(isinstance(v, (list, dict)) for v in data.values()):
            # If we have array/dict fields but they're all empty, it's not meaningful
            array_fields = [k for k, v in data.items() if isinstance(v, (list, dict))]
            return False, f"All array/dict fields are empty: {array_fields}"
        
        # Pattern 2: All counts are zero (common pattern: total_count: "0", hotels: [])
        count_fields = [k for k in data.keys() if "count" in k.lower() or "total" in k.lower() or "num" in k.lower()]
        if count_fields:
            all_counts_zero = True
            for field in count_fields:
                val = data.get(field)
                if isinstance(val, (int, float)) and val != 0:
                    all_counts_zero = False
                    break
                if isinstance(val, str):
                    try:
                        if int(val) != 0:
                            all_counts_zero = False
                            break
                    except (ValueError, TypeError):
                        pass
            
            if all_counts_zero and all(isinstance(data.get(k), (list, dict)) and len(data.get(k)) == 0 for k in data.keys() if k not in count_fields):
                # All counts are zero and all data structures are empty
                return False, f"All count fields are zero ({count_fields}) and all data structures are empty"
        
        return True, ""
