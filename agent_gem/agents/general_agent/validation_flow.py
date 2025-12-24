from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Optional, TYPE_CHECKING

from agent_gem.core.task_schema import TaskPackage, ToolSpec
from agent_gem.core.validation import CodeValidator
from agent_gem.sandbox import SandboxExecutor, SandboxFusionExecutor

from ..base import TaskContext

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
        sandbox: SandboxExecutor,
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
        allow_permissive = os.getenv("ALLOW_PERMISSIVE_VERIFIER", "0").strip().lower() in {"1", "true", "yes"}

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
            verify_data_tools = sorted(set(verify_called_tools) - {submit_tool})
            if not verify_data_tools:
                last_error = "verification must call at least one data tool to cross-check outputs"
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
            difficulty = int(getattr(package.task, "difficulty_level", 1) or 1)
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
                package = self._repair_package(
                    request,
                    package,
                    ctx,
                    error=last_error,
                    records=records,
                    repair_target=target,
                )
                continue
            if run.answer is not None:
                fmt_ok, fmt_err = self._validate_answer_format(
                    package.task.submit_result_format, run.answer
                )
                if not fmt_ok:
                    last_error = f"answer_format_mismatch: {fmt_err}"
                    package = self._repair_package(request, package, ctx, error=last_error, records=records)
                    continue
            used_data_tools = {name for name in used_tools if name != submit_tool}
            if len(used_data_tools) < 2:
                last_error = (
                    f"runtime tool calls insufficient; used={sorted(used_tools)} count={tool_call_count}"
                )
                package = self._repair_package(request, package, ctx, error=last_error, records=records)
                continue
            if allow_permissive and run.answer is not None and run.verified is False:
                meta = package.metadata or {}
                if meta.get("permissive_verifier") != "true":
                    package = package.copy(
                        update={
                            "verification": self._build_permissive_verifier(
                                package.task.submit_result_format
                            ),
                            "metadata": {**meta, "permissive_verifier": "true"},
                        }
                    )
                    # Fold permissive reruns into the same difficulty folder (user preference).
                    run = sandbox.run_task(package, run_group=group_prefix)
                    ctx.add_step(
                        {
                            "type": "verifier_relaxed",
                            "verified": run.verified,
                            "verification_error": run.verification_error,
                        }
                    )
                    if run.verified is True:
                        continue
            if (not run.answer or run.verified is False) and attempt < request.max_validation_rounds:
                try:
                    data_profile = json.loads((package.metadata or {}).get("data_profile", "{}"))
                    tool_selftest = json.loads((package.metadata or {}).get("tool_selftest", "{}"))
                    metadata_profile = data_profile
                except Exception:
                    data_profile, tool_selftest = {}, {}
                if not tool_selftest or all(
                    not v.get("fields") for v in tool_selftest.values() if isinstance(v, dict)
                ):
                    new_specs, new_code = self._synthesize_task_tools(
                        request.topic or package.task.task_title,
                        records,
                        ctx,
                        sandbox,
                        data_profile or {},
                    )
                    self._register_task_tools(new_specs, sandbox, ctx, tools_code=new_code)
                    new_selftest = self._self_test_tools(new_specs, sandbox, request.topic, ctx)
                    package = self._propose_task(
                        package.task.task_id,
                        request,
                        records,
                        new_specs,
                        ctx,
                        tools_code=new_code,
                        data_profile=data_profile or {},
                        tool_selftest=new_selftest,
                    )
                    package = package.copy(
                        update={
                            "metadata": {
                                **(package.metadata or {}),
                                "data_profile": json.dumps(data_profile, ensure_ascii=False)[:4000],
                                "tool_selftest": json.dumps(new_selftest, ensure_ascii=False)[:4000],
                                "tools_code": new_code,
                            }
                        }
                    )
                    continue
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
                if run.answer is None or run.verified is None:
                    last_error = run.error or run.verification_error or "missing_answer_or_verdict"
                    package = self._repair_package(request, package, ctx, error=last_error, records=records)
                    continue
            if run.verified is True:
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
                            package = self._repair_package(request, package, ctx, error=last_error, records=records)
                            continue
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
    def _validate_answer_format(fmt: Any, answer: Any) -> tuple[bool, str]:
        """Best-effort validation of answer against submit_result_format."""
        if fmt is None:
            return True, ""
        if isinstance(fmt, str):
            try:
                fmt = json.loads(fmt)
            except Exception:
                return True, ""
        # JSON schema-like object
        if isinstance(fmt, dict) and fmt.get("type") in {"object", "array"}:
            if fmt.get("type") == "object":
                if not isinstance(answer, dict):
                    return False, "answer must be object"
                required = fmt.get("required") or []
                if isinstance(required, list):
                    missing = [k for k in required if k not in answer]
                    if missing:
                        return False, f"missing keys: {missing}"
                props = fmt.get("properties") if isinstance(fmt.get("properties"), dict) else {}
                for key, spec in props.items():
                    if key not in answer:
                        continue
                    val = answer.get(key)
                    if isinstance(spec, dict) and "type" in spec:
                        expected = spec.get("type")
                        if expected == "array" and not isinstance(val, list):
                            return False, f"key '{key}' must be list"
                        if expected == "object" and not isinstance(val, dict):
                            return False, f"key '{key}' must be object"
                        if expected == "string" and not isinstance(val, str):
                            return False, f"key '{key}' must be string"
                return True, ""
            if fmt.get("type") == "array":
                return (isinstance(answer, list), "answer must be list")
        # Example object format (keys act as contract)
        if isinstance(fmt, dict):
            if not isinstance(answer, dict):
                return False, "answer must be object"
            missing = [k for k in fmt.keys() if k not in answer]
            if missing:
                return False, f"missing keys: {missing}"
            for key, tmpl in fmt.items():
                val = answer.get(key)
                if isinstance(tmpl, list) and not isinstance(val, list):
                    return False, f"key '{key}' must be list"
                if isinstance(tmpl, dict) and not isinstance(val, dict):
                    return False, f"key '{key}' must be object"
                if isinstance(tmpl, str) and not isinstance(val, str):
                    return False, f"key '{key}' must be string"
            return True, ""
        # List format
        if isinstance(fmt, list):
            if not isinstance(answer, list):
                return False, "answer must be list"
            if fmt and isinstance(fmt[0], dict):
                required_keys = set(fmt[0].keys())
                for item in answer:
                    if isinstance(item, dict) and required_keys - set(item.keys()):
                        return False, f"list item missing keys: {sorted(required_keys)}"
            return True, ""
        return True, ""
