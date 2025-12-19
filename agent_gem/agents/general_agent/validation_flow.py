from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Optional, TYPE_CHECKING

from agent_gem.core.task_schema import TaskPackage, ToolSpec
from agent_gem.core.validation import CodeValidator
from agent_gem.sandbox import SandboxExecutor

from ..base import TaskContext

if TYPE_CHECKING:  # pragma: no cover
    from agent_gem.generator import GenerationRequest  # noqa: F401


class ValidationMixin:
    """Validation, repair, and quality gates for generated tasks."""

    def _ensure_valid(
        self,
        request: "GenerationRequest",
        package: TaskPackage,
        ctx: TaskContext,
        sandbox: SandboxExecutor,
        records: list[dict[str, Any]],
        *,
        setup_snapshot: dict[str, str] | None = None,
        setup_bundle: dict[str, str] | None = None,
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
        for attempt in range(1, request.max_validation_rounds + 1):
            allowed_tools = {spec.name for spec in package.task.tool_set}
            called_tools = CodeValidator.extract_tool_calls(package.solution)
            if len(called_tools) < 2:
                called_tools = self._extract_tool_calls_fallback(package.solution, list(package.task.tool_set))
            if len(called_tools) < 2:
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

            pre_snapshot = setup_snapshot or sandbox.snapshot_fs()

            run_start = time.time()
            run = sandbox.run_task(package)
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
            if setup_bundle:
                meta_update |= {
                    "setup_returncode": setup_bundle.get("returncode", ""),
                    "setup_stdout_preview": (setup_bundle.get("stdout", "") or "")[:200],
                    "setup_stderr_preview": (setup_bundle.get("stderr", "") or "")[:200],
                    "setup_code": setup_bundle.get("setup_code", "")[:2000],
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
                }
            )
            if verifier_weak:
                last_error = run.verification_error or "verification returned no actionable signal (no score/details)"
                package = self._repair_package(request, package, ctx, error=last_error, records=records)
                continue
            if len(used_tools) < 2:
                last_error = (
                    f"runtime tool calls insufficient; used={sorted(used_tools)} count={tool_call_count}"
                )
                package = self._repair_package(request, package, ctx, error=last_error, records=records)
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
            if run.verified is True:
                if new_files:
                    for rel in new_files:
                        try:
                            target = sandbox.sandbox_dir / rel
                            if target.exists():
                                target.unlink(missing_ok=True)
                        except Exception:
                            continue
                    try:
                        negative = sandbox.run_task(package)
                        if negative.verified:
                            last_error = "negative_check_failed"
                            package = self._repair_package(request, package, ctx, error=last_error, records=records)
                            continue
                    except Exception:
                        pass
                return package

            if run.error or run.verification_error:
                last_error = run.error or run.verification_error or "unknown_error"
                package = self._repair_package(request, package, ctx, error=last_error, records=records)
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
        records: Optional[list[dict[str, Any]]] = None,
    ) -> TaskPackage:
        prompt = (
            "Repair the task so that solve(tools) is non-trivial and calls at least 2 allowed tools.\n"
            "Return JSON with keys: task_content, submit_result_format, solution, verification.\n"
            f"Allowed tools: {[spec.name for spec in package.task.tool_set]}\n"
            f"Error to fix: {error or ''}\n"
            f"Existing solution (truncated): {(package.solution or '')[:800]}\n"
        )
        max_tokens = getattr(request, "max_tokens", 10000)
        raw = self.llm.simple_complete(prompt, temperature=0.55, max_tokens=max_tokens)
        ctx.add_step({"type": "repair_attempt", "error": error, "content": raw})
        repaired = self._extract_json(raw) or {}

        task_content = repaired.get("task_content") or package.task.task_content
        submit_result_format = repaired.get("submit_result_format") or package.task.submit_result_format
        solution = repaired.get("solution") or package.solution
        verification = repaired.get("verification") or package.verification

        updated = package.copy(
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
                    **(package.metadata or {}),
                    "repair_error": error or "",
                    "repaired": "true",
                },
            }
        )
        return updated

    def _ensure_substantive_task(
        self,
        tool_specs: list[ToolSpec],
        package: TaskPackage,
        ctx: TaskContext,
        request: "GenerationRequest | None" = None,
    ) -> TaskPackage:
        allowed = {spec.name for spec in tool_specs}
        called = CodeValidator.extract_tool_calls(package.solution)
        if len(called) < 2:
            called = self._extract_tool_calls_fallback(package.solution, list(tool_specs))

        if called and called.issubset(allowed) and len(called) >= 2:
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
        if repaired_called and repaired_called.issubset(allowed) and len(repaired_called) >= 2:
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
    def _extract_tool_calls_fallback(code: str, tool_specs: list[ToolSpec]) -> set[str]:
        """String-based tool-call extractor when AST-based extraction fails."""
        if not code or not isinstance(code, str):
            return set()
        called: set[str] = set()
        for spec in tool_specs:
            name = spec.name
            if not name:
                continue
            patterns = [f"tools['{name}'](", f'tools["{name}"](', f"tools.{name}("]
            if any(p in code for p in patterns):
                called.add(name)
        return called

    @staticmethod
    def _is_trivial_answer(answer: Any, package: TaskPackage) -> bool:
        """Heuristic to detect trivial fallback-style answers."""
        fmt = package.task.submit_result_format

        if isinstance(answer, dict) and set(answer.keys()) == {"combined", "counts"}:
            combined = answer.get("combined", [])
            if isinstance(combined, list):
                if len(combined) == 0 or all(
                    (isinstance(item, list) and len(item) == 0) or (not isinstance(item, list) and not item)
                    for item in combined
                ):
                    return True
            return True

        if answer is None or (isinstance(answer, dict) and len(answer) == 0):
            return True

        if isinstance(fmt, dict) and fmt.get("type") == "object":
            props = fmt.get("properties") or {}
            if isinstance(props, dict) and props:
                expected_keys = set(props.keys())
                if isinstance(answer, dict):
                    if not (set(answer.keys()) & expected_keys):
                        return True
                    actual_keys = set(answer.keys()) & expected_keys
                    if actual_keys:
                        non_empty_values = [
                            v for k, v in answer.items() if k in actual_keys and v not in (None, [], {}, "")
                        ]
                        if len(non_empty_values) == 0:
                            return True

        if isinstance(answer, list) and len(answer) == 0:
            return True

        return False
