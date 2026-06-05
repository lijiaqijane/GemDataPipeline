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
    def _safe_metadata_json(payload: Any, *, max_chars: int = 4000) -> str:
        if payload is None:
            return "{}"

        def _compact(value: Any, *, max_items: int, max_str: int) -> Any:
            if isinstance(value, dict):
                return {k: _compact(v, max_items=max_items, max_str=max_str) for k, v in value.items()}
            if isinstance(value, list):
                return [_compact(v, max_items=max_items, max_str=max_str) for v in value[:max_items]]
            if isinstance(value, str) and len(value) > max_str:
                return value[:max_str] + "..."
            return value

        try:
            text = json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            return "{}"
        if len(text) <= max_chars:
            return text
        for max_items, max_str in ((6, 200), (4, 120), (2, 80), (1, 60)):
            compact = _compact(payload, max_items=max_items, max_str=max_str)
            text = json.dumps(compact, ensure_ascii=False, default=str)
            if len(text) <= max_chars:
                return text
        return json.dumps({"truncated": True}, ensure_ascii=False)

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
    def _extract_file_paths_from_error(error_message: str, available_files: list[str]) -> list[str]:
        """Extract file paths from error message that match available files.
        
        Args:
            error_message: Error message text
            available_files: List of available file paths from data_profile
            
        Returns:
            List of file paths found in error message that match available files
        """
        if not error_message or not available_files:
            return []
        
        found_paths: set[str] = set()
        error_lower = error_message.lower()
        
        # Try to find file paths in error message
        for file_path in available_files:
            # Check if file path or filename appears in error message
            file_name = file_path.split("/")[-1] if "/" in file_path else file_path
            if file_path in error_message or file_name in error_lower:
                found_paths.add(file_path)
        
        return sorted(found_paths)
    
    def _extract_file_paths_from_code(self, code: str, available_files: list[str]) -> list[str]:
        """Extract file paths from solution/verification code that match available files.
        
        Args:
            code: Solution or verification code
            available_files: List of available file paths from data_profile
            
        Returns:
            List of file paths found in code that match available files
        """
        if not code or not available_files:
            return []
        
        import re
        
        found_paths: set[str] = set()
        
        # Step 1: Extract string literals from code that might be file paths
        # Look for patterns like: "data/file.json", 'data/file.json', BASE_DIR / "data" / "file.json"
        for file_path in available_files:
            file_name = file_path.split("/")[-1] if "/" in file_path else file_path
            
            # Check for file path in string literals
            if file_path in code or file_name in code:
                # More precise: check if it's actually in a string literal or path expression
                # Look for patterns: "data/file.json", 'data/file.json', / "file.json", / 'file.json'
                patterns = [
                    rf'["\']{re.escape(file_path)}["\']',  # "data/file.json"
                    rf'["\']{re.escape(file_name)}["\']',  # "file.json"
                    rf'/[\s]*["\']{re.escape(file_name)}["\']',  # / "file.json"
                    rf'[\s]+["\']{re.escape(file_name)}["\']',  #  "file.json"
                ]
                for pattern in patterns:
                    if re.search(pattern, code):
                        found_paths.add(file_path)
                        break
        
        # Step 2: Extract tool names from code and match with file prefixes using the same logic as tool synthesis
        # Tool names are generated using _tool_prefix_from_path, so we need to use the same logic
        tool_calls = CodeValidator.extract_tool_calls(code)
        for tool_name in tool_calls:
            # Extract prefix from tool name (remove get_ or list_ prefix)
            if tool_name.startswith("get_"):
                tool_prefix = tool_name[4:]  # Remove "get_"
            elif tool_name.startswith("list_"):
                tool_prefix = tool_name[5:]  # Remove "list_"
            else:
                continue  # Skip tools that don't follow the naming convention
            
            # For each available file, extract its prefix using the same logic as _tool_prefix_from_path
            for file_path in available_files:
                file_prefix = self._tool_prefix_from_path(file_path)
                # Compare prefixes (normalize for comparison)
                # The prefix matching should be flexible - check if tool_prefix matches file_prefix
                if tool_prefix == file_prefix:
                    found_paths.add(file_path)
                # Also check if tool_prefix is a substring of file_prefix or vice versa
                # (to handle cases where the prefix extraction might differ slightly)
                elif tool_prefix in file_prefix or file_prefix in tool_prefix:
                    # Additional check: ensure they share significant common parts
                    tool_tokens = set(tool_prefix.split("_"))
                    file_tokens = set(file_prefix.split("_"))
                    # If they share at least 2 common tokens, consider it a match
                    if len(tool_tokens & file_tokens) >= 2:
                        found_paths.add(file_path)
        
        return sorted(found_paths)

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
        # Check if error is from verification or solution based on error message
        if run.error:
            error_lower = (run.error or "").lower()
            # Check if error explicitly mentions verification/verify in traceback
            if any(key in error_lower for key in ["in verify", "verify(", "def verify", "verification_src"]):
                target = "verification"
            # Check if error explicitly mentions solution/solve in traceback
            elif any(key in error_lower for key in ["in solve", "solve(", "def solve", "solution_src", "line 187", "line 188"]):
                # Common solution error patterns (line 187/188 are typical solution code lines)
                target = "solution"
            else:
                # Default: if we have both error and verification_error,
                # check which one came first
                # If answer is None, error likely happened before solve() completed
                # This could be solution exec error or solve() runtime error
                # If answer is not None but verified is None, error likely happened in verify()
                # But we can't check answer here, so default to solution for runtime errors
                target = "solution"
        elif self._verifier_is_broken(run):
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
            # Find submit_result tool (could be submit_result or submit_result_difficulty_X)
            submit_tools = {name for name in allowed_tools if name.startswith("submit_result")}
            
            # Extract tool calls from solution and verification
            called_tools = CodeValidator.extract_tool_calls(package.solution)
            verify_called_tools = CodeValidator.extract_tool_calls(package.verification or "")
            
            # If solution calls a submit_result tool that's not in allowed_tools, add it
            # This can happen if submit_result was added to tools.py but not to tool_set
            called_submit_tools = {name for name in called_tools if name.startswith("submit_result")}
            if called_submit_tools and not called_submit_tools.issubset(allowed_tools):
                # Solution calls a submit_result tool that's not in tool_set
                # This is likely submit_result_difficulty_X that exists in tools.py but wasn't added to tool_set
                # We should allow it since it's a system tool
                missing_submit = called_submit_tools - allowed_tools
                self.logger.warning(
                    "Solution calls submit_result tool(s) not in tool_set: %s. Allowing them as system tools.",
                    sorted(missing_submit)
                )
                # Add missing submit_result tools to allowed_tools for validation
                allowed_tools = allowed_tools | missing_submit
                submit_tools = submit_tools | missing_submit
            
            # Check verification tools
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
            
            # Check solution tools
            called_data_tools = {name for name in called_tools if name not in submit_tools}
            if len(called_data_tools) < 2:
                last_error = (
                    "solution must call at least 2 different tools; "
                    f"called={sorted(called_tools)} allowed={sorted(allowed_tools)}"
                )
                package = self._repair_package(request, package, ctx, error=last_error, records=records)
                continue
            missing_tools = called_tools - allowed_tools
            # Filter out submit_result tools from missing_tools (they are system tools)
            # If submit_result tools are missing, they should have been added to allowed_tools above
            missing_submit_in_missing = {name for name in missing_tools if name.startswith("submit_result")}
            if missing_submit_in_missing:
                # This shouldn't happen if the logic above worked correctly, but log a warning
                self.logger.warning(
                    "Solution calls submit_result tool(s) that are still missing from allowed_tools: %s. "
                    "This may indicate a bug in tool_set management.",
                    sorted(missing_submit_in_missing)
                )
                # Allow them anyway since they're system tools
                missing_tools = missing_tools - missing_submit_in_missing
            
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
            # tools_code is stored in tools.py file, no need to save in metadata
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
            # Find submit_result tools to exclude from data tools count
            submit_tools_in_allowed = {name for name in allowed_tools if name.startswith("submit_result")}
            used_data_tools = {name for name in used_tools if name not in submit_tools_in_allowed}
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
                                    "data_profile": self._safe_metadata_json(data_profile or {}),
                                    "tool_selftest": self._safe_metadata_json(refreshed_selftest),
                                }
                            }
                        )
                if not augmented_once:
                    # Determine which files need augmentation based on errors or code analysis
                    entries = self._iter_data_entries(data_profile or {})
                    if not entries:
                        # No data files available, skip augmentation
                        pass
                    else:
                        available_file_paths = [entry["path"] for entry in entries]
                        target_files: list[str] = []
                        
                        # Step 1: Try to extract file paths from error messages
                        error_text = f"{last_error or ''} {run.error or ''} {run.verification_error or ''}".strip()
                        if error_text:
                            error_files = self._extract_file_paths_from_error(error_text, available_file_paths)
                            if error_files:
                                target_files.extend(error_files)
                        
                        # Step 2: If no files found from errors, analyze solution/verification code
                        if not target_files:
                            solution_files: list[str] = []
                            if package.solution:
                                solution_files = self._extract_file_paths_from_code(package.solution, available_file_paths)
                            verify_files: list[str] = []
                            if package.verification:
                                verify_files = self._extract_file_paths_from_code(package.verification, available_file_paths)
                            # Combine files from solution and verification, prioritizing solution
                            target_files.extend(solution_files)
                            for f in verify_files:
                                if f not in target_files:
                                    target_files.append(f)
                        
                        # Step 3: If still no files found, use first available file as fallback
                        if not target_files:
                            target_files = [available_file_paths[0]] if available_file_paths else []
                        
                        # Remove duplicates while preserving order
                        seen: set[str] = set()
                        unique_target_files: list[str] = []
                        for f in target_files:
                            if f not in seen:
                                seen.add(f)
                                unique_target_files.append(f)
                        
                        # Augment tools for each target file separately
                        current_specs = list(package.task.tool_set)
                        # Preserve submit_result tools (they are system tools and should always be kept)
                        submit_result_specs = [s for s in current_specs if s.name.startswith("submit_result")]
                        all_new_code_parts: list[str] = []
                        augmented_any = False
                        
                        for target_file_path in unique_target_files:
                            new_specs, new_code, augmented = self._augment_toolset(
                                request.topic or package.task.task_title,
                                records,
                                current_specs,  # Use accumulated specs (including previously augmented)
                                data_profile or {},
                                ctx,
                                sandbox,
                                target_file_path=target_file_path,
                            )
                            if augmented:
                                augmented_any = True
                                current_specs = new_specs  # Update specs for next iteration
                                # Ensure submit_result tools are preserved after each augmentation
                                current_spec_names = {s.name for s in current_specs}
                                for submit_spec in submit_result_specs:
                                    if submit_spec.name not in current_spec_names:
                                        current_specs.append(submit_spec)
                                        self.logger.info(
                                            f"Preserved submit_result tool {submit_spec.name} in tool_set after augmentation"
                                        )
                                if new_code and new_code not in "\n".join(all_new_code_parts):
                                    all_new_code_parts.append(new_code)
                        
                        if augmented_any:
                            augmented_once = True
                            # Ensure submit_result tools are preserved in tool_set
                            current_spec_names = {s.name for s in current_specs}
                            for submit_spec in submit_result_specs:
                                if submit_spec.name not in current_spec_names:
                                    current_specs.append(submit_spec)
                                    self.logger.info(f"Preserved submit_result tool {submit_spec.name} in tool_set after augmentation")
                            
                            # Read current tools.py and append new code
                            tools_path = sandbox.sandbox_dir / "tools.py"
                            current_tools_code = ""
                            if tools_path.exists():
                                current_tools_code = tools_path.read_text(encoding="utf-8")
                            combined_new_code = "\n\n".join(all_new_code_parts).strip()
                            if combined_new_code and combined_new_code not in current_tools_code:
                                updated_tools_code = (current_tools_code + "\n\n" + combined_new_code).strip()
                                tools_path.write_text(updated_tools_code, encoding="utf-8")
                            try:
                                new_selftest = self._self_test_tools(current_specs, sandbox, request.topic, ctx)
                            except Exception:
                                new_selftest = tool_selftest
                            package = package.copy(
                                update={
                                    "task": package.task.copy(update={"tool_set": current_specs}),
                                    "metadata": {
                                        **(package.metadata or {}),
                                        "tool_selftest": self._safe_metadata_json(new_selftest),
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
                        # Don't delete submitted_result.json - it needs to be persisted
                        if rel == "submitted_result.json":
                            continue
                        try:
                            target = sandbox.sandbox_dir / rel
                            if target.exists():
                                target.unlink(missing_ok=True)
                        except Exception:
                            continue
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
                # Prioritize solution error if both exist, since verification error may be secondary
                if run.error:
                    last_error = run.error
                else:
                    last_error = run.verification_error or "unknown_error"
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
                        "data_profile": self._safe_metadata_json(metadata_profile),
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
        # Find submit_result tool (could be submit_result or submit_result_difficulty_X)
        submit_tools = {name for name in allowed if name.startswith("submit_result")}
        called = CodeValidator.extract_tool_calls(package.solution)
        
        # If solution calls a submit_result tool that's not in allowed, add it
        # This can happen if submit_result was added to tools.py but not to tool_set
        called_submit_tools = {name for name in called if name.startswith("submit_result")}
        if called_submit_tools and not called_submit_tools.issubset(allowed):
            # Allow submit_result tools even if not in tool_specs (they are system tools)
            missing_submit = called_submit_tools - allowed
            allowed = allowed | missing_submit
            submit_tools = submit_tools | missing_submit
        
        called_data = {name for name in called if name not in submit_tools}

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
        repaired_data = {name for name in repaired_called if name not in submit_tools}
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
        """Check if answer has any content (not None and not empty).
        
        Note: This function only checks if there's content, not if it's meaningful or correct.
        Type validation is done by submit_result function.
        
        Returns:
            tuple[bool, str]: (has_content, error_message)
            - has_content: True if answer has any content, False otherwise
            - error_message: Error message if no content, empty string otherwise
        """
        if answer is None:
            return False, "Answer is None"
        
        # Try to unwrap if answer is wrapped (e.g., {'result': [...]} or {'data': {...}})
        data = answer
        if isinstance(answer, dict):
            # Try common wrapper keys
            for key in ['result', 'data', 'answer', 'content']:
                if key in answer:
                    wrapped = answer[key]
                    if wrapped is not None:
                        data = wrapped
                        break
        
        # Check if data has any content
        if isinstance(data, dict):
            if not data:
                return False, "Answer is an empty dict"
        elif isinstance(data, list):
            if not data:
                return False, "Answer is an empty list"
        elif isinstance(data, str):
            if not data.strip():
                return False, "Answer is an empty string"
        # For other types (int, float, bool), any value is considered content
        
        return True, ""
