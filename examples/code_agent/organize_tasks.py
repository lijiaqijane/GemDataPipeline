#!/usr/bin/env python3
"""
Organize completed tasks from results directory into a single JSON file.

This script reads completed tasks from taskdb/code_agent/results and organizes
them into a single JSON file containing a list of tasks, each following a
standardized format.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_file_content(file_path: Path) -> Optional[str]:
    """Read file content, return None if file doesn't exist."""
    try:
        if file_path.exists():
            return file_path.read_text(encoding='utf-8')
        return None
    except Exception as e:
        logger.warning(f"Failed to read {file_path}: {e}")
        return None


def load_json_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file, return None if file doesn't exist or invalid."""
    try:
        if file_path.exists():
            return json.loads(file_path.read_text(encoding='utf-8'))
        return None
    except Exception as e:
        logger.warning(f"Failed to load JSON from {file_path}: {e}")
        return None


def load_pr_annotations(annotations_dir: Optional[Path]) -> Dict[str, Any]:
    """
    Load PR annotations from annotation files.
    
    Args:
        annotations_dir: Directory containing PR annotation files
        
    Returns:
        Dictionary mapping instance_id to annotation data
    """
    if not annotations_dir or not annotations_dir.exists():
        return {}
    
    annotations = {}
    
    # Find all *-prs-annotated.jsonl files
    annotation_files = list(annotations_dir.glob("*-prs-annotated.jsonl"))
    
    if not annotation_files:
        logger.warning(f"No *-prs-annotated.jsonl files found in {annotations_dir}")
        return {}
    
    logger.info(f"Loading PR annotations from {len(annotation_files)} files in directory: {annotations_dir}")
    for annotation_file in annotation_files:
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            annotation_data = json.loads(line)
                            instance_id = annotation_data.get('instance_id')
                            if instance_id:
                                if instance_id in annotations:
                                    logger.warning(f"Duplicate annotation for instance_id {instance_id}, keeping first occurrence")
                                else:
                                    annotations[instance_id] = annotation_data
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse annotation at line {line_num} in {annotation_file}: {e}")
        except Exception as e:
            logger.error(f"Error reading annotation file {annotation_file}: {e}")
    
    logger.info(f"Loaded {len(annotations)} PR annotations")
    return annotations


def is_task_completed(task_dir: Path) -> bool:
    """Check if a task is completed by reading status.json."""
    status_file = task_dir / "status.json"
    if not status_file.exists():
        return False
    
    status = load_json_file(status_file)
    if status and status.get("is_finish") is True:
        return True
    return False


def organize_task(task_dir: Path, annotations: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Organize a single task into the standardized JSON format.
    
    Args:
        task_dir: Directory containing task results
        annotations: Dictionary mapping instance_id to PR annotation data
        
    Returns:
        Organized task dictionary, or None if task should be skipped
    """
    task_id = task_dir.name
    
    # Check if task is completed
    if not is_task_completed(task_dir):
        logger.debug(f"Skipping incomplete task: {task_id}")
        return None
    
    logger.info(f"Organizing task: {task_id}")
    
    # Load meta.json
    meta_file = task_dir / "meta.json"
    meta = load_json_file(meta_file)
    if not meta:
        logger.warning(f"No meta.json found for task {task_id}, skipping")
        return None
    
    task_info = meta.get("task_info", {})
    setup_info = meta.get("setup_info", {})
    
    # Extract instance_id
    instance_id = task_info.get("instance_id") or task_id
    
    # Read Dockerfile
    dockerfile_path = task_dir / "Dockerfile"
    dockerfile_content = read_file_content(dockerfile_path)
    
    # Read eval.sh
    eval_script_path = task_dir / "eval.sh"
    eval_script_content = read_file_content(eval_script_path)
    
    # Read cost.json
    cost_file = task_dir / "cost.json"
    cost_data = load_json_file(cost_file) or {}
    
    # Read problem_statement.txt
    problem_statement_path = task_dir / "problem_statement.txt"
    problem_statement = read_file_content(problem_statement_path)
    
    # Read developer_patch.diff
    patch_path = task_dir / "developer_patch.diff"
    patch_content = read_file_content(patch_path)
    
    # Get PR annotation for this task
    pr_annotation = annotations.get(instance_id)
    
    # Build the organized task structure (following the format in agent_gem/test)
    organized_task = {
        "environment": {
            "dockerfile": dockerfile_content or "",
            "eval_script": eval_script_content or "",
            "base_commit": task_info.get("base_commit", ""),
            "repo": task_info.get("repo", "")
        },
        "tools": {},
        "tasks": {
            "instance_id": instance_id,
            "pull_number": task_info.get("pull_number"),
            "issue_numbers": task_info.get("issue_numbers", []),
            "problem_statement": problem_statement or task_info.get("problem_statement", ""),
            "patch": patch_content or task_info.get("patch", ""),
            "test_patch": task_info.get("test_patch", ""),
            "metadata": {
                "created_at": task_info.get("created_at", ""),
                "hints_text": task_info.get("hints_text", ""),
                "language": task_info.get("language", "")
            }
        }
    }
    
    # Add PR annotation if available
    if pr_annotation:
        organized_task["tasks"]["annotations"] = {
            "pr_category": pr_annotation.get("pr_category"),
            "issue_difficulty": pr_annotation.get("issue_difficulty"),
            "issue_description_reasonable": pr_annotation.get("issue_description_reasonable"),
            "gold_patch_solves_issue": pr_annotation.get("gold_patch_solves_issue"),
            "test_patch_designed_for_issue": pr_annotation.get("test_patch_designed_for_issue"),
            "requires_gpu": pr_annotation.get("requires_gpu"),
            "num_gold_files_changed": pr_annotation.get("num_gold_files_changed"),
            "num_test_files_changed": pr_annotation.get("num_test_files_changed"),
            "reasoning": pr_annotation.get("reasoning")
        }
        logger.debug(f"Added PR annotation for task {task_id} (instance_id: {instance_id})")
    else:
        logger.debug(f"No PR annotation found for task {task_id} (instance_id: {instance_id})")
    
    return organized_task


def organize_all_tasks(
    results_dir: Path, 
    output_file: Path,
    annotations_dir: Optional[Path] = None
) -> Dict[str, int]:
    """
    Organize all completed tasks from results directory into a single JSON file.
    
    Args:
        results_dir: Directory containing task results
        output_file: Path to save the combined JSON file
        annotations_dir: Optional directory containing PR annotation files
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        "total_tasks": 0,
        "completed_tasks": 0,
        "organized_tasks": 0,
        "failed_tasks": 0,
        "tasks_with_annotations": 0
    }
    
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return stats
    
    # Load PR annotations if directory is provided
    annotations = {}
    if annotations_dir:
        annotations = load_pr_annotations(annotations_dir)
    
    # Collect all organized tasks
    all_tasks = []
    
    # Iterate through all task directories
    for task_dir in results_dir.iterdir():
        if not task_dir.is_dir():
            continue
        
        stats["total_tasks"] += 1
        
        # Check if task is completed
        if not is_task_completed(task_dir):
            continue
        
        stats["completed_tasks"] += 1
        
        # Organize the task
        organized_task = organize_task(task_dir, annotations)
        if organized_task:
            all_tasks.append(organized_task)
            stats["organized_tasks"] += 1
            
            # Count tasks with annotations
            if organized_task.get("tasks", {}).get("annotations"):
                stats["tasks_with_annotations"] += 1
        else:
            stats["failed_tasks"] += 1
    
    # Save all tasks to a single JSON file
    if all_tasks:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_tasks, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(all_tasks)} organized tasks to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save organized tasks to {output_file}: {e}")
            stats["failed_tasks"] += len(all_tasks)
            stats["organized_tasks"] = 0
    
    return stats


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Organize completed tasks from results into a single JSON file"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="taskdb/code_agent/results",
        help="Directory containing task results (default: taskdb/code_agent/results)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="taskdb/code_agent/tasks/all_tasks.json",
        help="Path to save the combined JSON file (default: taskdb/code_agent/tasks/all_tasks.json)"
    )
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default="taskdb/code_agent/pr_annotations",
        help="Directory containing PR annotation files (default: taskdb/code_agent/pr_annotations). Use empty string to skip loading annotations."
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_file = Path(args.output_file)
    # Handle empty string as None to skip loading annotations
    annotations_dir = Path(args.annotations_dir) if args.annotations_dir and args.annotations_dir.strip() else None
    
    logger.info(f"Organizing tasks from: {results_dir}")
    logger.info(f"Output file: {output_file}")
    if annotations_dir:
        logger.info(f"PR annotations directory: {annotations_dir}")
    else:
        logger.info("PR annotations directory: not specified (skipping annotations)")
    
    stats = organize_all_tasks(results_dir, output_file, annotations_dir)
    
    logger.info("=" * 60)
    logger.info("Organization Summary:")
    logger.info(f"  Total tasks found: {stats['total_tasks']}")
    logger.info(f"  Completed tasks: {stats['completed_tasks']}")
    logger.info(f"  Successfully organized: {stats['organized_tasks']}")
    logger.info(f"  Tasks with PR annotations: {stats['tasks_with_annotations']}")
    logger.info(f"  Failed to organize: {stats['failed_tasks']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
