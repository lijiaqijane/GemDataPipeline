from pathlib import Path
import sys


def count_task_dirs(base_dir: str) -> None:
    base_path = Path(base_dir)

    if not base_path.is_dir():
        print(f"错误: 目录不存在或不是目录: {base_dir}")
        sys.exit(1)

    task_dirs = [
        path for path in base_path.iterdir()
        if path.is_dir() and path.name.startswith("task-")
    ]

    submitted_count = sum(
        1
        for task_dir in task_dirs
        if (task_dir / "_sandbox" / "submitted_result.json").is_file()
    )

    print(f"task-* 文件夹数量: {len(task_dirs)}")
    print(f"_sandbox 里有 submitted_result.json 的文件夹数量: {submitted_count}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python count_task_submitted.py <目录路径>")
        sys.exit(1)

    count_task_dirs(sys.argv[1])
