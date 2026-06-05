"""
Utilities adapter for code_agent module.

This module provides utility functions adapted from app.utils for use in code_agent.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
from os.path import join as pjoin
from subprocess import CalledProcessError
import shutil
import logging

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def cd(newdir: str):
    """
    Context manager for changing the current working directory.
    
    Args:
        newdir: Path to the new directory
    """
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def run_command(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """
    Run a command in the shell.
    
    Args:
        cmd: Command to run as a list of strings
        **kwargs: Additional arguments to pass to subprocess.run
    
    Returns:
        CompletedProcess object
    """
    try:
        cp = subprocess.run(cmd, check=True, **kwargs)
        return cp
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running command: {cmd}, {e}")
        raise e


def is_git_repo() -> bool:
    """
    Check if the current directory is a git repo.
    
    Returns:
        True if current directory is a git repository
    """
    git_dir = ".git"
    return os.path.isdir(git_dir)


def get_current_commit_hash() -> str:
    """
    Get the current commit hash.
    
    Returns:
        Current commit hash as string
    """
    command = ["git", "rev-parse", "HEAD"]
    cp = subprocess.run(command, text=True, capture_output=True)
    try:
        cp.check_returncode()
        return cp.stdout.strip()
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to get SHA-1 of HEAD: {cp.stderr}") from e


def repo_reset_and_clean_checkout(commit_hash: str) -> None:
    """
    Run commands to reset repo to the original commit state.
    Cleans both the uncommited changes and the untracked files, and submodule changes.
    Assumption: The current directory is the git repository.
    """
    # NOTE: do these before `git reset`. This is because some of the removed files below
    # may actually be in version control. So even if we deleted such files here, they
    # will be brought back by `git reset`.
    # Clean files that might be in .gitignore, but could have been created by previous runs
    if os.path.exists(".coverage"):
        os.remove(".coverage")
    if os.path.exists("tests/.coveragerc"):
        os.remove("tests/.coveragerc")
    other_cov_files = glob.glob(".coverage.TSS.*", recursive=True)
    for f in other_cov_files:
        os.remove(f)

    reset_cmd = ["git", "reset", "--hard", commit_hash]
    clean_cmd = ["git", "clean", "-fd"]
    checkout_cmd = ["git", "checkout", commit_hash]
    run_command(reset_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run_command(clean_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # need to checkout before submodule init. Otherwise submodule may init to another version
    run_command(checkout_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # this is a fail-safe combo to reset any changes to the submodule: first unbind all submodules
    # and then make a fresh checkout of them.
    # Reference: https://stackoverflow.com/questions/10906554/how-do-i-revert-my-changes-to-a-git-submodule
    submodule_unbind_cmd = ["git", "submodule", "deinit", "-f", "."]
    submodule_init_cmd = ["git", "submodule", "update", "--init"]
    run_command(
        submodule_unbind_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    run_command(
        submodule_init_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def repo_commit_current_changes() -> None:
    """
    Commit the current active changes so that it's safer to do git reset later on.
    Use case: for storing the changes made in pre_install and test_patch in a commit.
    Assumption: The current directory is the git repository.
    """

    # Fallback implementation
    add_all_cmd = ["git", "add", "."]
    commit_cmd = ["git", "commit", "-m", "Temporary commit for storing changes"]
    run_command(add_all_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run_command(commit_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def create_dir_if_not_exists(dir_path: str) -> None:
    """
    Create a directory if it does not exist.
    
    Args:
        dir_path: Path to the directory
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def clone_repo(clone_link: str, cloned_dir: str):
    """
    Clone a repo to dest_dir.

    Returns:
        - path to the newly cloned directory.
    """
    dest_dir = os.path.dirname(cloned_dir)  # 获取目录路径
    cloned_name = os.path.basename(cloned_dir)
    clone_cmd = ["git", "clone", clone_link, cloned_name]
    create_dir_if_not_exists(dest_dir)
    with cd(dest_dir):
        run_command(clone_cmd)

def clone_repo_and_checkout(
    clone_link: str, commit_hash: str, cloned_dir: str,
    # dest_dir: str, cloned_name: str
):
    """
    Clone a repo to dest_dir, and checkout to commit `commit_hash`.

    Returns:
        - path to the newly cloned directory.
    """
    # cloned_dir = 
    if clone_link.endswith('.git'):
        clone_repo(clone_link, cloned_dir)
    else:
        if os.path.isdir(cloned_dir):
            shutil.rmtree(cloned_dir)
        shutil.copytree(clone_link, cloned_dir)
    if commit_hash != "":
        checkout_cmd = ["git", "checkout", commit_hash]
        with cd(cloned_dir):
            run_command(checkout_cmd)
