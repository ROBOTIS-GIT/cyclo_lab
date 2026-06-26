# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Kiwoong Park

from __future__ import annotations

import os
from pathlib import Path
import subprocess

_GIT_STATE_TXT_FILENAME = "git_state.txt"
_GIT_DIFF_PATCH_FILENAME = "git_diff.patch"
_CONTAINER_REPO_ROOT = "/workspace/cyclo_lab"


def _run_git(repo_path: Path, *args: str) -> str:
    env = os.environ.copy()
    env["GIT_DISCOVERY_ACROSS_FILESYSTEM"] = "1"
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={_CONTAINER_REPO_ROOT}", "-C", str(repo_path), *args],
        env=env,
        stderr=subprocess.STDOUT,
        text=True,
    ).strip()


def _repo_root(repo_path: str | os.PathLike[str]) -> Path:
    start_path = Path(repo_path).resolve()
    if start_path.is_file():
        start_path = start_path.parent
    return Path(_run_git(start_path, "rev-parse", "--show-toplevel"))


def _format_git_state(repo_path: Path) -> str:
    branch = _run_git(repo_path, "branch", "--show-current") or "HEAD"
    commit_log = _run_git(repo_path, "log", "-1", "--decorate", "--date=iso")
    return f"branch: {branch}\n\n{commit_log}\n"


def export_git_state(log_dir: str, repo_path: str | os.PathLike[str]) -> None:
    """Export branch, commit hash, and diff for the training run."""
    output_dir = Path(log_dir) / "git"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / _GIT_STATE_TXT_FILENAME
    diff_path = output_dir / _GIT_DIFF_PATCH_FILENAME

    try:
        repo_path = _repo_root(repo_path)
        state_path.write_text(_format_git_state(repo_path))
        diff_path.write_text(_run_git(repo_path, "diff", "--binary", "HEAD", "--submodule=diff"))
    except (subprocess.CalledProcessError, RuntimeError) as error:
        error_message = error.output.strip() if isinstance(error, subprocess.CalledProcessError) else str(error)
        state_path.write_text(f"Git state export failed.\n\n{error_message}\n")
        diff_path.write_text("")
        print(f"[WARNING]: Skipping Git state export: {error_message}")
        return

    print(f"[INFO]: Exported Git state to {state_path} and {diff_path}")
