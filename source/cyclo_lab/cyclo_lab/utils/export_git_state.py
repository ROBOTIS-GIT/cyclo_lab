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


def _run_git(repo_path: Path, *args: str) -> bytes:
    env = os.environ.copy()
    env["GIT_DISCOVERY_ACROSS_FILESYSTEM"] = "1"
    return subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={_CONTAINER_REPO_ROOT}",
            "-C",
            str(repo_path),
            *args,
        ],
        env=env,
        stderr=subprocess.PIPE,
    )


def _repo_root(repo_path: str | os.PathLike[str]) -> Path:
    start_path = Path(repo_path).resolve()
    if start_path.is_file():
        start_path = start_path.parent
    return Path(os.fsdecode(_run_git(start_path, "rev-parse", "--show-toplevel").rstrip(b"\n")))


def _format_git_state(repo_path: Path) -> str:
    branch = _run_git(repo_path, "branch", "--show-current").decode(errors="replace").rstrip("\n") or "HEAD"
    commit_log = _run_git(repo_path, "log", "-1", "--decorate", "--date=iso").decode(errors="replace").rstrip("\n")
    return f"branch: {branch}\n\n{commit_log}\n"


def export_git_state(
    log_dir: str | os.PathLike[str], repo_path: str | os.PathLike[str]
) -> None:
    """Export branch, commit hash, and diff for the training run."""
    output_dir = Path(log_dir) / "git"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / _GIT_STATE_TXT_FILENAME
    diff_path = output_dir / _GIT_DIFF_PATCH_FILENAME

    try:
        resolved_repo_path = _repo_root(repo_path)
        state = _format_git_state(resolved_repo_path)
        diff = _run_git(resolved_repo_path, "diff", "--binary", "HEAD", "--submodule=diff")
    except (OSError, subprocess.CalledProcessError) as error:
        message = str(error)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            message = error.stderr.decode(errors="replace").strip()
        state_path.write_text(f"Git state export failed.\n\n{message}\n", encoding="utf-8")
        diff_path.write_bytes(b"")
        print(f"[WARNING]: Skipping Git state export: {message}")
        return

    state_path.write_text(state, encoding="utf-8")
    diff_path.write_bytes(diff)
    print(f"[INFO]: Exported Git state to {state_path} and {diff_path}")
