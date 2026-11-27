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

"""Checkpoint selection using the same resume flags as the RSL runner."""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ResumeOptions:
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "last.pt"


def select_resume_checkpoint(log_root, options, *, skip_discovered_symlinks=False, run_sort_key=None):
    if not options.resume:
        if options.load_run != ".*" or options.load_checkpoint != "last.pt":
            raise ValueError("Checkpoint selection requires --agent.resume")
        return None
    checkpoint = Path(options.load_checkpoint).expanduser()
    if checkpoint.is_absolute():
        if not checkpoint.is_file():
            raise ValueError(f"Checkpoint does not exist: {checkpoint}")
        return checkpoint.resolve()

    root = Path(log_root)
    explicit = Path(options.load_run).expanduser()
    if not explicit.is_absolute():
        explicit = root / explicit
    explicit_run = explicit.is_dir()
    if explicit_run:
        runs = [explicit]
    else:
        pattern = re.compile(options.load_run)
        runs = sorted(
            (
                p
                for p in root.glob("*")
                if p.is_dir() and pattern.match(p.name)
            ),
            key=run_sort_key,
            reverse=True,
        )

    def eligible(path):
        return path.is_file() and (
            not skip_discovered_symlinks or explicit_run or not path.is_symlink()
        )

    pattern = re.compile(options.load_checkpoint)
    for run in runs:
        exact = run / checkpoint
        if eligible(exact):
            return exact.resolve()
        candidates = []
        for directory in (run, run / "checkpoints"):
            for path in directory.glob("*.pt"):
                number = re.fullmatch(r"(?:model|ckpt)_([0-9]+)\.pt", path.name)
                if path.name != "last.pt" and not number:
                    continue
                if pattern.fullmatch(path.name) and eligible(path):
                    # The latest full checkpoint includes the optimizer state.
                    candidates.append((path.name == "last.pt", int(number[1]) if number else 0, path))
        if candidates:
            return max(candidates)[2].resolve()
    raise ValueError(
        f"No resume checkpoint {options.load_checkpoint!r} in "
        f"{root} (run={options.load_run!r})"
    )
