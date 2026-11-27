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

"""Resolve task motion paths without loading motion data."""

import glob
from pathlib import Path


def resolve_motion_files(motion_paths):
    files = []
    seen = set()
    for pattern in motion_paths:
        path = Path(pattern).expanduser()
        if path.is_file():
            matches = [path]
        elif path.is_dir():
            matches = sorted(path.rglob("*.npz"))
        else:
            matches = sorted(Path(p) for p in glob.iglob(str(path), recursive=True))
        if not matches:
            raise ValueError(f"No motion files matched task path: {pattern}")
        for match in matches:
            if not match.is_file() or match.suffix.lower() != ".npz":
                raise ValueError(f"Task motion must be an NPZ file: {match}")
            resolved = match.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(resolved)
    if not files:
        raise ValueError("Task requires at least one motion")
    return files
