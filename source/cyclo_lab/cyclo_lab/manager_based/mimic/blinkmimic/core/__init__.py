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

"""Use BlinkMimic's torch modules with Isaac motion and kinematics adapters."""

from ... import REPO_ROOT

_core = REPO_ROOT / "third_party/blinkmimic/src/blinkmimic"
if not (_core / "networks.py").is_file():
    raise ImportError("Initialize the third_party/blinkmimic submodule")
__path__.append(str(_core))
