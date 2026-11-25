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
# Authors: Woojin Wie, Kiwoong Park

"""K1 Rev.2 uses the K1 control configuration with its own robot description."""

from .K1_rev1 import AI_SAPIENS_DESCRIPTION_DIR, K1_REV1_INERTIA_TUNED_CFG

K1_REV2_URDF_PATH = f"{AI_SAPIENS_DESCRIPTION_DIR}/urdf/k1_rev2/k1.urdf"

K1_REV2_INERTIA_TUNED_CFG = K1_REV1_INERTIA_TUNED_CFG.replace(
    spawn=K1_REV1_INERTIA_TUNED_CFG.spawn.replace(asset_path=K1_REV2_URDF_PATH)
)
