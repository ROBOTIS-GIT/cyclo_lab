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

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass

from cyclo_lab.manager_based.mimic._shared.motion_library_env_cfg import K1MotionLibraryEnvCfg, MotionLibraryObservationsCfg
from cyclo_lab.manager_based.mimic._shared.motion_bridge import generalized_state


@configclass
class TrackingStateCfg(ObservationGroupCfg):
    state = ObservationTermCfg(func=generalized_state)


@configclass
class BlinkObservationsCfg(MotionLibraryObservationsCfg):
    tracking_state = TrackingStateCfg()


@configclass
class BlinkMimicLibraryEnvCfg(K1MotionLibraryEnvCfg):
    foot_contact_z: float = 0.074
    compute_final_obs: bool = True
    observations: BlinkObservationsCfg = BlinkObservationsCfg()
