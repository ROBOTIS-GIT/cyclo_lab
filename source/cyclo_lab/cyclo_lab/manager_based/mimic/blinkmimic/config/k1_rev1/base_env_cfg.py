# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass

from cyclo_lab.manager_based.mimic.motion_library_env_cfg import K1MotionLibraryEnvCfg, MotionLibraryObservationsCfg
from cyclo_lab.manager_based.mimic.motion_bridge import generalized_state


@configclass
class TrackingStateCfg(ObservationGroupCfg):
    state = ObservationTermCfg(func=generalized_state)


@configclass
class BlinkObservationsCfg(MotionLibraryObservationsCfg):
    tracking_state = TrackingStateCfg()


@configclass
class BlinkMimicLibraryEnvCfg(K1MotionLibraryEnvCfg):
    compute_final_obs: bool = True
    observations: BlinkObservationsCfg = BlinkObservationsCfg()
