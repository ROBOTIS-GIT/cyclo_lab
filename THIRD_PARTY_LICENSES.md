# Third-Party Licenses

This project includes code adapted from third-party open-source projects.

## HybridRobotics/whole_body_tracking

- Source: https://github.com/HybridRobotics/whole_body_tracking
- License: MIT
- Used in: `source/cyclo_lab/cyclo_lab/manager_based/mimic`

The AI Sapiens mimic task stack adapts the reference-motion tracking MDP structure from `whole_body_tracking`.

```text
Copyright (c) 2024, The Isaac Lab Project Developers.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## ROBOTIS-GIT/zenoh_ros2_sdk

- Source: https://github.com/ROBOTIS-GIT/zenoh_ros2_sdk
- License: Apache-2.0
- Used in: `source/cyclo_lab/cyclo_lab/runtime/transport`

The SDK provides ROS2-compatible message transport over Zenoh without a full
ROS installation in the Isaac Sim Python environment.

## NVIDIA IsaacLab-Arena

- Source: https://github.com/isaac-sim/IsaacLab-Arena
- License: Apache-2.0
- Used in: `source/cyclo_lab/cyclo_lab/manager_based/manipulation/pick_place/config/ffw_sg2/galileo`

Cyclo Lab reuses the upstream Galileo scene, task, and metrics while supplying
the FFW-SG2 embodiment and action configuration.

## SuperMimic

Copyright 2026 The SuperMimic Authors. Apache-2.0.
Source: https://github.com/HybridRobotics/supermimic, revision
`abc124c1bd5976341034668af3d12c087f2f1f55`.
Selected algorithm sources, complete LICENSE and original README acknowledgements
are retained under `third_party/supermimic`. ROBOTIS modified only simulator import
boundaries, the WM environment constructor contract and removed mjlab registration;
marked files and `UPSTREAM.md` describe the port. Dynamics/replay/WM and policy
learners remain verbatim; imagined-rollout math is unchanged. The new runner collects states in the existing Isaac Lab runtime.
K1 assets and input motion data retain their existing licensing and gitlinks.

## Historical K1 motion inertial profile

`learning/supermimic/k1_motion_profile.json` contains inertial masses/centers
extracted from ROBOTIS-GIT/ai_sapiens, Apache-2.0, revision
`b0ef567ca77db0f37089ed3b23e492baba60d6be`, file
`ai_sapiens_description/urdf/k1_rev1/k1.urdf`. Original source/blob hashes are
recorded in the profile. The existing ai_sapiens submodule retains its LICENSE.
ROBOTIS added numerical reference verification and fixed-link COM aggregation;
the values were not fitted to motion data. This is asset provenance, not a source
of SuperMimic algorithm changes.
