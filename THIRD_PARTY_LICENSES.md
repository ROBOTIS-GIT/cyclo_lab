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
The [ROBOTIS SuperMimic fork](https://github.com/ROBOTIS-GIT/supermimic) in
`third_party/supermimic` derives from [HybridRobotics/SuperMimic](https://github.com/HybridRobotics/supermimic).
See its [Apache license](third_party/supermimic/LICENSE),
[BSD-3-Clause math notice](third_party/supermimic/licenses/IsaacLab-BSD-3-Clause.txt)
and [patch record](third_party/supermimic/PATCHES.md).
Models, replay, WM/policy learners
and imagined environments remain official; native Isaac/K1 topology, collection,
motion validation and checkpoint adapters live in Cyclo Lab. No research MJLab
runtime is required by the Lab integration.
Reference motions are subject to the terms of their source datasets.

## Historical K1 motion inertial profile

`cyclo_lab/utils/k1_motion_profile.json` contains inertial masses/centers
extracted from ROBOTIS-GIT/ai_sapiens, Apache-2.0, revision
`b0ef567ca77db0f37089ed3b23e492baba60d6be`, file
`ai_sapiens_description/urdf/k1_rev1/k1.urdf`. Original source/blob hashes are
recorded in the profile. The [ai_sapiens license](third_party/ai_sapiens/LICENSE) is included in the submodule.
ROBOTIS added numerical reference verification and fixed-link COM aggregation;
the values were not fitted to motion data. This is asset provenance, not a source
of SuperMimic algorithm changes.

## BlinkMimic

The native Isaac adapter uses Fuyi Chen's BlinkMimic actor, critic, world model,
reference goals, replay, rewards and differentiable actor objective from the
pinned ROBOTIS fork in `third_party/blinkmimic`. See its [MIT license](third_party/blinkmimic/LICENSE) and
[third-party notices](third_party/blinkmimic/THIRD_PARTY_NOTICES.md).
