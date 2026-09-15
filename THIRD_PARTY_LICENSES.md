# Third-Party Licenses

This project includes code adapted from third-party open-source projects.

## RealSense/realsense-ros

- Source: https://github.com/realsenseai/realsense-ros/tree/9a11121700cb4780e273e34141f6402fe184321d/realsense2_description
- Source asset: `realsense2_description/meshes/d405.stl`
- License: Apache-2.0; see [LICENSE-RealSense](source/cyclo_lab/data/robots/OMY/licenses/LICENSE-RealSense).
- Copyright notice: [COPYRIGHT-RealSense](source/cyclo_lab/data/robots/OMY/licenses/COPYRIGHT-RealSense).
- Used in: `source/cyclo_lab/data/robots/OMY/OMY_HX5_LEFT.usd` and `OMY_HX5_RIGHT.usd`.

The D405 housing mesh is embedded in these USD assets after STL-to-USD conversion,
unit scaling, placement, and application of the robot's black visual material.
The RealSense ROS driver and its software dependencies are not bundled.

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
