# Changelog for package robotis_lab

1.2.1 (2026-06-02)
------------------
* Added swerve odometry calculation. 
* Updated `/odom` publishing in SH5 DDS bringup.
* Contributors: Howon Kim

1.2.0 (2026-05-28)
------------------
* Updated FFW_SH5.usd for visualize simulation
* Added SH5 robot configuration for swerve base, lift, arms, hands, and head actuators.
* Added environment helpers for Simple Warehouse spawning and graspable card boxes.
* Added swerve drive inverse kinematics helper for converting `cmd_vel` into wheel commands.
* Added SH5 DDS bringup using the DDS SDK for robot command input and state publishing.
* Contributors: Howon Kim, Bosung Kwon

1.1.0 (2025-12-15)
------------------
Added
^^^^^
* FFW_SG2 Sim2Real Imitation Learning Pipeline:
    * Implemented full sim2real pipeline for FFW_SG2 dual-arm humanoid robot
    * Added ``ffw_sg2_sdk.py`` DDS SDK for real robot teleoperation and state publishing
    * Created pick-and-place task environment with dual-arm support (``FFW_SG2/pick_place/``)
    * Implemented MDP components: observations, terminations, and event handlers for pick-and-place task
    * Added support for data recording, sub-task annotation, action conversion, and LeRobot dataset export
* New 3D Object Assets:
    * brush_ring, pliers_ring, scissors_ring, screw_driver_ring, silicone_tube_ring, tooth_brush
    * plastic_basket2, robotis_net_table
    * background_cube for visual domain randomization
    * Added corresponding texture files (AO, normal, diffuse maps) for all new objects
* Documentation:
    * Updated README with inference command for simulation
    * Updated keyboard control instructions for FFW_SG2 robot including joystick button mappings

Fixed
^^^^^
* Actuator Configuration Improvements:
    * Tuned actuator parameters (stiffness, damping, effort limits) for FFW_SG2, FFW_BG2, and OMY robots for improved simulation-to-real transfer
    * Split arm actuators into motor-specific groups (DY_80, DY_70, DP-42) matching real hardware specifications
    * Adjusted solver iteration counts for better physics stability
* Updated FFW_BG2.usd, FFW_SG2.usd, and OMY.usd for enhanced simulation fidelity
* Renamed camera references from ``cam_head_left`` to ``cam_head`` for consistency
* Updated ``robotis_dds_python`` submodule to latest version

* Contributors: Taehyeong Kim

1.0.0 (2025-11-17)
------------------
### Added
* Added comprehensive Docker containerization for consistent development environment
* Docker Infrastructure:
    * Created `Dockerfile.base` based on NVIDIA Isaac Sim container image
    * Implemented multi-stage build with Isaac Lab and Robotis Lab integration
    * Added `docker-compose.yaml` with volume management for caches, logs, and datasets
    * Created `container.sh` management script with build, start, enter, stop, clean, and logs commands
    * Implemented X11 forwarding support for GUI applications through `x11.yaml`
    * Added `entrypoint.sh` for runtime symbolic link setup
    * Configured `.dockerignore` files to optimize build context
* Dependencies Installation:
    * Automated CycloneDDS build and installation from third_party submodule
    * Integrated robotis_dds_python installation from third_party submodule
    * Created separate Python virtual environment for LeRobot with version 0.3.3
    * Installed Isaac Lab and all required dependencies
* Container Features:
    * Volume persistence for Isaac Sim caches, logs, and data
    * Bind mounts for source code, scripts, datasets, and third_party submodules
    * GPU acceleration support via NVIDIA Container Toolkit
    * Bash history preservation across container sessions
    * Pre-configured environment variables and command aliases
    * Network host mode for seamless communication
* Configuration:
    * Added `.env.base` for centralized environment configuration
    * Support for Isaac Sim 5.1.0 as default version
    * Customizable paths and container naming

* Contributors: Taehyeong Kim

0.2.2 (2025-11-14)
------------------
### Fixed
* Resolved a problem where the initial pose for the pick-and-place task using the FFW_BG2 model was not being applied correctly.
* Addressed an issue where the gripper could not grasp objects due to the mimic joint configuration, and updated the gripper setup accordingly.
* Contributors: Taehyeong Kim

0.2.1 (2025-11-13)
------------------
### OMY Reach Sim2Real Reinforcement Learning Pipeline
* Refactored OMY reach policy inference code to use DDS for joint state handling and trajectory publishing
* Removed ROS 2 dependency by integrating robotis_dds_python library for direct communication

### Documentation Update
* Renamed ReleaseNote.md to CHANGELOG.rst for better standardization and readability.

0.2.0 (2025-10-28)
------------------
### OMY Sim2Real Imitation Learning Pipeline
* Folder Structure Refactor:
    * Tasks are now separated and organized into two categories:
        * real_world_tasks – for real robot execution
        * simulator_tasks – for simulation environments
* Sim2Real Pipeline Implementation:
    * Task Recording: Added functionality to record demonstrations for the OMY plastic bottle pick-and-place task in simulation.
    * Sub-task Annotation: Introduced annotation tools for splitting demonstrations into meaningful sub-tasks, improving policy learning efficiency.
    * Action Representation Conversion: Converted control commands from joint-space to IK-based end-effector pose commands for better real-world transfer.
    * Data Augmentation: Added augmentation techniques to increase dataset diversity and enhance policy generalization.
    * Dataset Conversion: Integrated data conversion to the LeRobot dataset format, enabling compatibility with LeRobot’s training framework.
* ROS 2 Integration:
    * Modified to receive the leader’s /joint_trajectory values using the robotis_dds_python library without any ROS 2 dependency.

0.1.2 (2025-07-29)
------------------
### FFW BG2 Pick-and-Place Imitation Learning Environment
Built an imitation learning environment for cylindrical rod pick-and-place using the FFW BG2 robot.

* Implemented the full pipeline:
    * Data recording
    * Sub-task annotation
    * Data augmentation
    * Training

Enabled observation input support for right_wrist_cam and head_cam.
Fixed the issue with OMY STACK task not functioning correctly.
Performed parameter tuning and code cleanup for OMY Reach task's Sim2Real code (no functional issues, just improvements).

0.1.1 (2025-07-16)
------------------
### Sim2Real Deployment Support Added
* Developed Sim2Real deployment pipeline for the OMY Reach task.
* Enabled running policies trained in Isaac Sim on real-world OMY robots.
* Provided detailed usage instructions and demonstration videos in the README.
* Refactored folder structure for source and scripts to improve maintainability.

0.1.0 (2025-07-01)
------------------
### Initial Release
* Developed as an external package for Isaac Lab
* Verified compatibility with the following environments:
    * Isaac Sim 4.5.0 and 5.0.0
    * Isaac Lab 2.1.0 and feature/isaacsim_5_0 branch
* Introduced simulation environments for reinforcement learning and imitation learning, featuring two Robotis robots:
    * OMY
    * FFW
* Enables users to conduct training and research using Robotis robots with Isaac Lab, including full support for custom tutorials in reinforcement and imitation learning scenarios.
