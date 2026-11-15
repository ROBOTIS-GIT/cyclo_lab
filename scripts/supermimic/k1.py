# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""K1 adapter for the unmodified upstream differentiable world-model runner."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
XML = ROOT / 'third_party/ai_sapiens/ai_sapiens_description/mujoco/k1/k1.xml'
TASK = 'SuperMimic-GenCoord-K1'
BODIES = ('pelvis', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link',
          'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link', 'torso_link',
          'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_roll_rubber_hand',
          'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_roll_rubber_hand')


def joint_names(xml=XML):
    """MuJoCo depth-first articulated joint order, independent of Isaac ordering."""
    tree = ET.parse(xml)
    return [j.attrib['name'] for j in tree.findall('.//worldbody//joint')]


def get_spec():
    import mujoco
    tree = ET.parse(XML)
    root = tree.getroot()
    compiler = root.find('compiler')
    compiler.set('meshdir', str((XML.parent / compiler.get('meshdir')).resolve()))
    # XML torque motors must be removed before mjlab creates position actuators.
    for tag in ('actuator', 'keyframe'):
        element = root.find(tag)
        if element is not None:
            root.remove(element)
    pelvis = root.find(".//body[@name='pelvis']")
    ET.SubElement(pelvis, 'site', name='supermimic_imu', size='0.01')
    sensor = root.find('sensor')
    if sensor is None:
        sensor = ET.SubElement(root, 'sensor')
    ET.SubElement(sensor, 'gyro', name='imu_ang_vel', site='supermimic_imu')
    ET.SubElement(sensor, 'velocimeter', name='imu_lin_vel', site='supermimic_imu')
    return mujoco.MjSpec.from_string(ET.tostring(root, encoding='unicode'))


def robot_cfg():
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
    actuators = []
    scales = {}
    for name in joint_names():
        small = any(s in name for s in ('shoulder', 'elbow', 'wrist', 'ankle_roll'))
        inertia, effort = (0.00564892, 47.277) if small else (0.01936542, 96.864)
        frequency = 10 * 2 * math.pi
        stiffness, damping = inertia * frequency**2, 4 * inertia * frequency
        actuators.append(BuiltinPositionActuatorCfg(
            target_names_expr=(name,), stiffness=stiffness, damping=damping,
            effort_limit=effort, armature=inertia))
        scales[name] = 0.25 * effort / stiffness
    cfg = EntityCfg(spec_fn=get_spec, articulation=EntityArticulationInfoCfg(
        actuators=tuple(actuators), soft_joint_pos_limit_factor=0.9),
        init_state=EntityCfg.InitialStateCfg(pos=(0, 0, 0.764), joint_pos={
            '.*': 0.0, '.*_hip_pitch_joint': -0.3, '.*_knee_joint': 0.63,
            '.*_ankle_pitch_joint': -0.33, '.*_elbow_joint': 0.6,
            '.*_shoulder_pitch_joint': 0.2, 'left_shoulder_roll_joint': 0.2,
            'right_shoulder_roll_joint': -0.2}))
    return cfg, scales


def env_cfg(play=False):
    from supermimic.tasks.env_cfgs import gencoord_env_cfg
    cfg = gencoord_env_cfg(play=play)
    robot, scales = robot_cfg()
    cfg.scene.entities = {'robot': robot}
    cfg.scene.sensors = ()
    cfg.actions['joint_pos'].scale = scales
    cfg.commands['motion'].anchor_body_name = 'pelvis'
    cfg.commands['motion'].body_names = BODIES
    cfg.events['foot_friction'].params['asset_cfg'].geom_names = '.*_ankle_roll_link_collision_.*'
    cfg.viewer.body_name = 'pelvis'
    return cfg


def register():
    import mjlab.tasks  # noqa: F401
    import supermimic.tasks  # noqa: F401
    from mjlab.tasks.registry import register_mjlab_task, load_rl_cfg
    register_mjlab_task(task_id=TASK, env_cfg=env_cfg(), play_env_cfg=env_cfg(True),
                       rl_cfg=load_rl_cfg('Mjlab-Tracking-Flat-Unitree-G1'))
