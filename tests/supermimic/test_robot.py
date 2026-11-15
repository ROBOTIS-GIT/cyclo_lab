import numpy as np
import mujoco
from k1 import get_spec, joint_names, BODIES


def test_real_k1_asset_cpu_simulation():
    model = get_spec().compile()
    assert (model.nq, model.nv, model.nu) == (30, 29, 0)
    assert [model.joint(i).name for i in range(1, model.njnt)] == joint_names()
    for name in BODIES:
        assert model.body(name).id > 0
    assert model.sensor('imu_ang_vel').dim == 3
    data = mujoco.MjData(model)
    for _ in range(10):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert data.time > 0


def test_k1_runtime_config_has_no_g1_robot_contracts():
    from k1 import env_cfg
    cfg = env_cfg()
    play = env_cfg(play=True)
    assert cfg.commands['motion'].body_names == BODIES
    assert cfg.commands['motion'].anchor_body_name == 'pelvis'
    assert len(cfg.scene.entities['robot'].articulation.actuators) == 23
    assert len(cfg.actions['joint_pos'].scale) == 23
    assert cfg.scene.sensors == ()
    assert play.commands['motion'].sampling_mode == 'start'
    cfg.actions['joint_pos'].scale.clear()
    assert len(play.actions['joint_pos'].scale) == 23
