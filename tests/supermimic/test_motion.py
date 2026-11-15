import numpy as np
import pytest
from k1 import joint_names
from convert_motion import convert


def source(tmp_path, **overrides):
    names = joint_names()[::-1]
    values = np.tile(np.arange(23), (4, 1)).astype(float)
    fields = dict(fps=[50], joint_names=names, body_names=['pelvis'],
                  quaternion_order='xyzw', joint_pos=values, joint_vel=values + 1,
                  body_pos_w=np.zeros((4, 1, 3)),
                  body_quat_w=np.tile([0, 0, 0, 1.], (4, 1, 1)),
                  body_lin_vel_w=np.zeros((4, 1, 3)), body_ang_vel_w=np.ones((4, 1, 3)))
    fields.update(overrides)
    path = tmp_path / 'source.npz'
    np.savez(path, **fields)
    return path


def test_reorders_joints_and_root_quaternion(tmp_path):
    path = source(tmp_path)
    out = tmp_path / 'out.npz'
    convert(path, out)
    with np.load(out) as data:
        assert data['joint_names'].tolist() == joint_names()
        np.testing.assert_array_equal(data['joint_pos'][0], np.arange(23)[::-1])
        np.testing.assert_array_equal(data['root_quat_w'][0], [1, 0, 0, 0])
        np.testing.assert_array_equal(data['root_ang_vel_w'][0], [1, 1, 1])
    with pytest.raises(FileExistsError):
        convert(path, out)


@pytest.mark.parametrize('overrides', [dict(fps=[30]), dict(joint_names=['bad']*23),
    dict(quaternion_order='unknown'), dict(body_names=['torso']),
    dict(body_quat_w=np.zeros((4, 1, 4))), dict(joint_vel=np.full((4, 23), np.nan)),
    dict(joint_pos=np.zeros((4, 22)))])
def test_rejects_incompatible_motion(tmp_path, overrides):
    with pytest.raises((ValueError, IndexError)):
        convert(source(tmp_path, **overrides), tmp_path/'out.npz')
    assert not (tmp_path/'out.npz').exists()
