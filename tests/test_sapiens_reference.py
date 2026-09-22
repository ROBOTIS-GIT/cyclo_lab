"""CPU tests for bundled legacy motion convention at the Isaac Lab 3 boundary."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reference', ROOT / 'source/cyclo_lab/cyclo_lab/manager_based/mimic/mdp/reference_trajectory.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReferenceConvention(unittest.TestCase):
    def test_legacy_identity_and_yaw_are_converted_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'motion.npz'
            quats = np.array([[[1.,0,0,0]], [[2**-.5,0,0,2**-.5]]])
            np.savez(path, fps=50, joint_pos=np.zeros((2,1)), joint_vel=np.zeros((2,1)),
                     body_pos_w=np.zeros((2,1,3)), body_quat_w=quats,
                     body_lin_vel_w=np.zeros((2,1,3)), body_ang_vel_w=np.zeros((2,1,3)))
            loaded = module.ReferenceTrajectory(str(path), [0], quaternion_order='wxyz')
            torch.testing.assert_close(loaded.body_orientation_w[0,0], torch.tensor([0.,0,0,1]))
            torch.testing.assert_close(loaded.body_orientation_w[1,0], torch.tensor([0.,0,2**-.5,2**-.5]))
            unchanged = module.ReferenceTrajectory(str(path), [0], quaternion_order='xyzw')
            torch.testing.assert_close(unchanged.body_orientation_w, torch.tensor(quats,dtype=torch.float32))
            with self.assertRaises(ValueError):
                module.ReferenceTrajectory(str(path), [0], quaternion_order='auto')

    def test_bundled_dances_have_upright_normalized_initial_root(self):
        for name in ['dance1', 'dance2']:
            path = ROOT / f'source/cyclo_lab/data/motions/K1_rev1/{name}/{name}.npz'
            reference = module.ReferenceTrajectory(str(path), [0], quaternion_order='wxyz')
            x,y,z,w = reference.body_orientation_w[0,0]
            self.assertGreater(float(1-2*(x*x+y*y)), .95)
            torch.testing.assert_close(torch.linalg.vector_norm(reference.body_orientation_w, dim=-1),
                                       torch.ones((reference.num_frames,1)), atol=1e-5, rtol=1e-5)
            self.assertEqual(reference.joint_position.shape[1],23)


if __name__ == '__main__':
    unittest.main()
