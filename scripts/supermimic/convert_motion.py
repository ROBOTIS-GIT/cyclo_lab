# SPDX-License-Identifier: Apache-2.0
"""Convert Cyclo Isaac Lab NPZ to SuperMimic's named-joint root-state format."""
import argparse
from pathlib import Path
import numpy as np
from k1 import joint_names


def convert(source, target, quaternion_order=None):
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve():
        raise ValueError('Input and output must differ')
    with np.load(source, allow_pickle=False) as data:
        names = data['joint_names'].tolist()
        expected = joint_names()
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError('Motion must name exactly the 23 K1 joints without duplicates')
        fps = float(np.asarray(data['fps']).reshape(-1)[0])
        if fps != 50:
            raise ValueError('SuperMimic task runs at 50 Hz; resample input to 50 Hz first')
        order = quaternion_order
        if order is None and 'quaternion_order' in data:
            order = str(data['quaternion_order'].item())
        if order not in ('xyzw', 'wxyz'):
            raise ValueError('Specify quaternion order explicitly for legacy NPZ')
        body_names = data['body_names'].tolist()
        if body_names.count('pelvis') != 1:
            raise ValueError('Motion must have one pelvis body')
        pelvis = body_names.index('pelvis')
        index = [names.index(n) for n in expected]
        result = {k: data[k][:, index].copy() for k in ('joint_pos', 'joint_vel')}
        for out, field in [('root_pos_w', 'body_pos_w'), ('root_quat_w', 'body_quat_w'),
                           ('root_lin_vel_w', 'body_lin_vel_w'), ('root_ang_vel_w', 'body_ang_vel_w')]:
            result[out] = data[field][:, pelvis].copy()
        if order == 'xyzw':
            result['root_quat_w'] = result['root_quat_w'][:, [3, 0, 1, 2]]
        frames = len(result['joint_pos'])
        for field, width in [('joint_pos', 23), ('joint_vel', 23), ('root_pos_w', 3),
                             ('root_quat_w', 4), ('root_lin_vel_w', 3), ('root_ang_vel_w', 3)]:
            value = result[field]
            if value.shape != (frames, width) or frames < 2 or not np.isfinite(value).all():
                raise ValueError(f'Invalid shape or nonfinite data in {field}')
        q = result['root_quat_w']
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        if (norm < 1e-8).any():
            raise ValueError('Zero quaternion')
        result['root_quat_w'] = q / norm
        result.update(joint_names=np.asarray(expected), fps=np.asarray([50]),
                      quaternion_order=np.asarray('wxyz'))
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create protects existing reference data against accidental overwrite.
    with target.open('xb') as output:
        np.savez_compressed(output, **result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    parser.add_argument('--quaternion-order', choices=('xyzw', 'wxyz'))
    args = parser.parse_args()
    convert(args.source, args.target, args.quaternion_order)
