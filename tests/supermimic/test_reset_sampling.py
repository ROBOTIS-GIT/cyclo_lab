from types import SimpleNamespace

import torch

from cyclo_lab.manager_based.mimic.blinkmimic.core.reset_sampling import BlinkResetSampler
from cyclo_lab.manager_based.mimic.motion_library_command import MotionLibraryCommand


def command(clips):
    data = SimpleNamespace(
        device="cpu",
        motion_contracts=[{"sha256": key} for key, size in clips],
        motion_ends=torch.tensor([size for key, size in clips]).cumsum(0),
        start_bin_count=2,
        failure_score_by_bin=torch.zeros(2),
    )
    data.reference = SimpleNamespace(num_frames=int(data.motion_ends[-1]))
    data.sampler = BlinkResetSampler(SimpleNamespace(device="cpu", num_envs=1), data, 0.99)
    data._sampler_snapshot = lambda: MotionLibraryCommand._sampler_snapshot(data)
    return data


def test_reset_bins_skip_final_only_bin_and_preserve_scores_across_new_clips():
    original = command([("a", 51), ("b", 101)])
    assert original.sampler.starts.tolist() == [0, 51, 101]
    original.sampler.returns[:] = torch.tensor([1, 2, 3])
    saved = MotionLibraryCommand.sampling_state(original)
    for clips, expected in [
        ([("a", 51), ("b", 101)], [1, 2, 3]),
        ([("b", 101), ("c", 2), ("a", 51)], [2, 3, 0, 1]),
    ]:
        target = command(clips)
        MotionLibraryCommand.restore_sampling(target, saved)
        assert target.sampler.returns.tolist() == expected
        frame = target.sampler.sample(torch.arange(1000))
        assert not torch.isin(frame, target.motion_ends - 1).any()
