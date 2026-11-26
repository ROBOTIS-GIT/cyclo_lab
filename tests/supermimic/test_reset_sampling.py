from types import SimpleNamespace

import torch

from cyclo_lab.manager_based.mimic.blinkmimic.training.reset_sampling import BlinkResetSampler
from cyclo_lab.manager_based.mimic._shared.motion_library_command import MotionLibraryCommand


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


def test_removed_clip_measurements_survive_readdition_but_not_replacement():
    original = command([("a", 51), ("b", 101)])
    original.sampler.returns[:] = torch.tensor([1, 2, 3])
    original.sampler.effort.fill_(0.9)
    shortened = command([("a", 51)])
    MotionLibraryCommand.restore_sampling(shortened, MotionLibraryCommand.sampling_state(original))
    assert "b" in shortened.inactive_sampling
    assert "effort" not in shortened.inactive_sampling["b"]["stats"]
    saved = MotionLibraryCommand.sampling_state(shortened)
    returned = command([("a", 51), ("b", 101)])
    MotionLibraryCommand.restore_sampling(returned, saved)
    assert returned.sampler.returns.tolist() == [1, 2, 3]
    assert not returned.inactive_sampling
    replaced = command([("a", 51), ("new-content", 101)])
    MotionLibraryCommand.restore_sampling(replaced, saved)
    assert replaced.sampler.returns.tolist() == [1, 0, 0]


def test_sampler_mode_change_resets_statistics():
    import pytest

    original = command([("a", 51)])
    original.sampler.returns.fill_(5)
    saved = MotionLibraryCommand.sampling_state(original)
    saved["sampler"]["mode"] = "ReturnResetSampler"
    target = command([("a", 51)])
    with pytest.warns(UserWarning, match="Sampler mode"):
        MotionLibraryCommand.restore_sampling(target, saved)
    assert not target.sampler.returns.any()
