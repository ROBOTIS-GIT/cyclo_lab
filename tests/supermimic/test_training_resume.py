# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pytest

from cyclo_lab.utils.training_resume import (
    ResumeOptions,
    select_resume_checkpoint,
)


def test_latest_saved_run_and_nested_numeric_checkpoints(tmp_path):
    for run in ("2026-01", "2026-02", "2026-03"):
        (tmp_path / run).mkdir()
    (tmp_path / "2026-01/last.pt").touch()
    (tmp_path / "2026-02/last.pt").touch()
    assert (
        select_resume_checkpoint(tmp_path, ResumeOptions(True))
        == tmp_path / "2026-02/last.pt"
    )
    models = tmp_path / "2026-02/checkpoints"
    models.mkdir()
    for name in ("model_9000.pt", "model_10000.pt", "model_2000.pt"):
        (models / name).touch()
    assert (
        select_resume_checkpoint(
            tmp_path, ResumeOptions(True, "2026-02", "model_.*.pt")
        )
        == models / "model_10000.pt"
    )
    assert (
        select_resume_checkpoint(
            tmp_path,
            ResumeOptions(True, "2026-02", "checkpoints/model_2000.pt"),
        )
        == models / "model_2000.pt"
    )
    assert (
        select_resume_checkpoint(
            tmp_path,
            ResumeOptions(True, load_checkpoint=str(models / "model_9000.pt")),
        )
        == models / "model_9000.pt"
    )


def test_missing_explicit_run_is_not_replaced(tmp_path):
    (tmp_path / "other").mkdir()
    (tmp_path / "other/last.pt").touch()
    with pytest.raises(ValueError, match="No resume checkpoint"):
        select_resume_checkpoint(tmp_path, ResumeOptions(True, "missing"))
    with pytest.raises(ValueError, match="requires --agent.resume"):
        select_resume_checkpoint(tmp_path, ResumeOptions(False, "other"))
    assert select_resume_checkpoint(tmp_path, ResumeOptions()) is None


def test_wide_pattern_only_selects_resumable_models(tmp_path):
    run = tmp_path / "run"
    models = run / "checkpoints"
    models.mkdir(parents=True)
    for path in (run / "world_model.pt", run / "actor.pt", models / "model_20.pt"):
        path.touch()
    options = ResumeOptions(True, load_checkpoint=r".*\.pt")
    assert select_resume_checkpoint(tmp_path, options) == models / "model_20.pt"
    (run / "last.pt").touch()
    assert select_resume_checkpoint(tmp_path, options) == run / "last.pt"


def test_blinkmimic_play_and_resume_choose_the_same_custom_run(tmp_path):
    import os
    from cyclo_lab.manager_based.mimic._shared.checkpoint import find_checkpoint, training_run_order

    for run, time in (("aaa_new", 20), ("zzz_old", 10)):
        checkpoint = tmp_path / "task" / run / "last.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        os.utime(checkpoint, (time, time))
    resume = select_resume_checkpoint(tmp_path / "task", ResumeOptions(True), run_sort_key=training_run_order)
    assert find_checkpoint(tmp_path, "task", None) == resume == tmp_path / "task/aaa_new/last.pt"

def test_blinkmimic_discovers_training_files_and_honors_explicit_links(tmp_path):
    old = tmp_path / "2026-01/last.pt"
    new = tmp_path / "2026-02/last.pt"
    link = tmp_path / "zz-eval/last.pt"
    for file in (old, new, link):
        file.parent.mkdir()
    old.touch()
    new.touch()
    link.symlink_to(old)
    for selector in ("last.pt", r"last.*"):
        automatic = ResumeOptions(True, load_checkpoint=selector)
        assert select_resume_checkpoint(tmp_path, automatic, skip_discovered_symlinks=True) == new
    for explicit in (ResumeOptions(True, "zz-eval"), ResumeOptions(True, load_checkpoint=str(link))):
        assert select_resume_checkpoint(tmp_path, explicit, skip_discovered_symlinks=True) == old
