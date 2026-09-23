"""Reward changes validate completely before mutating the training configuration."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location('service_rewards', Path(__file__).resolve().parents[1] / 'scripts/codex-service-rewards.py')
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)
TASK = 'Cyclo-Velocity-Flat-K1-Rev1-v0'


def cfg():
    return SimpleNamespace(rewards=SimpleNamespace(track_lin_vel_xy_exp=SimpleNamespace(weight=1.0), flat_orientation_l2=SimpleNamespace(weight=-5.0)))


def test_partial_override_preserves_other_weights_and_accepts_zero(tmp_path):
    config = cfg()
    path = tmp_path / 'rewards.json'
    path.write_text(json.dumps({'task_id': TASK, 'reward_terms': [{'term_id': 'track_lin_vel_xy_exp', 'weight': 0}]}))
    mod.apply_reward_file(config, TASK, path)
    assert config.rewards.track_lin_vel_xy_exp.weight == 0
    assert config.rewards.flat_orientation_l2.weight == -5


@pytest.mark.parametrize('term', [
    {'term_id': 'track_lin_vel_xy_exp', 'weight': 10**1000},
    {'term_id': '__dict__', 'weight': 1}, {'term_id': 'reference_anchor_position', 'weight': .5},
    {'term_id': 'track_lin_vel_xy_exp', 'weight': True}, {'term_id': 'track_lin_vel_xy_exp', 'weight': '1'},
    {'term_id': 'track_lin_vel_xy_exp', 'weight': None}, {'term_id': 'track_lin_vel_xy_exp', 'weight': 2.1},
    {'term_id': 'track_lin_vel_xy_exp', 'weight': -1}, {'term_id': 'flat_orientation_l2', 'weight': float('nan')},
    {'term_id': 'flat_orientation_l2', 'weight': float('inf')}, {'term_id': 'flat_orientation_l2', 'weight': -2, 'func': 'exec'},
])
def test_bad_payload_leaves_every_weight_unchanged(tmp_path, term):
    config = cfg()
    path = tmp_path / 'rewards.json'
    path.write_text(json.dumps({'task_id': TASK, 'reward_terms': [{'term_id': 'track_lin_vel_xy_exp', 'weight': 1.5}, term]}))
    with pytest.raises(ValueError): mod.apply_reward_file(config, TASK, path)
    assert config.rewards.track_lin_vel_xy_exp.weight == 1
    assert config.rewards.flat_orientation_l2.weight == -5


@pytest.mark.parametrize('body', [
    {'task_id': 'Cyclo-Mimic-K1-Rev1-Dance1', 'reward_terms': []},
    {'task_id': TASK, 'reward_terms': [], 'hydra': 'foo'},
    {'task_id': TASK, 'reward_terms': [{'term_id': 'track_lin_vel_xy_exp', 'weight': 1.5}]*2},
])
def test_task_extra_fields_and_duplicates_rejected(tmp_path, body):
    path = tmp_path / 'rewards.json'; path.write_text(json.dumps(body))
    with pytest.raises(ValueError): mod.apply_reward_file(cfg(), TASK, path)


def test_duplicate_json_keys_disabled_terms_and_large_files_rejected(tmp_path):
    path = tmp_path / 'rewards.json'
    path.write_text('{"task_id":"x","task_id":"'+TASK+'","reward_terms":[]}')
    with pytest.raises(ValueError): mod.apply_reward_file(cfg(), TASK, path)
    path.write_text(json.dumps({'task_id': TASK, 'reward_terms': [{'term_id': 'track_ang_vel_z_exp', 'weight': .5}]}))
    with pytest.raises(ValueError): mod.apply_reward_file(cfg(), TASK, path)
    path.write_text(' ' * 32769)
    with pytest.raises(ValueError): mod.apply_reward_file(cfg(), TASK, path)
