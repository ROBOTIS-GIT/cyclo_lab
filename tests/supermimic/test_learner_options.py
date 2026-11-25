"""Upstream learner settings remain independent through CLI and checkpoint loading."""

from dataclasses import asdict

import pytest

from cyclo_lab.manager_based.mimic._shared.motion_tracker import parse_cli, resolve_learner
from cyclo_lab.manager_based.mimic.blinkmimic import experiment as blinkmimic
from cyclo_lab.manager_based.mimic.supermimic import experiment as supermimic


def test_legacy_blinkmimic_config_preserves_model_and_learning_settings():
    legacy = dict(
        history_length=5, hidden=(16, 12), world_hidden=(24, 8),
        batch_size=7, horizon=3, policy_horizon=9, buffer_size=64,
        updates_per_step=0.2, reference_skip=True, structured_dynamics=False,
        ema=0.9, noise=0.12, probe_start=200, probe_interval=25,
    )
    cfg = blinkmimic.restore_config(legacy)
    assert cfg.hist == 5 and cfg.enc_hidden == cfg.dec_hidden == (16, 12)
    assert cfg.wm_hidden == (24, 8) and cfg.wm_batch == cfg.pi_batch == 7
    assert (cfg.wm_window, cfg.horizon, cfg.buffer_steps, cfg.utd) == (3, 9, 64, 0.2)
    assert cfg.pi_ref_skip and not cfg.wm_structured
    assert (cfg.pi_ema, cfg.action_noise, cfg.eprobe_start, cfg.eprobe_every) == (0.9, 0.12, 200, 25)
    assert asdict(blinkmimic.restore_config(asdict(cfg))) == asdict(cfg)
    assert "history_length" in legacy  # Loading never edits the source bundle.


def test_legacy_supermimic_config_restores_native_nested_learners():
    legacy = dict(
        representation="gencoord", hidden=(16, 12), world_hidden=(24, 8), critic_hidden=(),
        latent=8, batch_size=7, wm_batch_size=5, wm_batches=2, horizon=3,
        policy_horizon=9, buffer_size=64, collection_steps=16, velocity_fraction=0.25,
    )
    cfg = supermimic.restore_config(legacy)
    assert cfg.actor_hidden_dims == cfg.critic_hidden_dims == (16, 12)
    assert cfg.dynamics_hidden_dims == (24, 8) and cfg.latent_dim == 8
    assert (cfg.wm.batch_size, cfg.wm.num_batches, cfg.wm.horizon) == (5, 2, 3)
    assert (cfg.policy.batch_size, cfg.policy.horizon) == (7, 9)
    assert cfg.slot_mix == {"motion": 0.75, "velocity": 0.25}
    assert asdict(supermimic.restore_config(asdict(cfg))) == asdict(cfg)
    assert "hidden" in legacy


@pytest.mark.parametrize("family,task,obsolete", [
    (blinkmimic, "Cyclo-BlinkMimic-K1-Rev2-001", "history_length"),
    (blinkmimic, "Cyclo-BlinkMimic-K1-Rev2-001", "structured_dynamics"),
    (blinkmimic, "Cyclo-BlinkMimic-K1-Rev2-001", "batch_size"),
    (supermimic, "Cyclo-SuperMimic-GenCoord-K1-Rev2-001", "hidden"),
    (supermimic, "Cyclo-SuperMimic-GenCoord-K1-Rev2-001", "horizon"),
    (supermimic, "Cyclo-SuperMimic-GenCoord-K1-Rev2-001", "wm.batch"),
])
def test_obsolete_or_unknown_learner_options_are_not_silently_ignored(family, task, obsolete):
    from isaaclab_tasks.utils import resolve_task_config

    args, overrides = parse_cli("train", [f"agent.{obsolete}=4"])
    _, current = resolve_task_config(task, "learner_cfg_entry_point", overrides=overrides)
    with pytest.raises(ValueError, match="Unknown learner option"):
        resolve_learner(current, args, None, family, overrides=overrides)


@pytest.mark.parametrize("option", ["--hidden", "--batch-size", "--horizon", "--policy-horizon"])
def test_combined_learning_switches_are_removed(option):
    with pytest.raises(SystemExit):
        parse_cli("train", [option, "4"])


@pytest.mark.parametrize("family,task,overrides", [
    (blinkmimic, "Cyclo-BlinkMimic-K1-Rev2-001", [
        "agent.wm_batch=7", "agent.pi_batch=4", "agent.enc_hidden=[16,12]",
        "agent.dec_hidden=[24,8]", "agent.wm_hidden=[32,16]",
    ]),
    (supermimic, "Cyclo-SuperMimic-GenCoord-K1-Rev2-001", [
        "agent.wm.batch_size=7", "agent.policy.batch_size=4",
        "agent.actor_hidden_dims=[16,12]", "agent.critic_hidden_dims=[24,8]",
        "agent.dynamics_hidden_dims=[32,16]",
    ]),
])
def test_agent_yaml_records_independent_native_fields(tmp_path, family, task, overrides):
    import yaml
    from isaaclab_tasks.utils import resolve_task_config
    from isaaclab_rl.entrypoints.common import dump_train_configs

    args, overrides = parse_cli("train", overrides)
    env, cfg = resolve_task_config(task, "learner_cfg_entry_point", overrides=overrides)
    cfg = resolve_learner(cfg, args, None, family, overrides=overrides)
    dump_train_configs(str(tmp_path), env, cfg)
    saved = yaml.safe_load((tmp_path / "params/agent.yaml").read_text())
    assert (tmp_path / "params/env.yaml").is_file()
    if family is blinkmimic:
        assert (saved["wm_batch"], saved["pi_batch"]) == (7, 4)
        assert saved["enc_hidden"] == [16, 12] and saved["dec_hidden"] == [24, 8]
        assert saved["wm_hidden"] == [32, 16]
    else:
        assert (saved["wm"]["batch_size"], saved["policy"]["batch_size"]) == (7, 4)
        assert saved["actor_hidden_dims"] == [16, 12] and saved["critic_hidden_dims"] == [24, 8]
        assert saved["dynamics_hidden_dims"] == [32, 16]
