"""Upstream learner settings remain independent through CLI and checkpoint loading."""

from dataclasses import asdict

import pytest

from cyclo_lab.manager_based.mimic._shared.motion_tracker import parse_cli, resolve_learner
from cyclo_lab.manager_based.mimic.blinkmimic import experiment as blinkmimic
from cyclo_lab.manager_based.mimic.supermimic import experiment as supermimic


@pytest.mark.parametrize("world_only", [False, True])
def test_blinkmimic_resume_restores_optimizer_choices(world_only):
    options = ["--world-model-checkpoint", "last.pt"] if world_only else []
    args, _ = parse_cli("train", options)
    current = blinkmimic.BlinkCfg()
    saved = asdict(blinkmimic.BlinkCfg(pi_opt="adam", wm_opt="adam"))
    resolved = resolve_learner(current, args, {"cfg": saved}, blinkmimic)
    expected = "muon" if world_only else "adam"
    assert resolved.pi_opt == resolved.wm_opt == expected


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
        "agent.pi_opt=adam", "agent.wm_opt=adam",
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
        assert saved["pi_opt"] == saved["wm_opt"] == "adam"
    else:
        assert (saved["wm"]["batch_size"], saved["policy"]["batch_size"]) == (7, 4)
        assert saved["actor_hidden_dims"] == [16, 12] and saved["critic_hidden_dims"] == [24, 8]
        assert saved["dynamics_hidden_dims"] == [32, 16]
