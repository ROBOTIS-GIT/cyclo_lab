"""Tests for supermimic.sim_context — spec bind parsers and SimContext aggregation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from supermimic.sim_context import (
  CoMOffsetSpec,
  EncoderBiasSpec,
  FootFrictionSpec,
  SimContext,
)


def _env(
  *,
  events: dict | None = None,
  scene: dict | None = None,
  sim_model=None,
  num_envs: int = 4,
):
  return SimpleNamespace(
    cfg=SimpleNamespace(events=events or {}),
    scene=scene or {},
    sim=SimpleNamespace(model=sim_model) if sim_model is not None else None,
    num_envs=num_envs,
    device="cpu",
  )


def _identity_indexing(num: int) -> SimpleNamespace:
  return SimpleNamespace(
    body_ids=torch.arange(num),
    geom_ids=torch.arange(num),
    site_ids=torch.arange(num),
  )


def _asset(*, geom_names: list[str] | None = None, num_joints: int = 0, **data):
  geoms = geom_names or []
  asset = SimpleNamespace(
    geom_names=geoms,
    num_joints=num_joints,
    indexing=_identity_indexing(max(len(geoms), 1)),
  )
  asset.data = SimpleNamespace(**data)
  return asset


def _friction_env(
  *,
  geom_names,
  asset_geom_pat,
  ranges=(0.3, 1.2),
  geom_friction=None,
  geom_id_offset: int = 0,
):
  event = SimpleNamespace(
    params={
      "ranges": ranges,
      "asset_cfg": SimpleNamespace(name="robot", geom_names=asset_geom_pat),
    }
  )
  asset = _asset(geom_names=geom_names)
  asset.indexing.geom_ids = torch.arange(len(geom_names)) + geom_id_offset
  return _env(
    events={"foot_friction": event},
    scene={"robot": asset},
    sim_model=SimpleNamespace(geom_friction=geom_friction)
    if geom_friction is not None
    else None,
  )


def _com_env(*, body_names, body_name, ranges, body_ipos=None, body_id_offset: int = 0):
  event = SimpleNamespace(
    params={
      "ranges": ranges,
      "asset_cfg": SimpleNamespace(name="robot", body_names=(body_name,)),
    }
  )
  asset = SimpleNamespace(
    body_names=tuple(body_names),
    indexing=SimpleNamespace(body_ids=torch.arange(len(body_names)) + body_id_offset),
  )
  asset.data = SimpleNamespace()
  return _env(
    events={"base_com": event},
    scene={"robot": asset},
    sim_model=SimpleNamespace(body_ipos=body_ipos) if body_ipos is not None else None,
  )


def _encoder_env(*, num_joints, encoder_bias=None, bias_range=(-0.01, 0.01)):
  params = {"asset_cfg": SimpleNamespace(name="robot"), "bias_range": bias_range}
  return _env(
    events={"encoder_bias": SimpleNamespace(params=params)},
    scene={
      "robot": _asset(
        num_joints=num_joints,
        encoder_bias=encoder_bias
        if encoder_bias is not None
        else torch.zeros(4, num_joints),
      )
    },
  )


@pytest.mark.parametrize("half_width", [0.0, 1e-9, 1e-7, 1e-5, 1e-4])
def test_sub_precision_range_gets_unit_scale(half_width):
  assert SimContext([], [(0.5 - half_width, 0.5 + half_width)]).scale.item() == 1.0


class TestFootFrictionSpec:
  def test_bind_regex_geom_names(self):
    spec = FootFrictionSpec()
    env = _friction_env(
      geom_names=["pelvis", "left_ankle_roll", "right_ankle_roll", "head"],
      asset_geom_pat=".*ankle_roll",
      ranges=(0.4, 1.1),
    )
    assert spec.bind(env) == [(0.4, 1.1)]
    assert spec._geom_idx == 1

  def test_bind_list_geom_names(self):
    spec = FootFrictionSpec()
    spec.bind(
      _friction_env(
        geom_names=["pelvis", "foot_l", "foot_r"],
        asset_geom_pat=["foot_l", "foot_r"],
      )
    )
    assert spec._geom_idx == 1

  def test_bind_axis_keyed_dict_ranges_picks_axis_zero(self):
    # mjlab's Ranges supports dict[int, tuple] keyed by friction axis
    # (0=sliding, 1=torsional, 2=rolling). FootFrictionSpec reads axis 0.
    ranges = {0: (0.2, 1.0), 1: (0.01, 0.05)}
    env = _friction_env(geom_names=["foot"], asset_geom_pat=["foot"], ranges=ranges)
    assert FootFrictionSpec().bind(env) == [ranges[0]]

  def test_bind_missing_event_raises(self):
    with pytest.raises(RuntimeError, match="no event"):
      FootFrictionSpec().bind(_env())

  def test_bind_no_matching_geoms_raises(self):
    with pytest.raises(ValueError, match="no geoms"):
      FootFrictionSpec().bind(
        _friction_env(geom_names=["pelvis", "head"], asset_geom_pat=".*ankle_roll")
      )

  def test_bind_empty_geom_names_raises(self):
    with pytest.raises(ValueError, match="empty geom_names"):
      FootFrictionSpec().bind(_friction_env(geom_names=["foot"], asset_geom_pat=[]))

  def test_read_before_bind_raises(self):
    with pytest.raises(RuntimeError, match="before bind"):
      FootFrictionSpec().read(_env())

  def test_read_returns_target_geom_friction(self):
    # geom_friction shape: (num_envs, num_geoms, 3) — last dim is (slide, tors, roll).
    geom_friction = torch.tensor(
      [
        [[0.0, 0.0, 0.0], [0.5, 0.01, 0.001]],
        [[0.0, 0.0, 0.0], [0.8, 0.01, 0.001]],
      ]
    )
    spec = FootFrictionSpec()
    env = _friction_env(
      geom_names=["other", "foot"],
      asset_geom_pat=["foot"],
      geom_friction=geom_friction,
    )
    spec.bind(env)
    assert torch.equal(spec.read(env), torch.tensor([[0.5], [0.8]]))


class TestCoMOffsetSpec:
  def test_bind_returns_per_axis_ranges_and_resolves_body_id(self):
    ranges = {0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)}
    env = _com_env(body_names=["pelvis", "torso"], body_name="torso", ranges=ranges)
    spec = CoMOffsetSpec()
    assert spec.bind(env) == [ranges[0], ranges[1], ranges[2]]
    assert spec._body_id == 1

  def test_bind_missing_event_raises(self):
    with pytest.raises(RuntimeError, match="no event"):
      CoMOffsetSpec().bind(_env())

  def test_read_before_bind_raises(self):
    with pytest.raises(RuntimeError, match="before bind"):
      CoMOffsetSpec().read(_env())

  def test_read_returns_body_ipos_slice(self):
    body_ipos = torch.tensor(
      [
        [[0.0, 0.0, 0.0], [0.01, -0.02, 0.03]],
        [[0.0, 0.0, 0.0], [-0.01, 0.02, -0.03]],
      ]
    )
    spec = CoMOffsetSpec()
    env = _com_env(
      body_names=["pelvis", "torso"],
      body_name="torso",
      ranges={0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
      body_ipos=body_ipos,
    )
    spec.bind(env)
    assert torch.equal(spec.read(env), body_ipos[:, 1, :])


class TestEncoderBiasSpec:
  def test_bind_sets_dim_and_per_joint_range(self):
    spec = EncoderBiasSpec()
    env = _encoder_env(num_joints=29, bias_range=(-0.5, 0.5))
    assert spec.bind(env) == [(-0.5, 0.5)] * 29
    assert spec.dim == 29

  def test_bind_missing_event_raises(self):
    with pytest.raises(RuntimeError, match="no event"):
      EncoderBiasSpec().bind(_env())

  def test_read_before_bind_raises(self):
    with pytest.raises(RuntimeError, match="before bind"):
      EncoderBiasSpec().read(_env())

  def test_read_returns_encoder_bias(self):
    spec = EncoderBiasSpec()
    bias = torch.randn(4, 29)
    env = _encoder_env(num_joints=29, encoder_bias=bias)
    spec.bind(env)
    assert torch.equal(spec.read(env), bias)


class _ConstSpec:
  """Test-only spec: returns a fixed tensor, no env coupling."""

  def __init__(self, name, dim, value):
    self.name = name
    self.dim = dim
    self._value = value

  def read(self, env) -> torch.Tensor:
    return torch.full((env.num_envs, self.dim), self._value)


class TestSimContext:
  def test_output_dim_sums_spec_dims(self):
    specs = [_ConstSpec("a", 3, 0.5), _ConstSpec("b", 5, 0.5)]
    assert SimContext(specs, [(0.0, 1.0)] * 8).output_dim == 8

  def test_read_concatenates_in_spec_order_and_normalizes(self):
    specs = [_ConstSpec("a", 2, 0.5), _ConstSpec("b", 3, 1.0)]
    out = SimContext(specs, [(0.0, 1.0)] * 2 + [(-1.0, 1.0)] * 3).read(_env(num_envs=4))
    assert out.shape == (4, 5)
    assert torch.allclose(out[:, :2], torch.zeros(4, 2), atol=1e-7)
    assert torch.allclose(out[:, 2:], torch.ones(4, 3), atol=1e-7)

  def test_find_and_denormalize(self):
    specs = [_ConstSpec("a", 2, 0.25), _ConstSpec("b", 3, 0.007)]
    ctx = SimContext(specs, [(0.0, 1.0)] * 2 + [(-0.01, 0.01)] * 3)
    assert ctx.find("b") == slice(2, 5)
    assert ctx.find("nope") is None
    raw = ctx.denormalize(ctx.read(_env(num_envs=4)), slice(2, 5))
    assert torch.allclose(raw, torch.full((4, 3), 0.007))


def _full_env(*event_names: str, num_envs: int = 2):
  """Build a fake env that all three concrete specs can bind to.

  Restrict the events dict via ``event_names`` to control which specs
  ``from_env`` discovers.
  """
  asset = SimpleNamespace(
    geom_names=["foot"],
    body_names=("pelvis", "torso"),
    num_joints=3,
    indexing=_identity_indexing(2),
  )
  asset.data = SimpleNamespace(encoder_bias=torch.zeros(num_envs, 3))
  all_events = {
    "foot_friction": SimpleNamespace(
      params={
        "ranges": (0.3, 1.2),
        "asset_cfg": SimpleNamespace(name="robot", geom_names=["foot"]),
      }
    ),
    "base_com": SimpleNamespace(
      params={
        "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
        "asset_cfg": SimpleNamespace(name="robot", body_names=("torso",)),
      }
    ),
    "encoder_bias": SimpleNamespace(
      params={
        "asset_cfg": SimpleNamespace(name="robot"),
        "bias_range": (-0.01, 0.01),
      }
    ),
  }
  events = {name: all_events[name] for name in event_names}
  return SimpleNamespace(
    cfg=SimpleNamespace(events=events),
    scene={"robot": asset},
    sim=SimpleNamespace(
      model=SimpleNamespace(
        geom_friction=torch.zeros(num_envs, 1, 3),
        body_ipos=torch.zeros(num_envs, 2, 3),
      )
    ),
    num_envs=num_envs,
    device="cpu",
  )


class TestGlobalEntityIdResolution:
  """When the scene places other entities ahead of the asset in mjModel
  (e.g. terrain), specs must read at the GLOBAL id where DR wrote — not the
  asset-local id, which would silently return an unrandomized neighbor."""

  def test_foot_friction_indexes_global_geom(self):
    # Asset's "foot" is local 0, but global 2 (e.g. terrain occupies 0-1).
    geom_friction = torch.tensor(
      [[[0.7, 0.01, 0.001], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.01, 0.001]]]
    )  # values at global 0 and 3
    spec = FootFrictionSpec()
    env = _friction_env(
      geom_names=["foot"],
      asset_geom_pat=["foot"],
      geom_friction=geom_friction,
      geom_id_offset=3,
    )
    spec.bind(env)
    assert spec._geom_idx == 3
    assert torch.equal(spec.read(env), torch.tensor([[0.5]]))

  def test_com_offset_indexes_global_body(self):
    body_ipos = torch.tensor(
      [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.01, -0.02, 0.03]],
      ]
    )  # asset body 1 lives at global 3 (offset=2)
    spec = CoMOffsetSpec()
    env = _com_env(
      body_names=["pelvis", "torso"],
      body_name="torso",
      ranges={0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
      body_ipos=body_ipos,
      body_id_offset=2,
    )
    spec.bind(env)
    assert spec._body_id == 3
    assert torch.equal(spec.read(env), body_ipos[:, 3, :])


class TestFromEnv:
  def test_raises_when_no_events(self):
    with pytest.raises(RuntimeError, match="No sim_ctx specs"):
      SimContext.from_env(_full_env())

  def test_includes_only_present_events(self):
    ctx = SimContext.from_env(_full_env("foot_friction"))
    assert [s.name for s in ctx.specs] == ["foot_friction"]

  def test_canonical_order_independent_of_events_order(self):
    """Order matches the candidate tuple in ``from_env`` so buffer field
    layout stays stable regardless of how the cfg lists events."""
    ctx = SimContext.from_env(_full_env("encoder_bias", "foot_friction", "base_com"))
    assert [s.name for s in ctx.specs] == [
      "foot_friction",
      "base_com",
      "encoder_bias",
    ]
