"""Per-env sim-context conditioning for the world model."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEGENERATE_HALF_WIDTH = 1e-3
"""Half-widths at or below this are treated as constant. Float32 ULP for
O(1) values is ~1e-7; dividing ``(raw - mean)`` by ``half_width`` amplifies
that noise by ``1/half_width``. At 1e-3 the worst-case amplified noise is
~1e-4 — well below the [-1, 1] signal range. Below ~1e-6 the noise
overwhelms the signal entirely. All default DR ranges
(foot_friction ±0.45, base_com ±0.025–0.05, encoder_bias ±0.01) clear
this threshold."""


def _mean_scale(lo: float, hi: float) -> tuple[float, float]:
  """Range → (mean, scale). Half-widths below the precision floor get
  scale=1.0 so ``(raw - mean) ≈ 0`` instead of amplifying float-ULP noise."""
  assert hi >= lo, f"empty range: lo={lo}, hi={hi}"
  half = (hi - lo) / 2.0
  return (lo + hi) / 2.0, half if half > _DEGENERATE_HALF_WIDTH else 1.0


_ENTITY_IDS_ATTR = {"body": "body_ids", "site": "site_ids", "geom": "geom_ids"}


class SimContextSpec(ABC):
  name: str
  dim: int

  @abstractmethod
  def bind(self, env: ManagerBasedRlEnv) -> list[tuple[float, float]]:
    """Resolve indices and ``dim``; return each column's DR range ``(lo, hi)``."""
    ...

  @abstractmethod
  def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return raw per-env values of shape ``(num_envs, dim)``."""
    ...

  @staticmethod
  def _global_entity_id(asset, entity_type: str, local_idx: int) -> int:
    """Asset-local entity index → mjModel global index. DR writes at global
    ids; reading at asset-local silently returns an unrandomized neighbor."""
    return int(getattr(asset.indexing, _ENTITY_IDS_ATTR[entity_type])[local_idx].item())


class FootFrictionSpec(SimContextSpec):
  """Per-env tangential foot friction read from one matching geom.

  mjlab's ``foot_friction`` event uses ``shared_random=True``, so all matching
  geoms in one env carry the same value; we read from the first match.
  """

  name = "foot_friction"
  dim = 1

  def __init__(self) -> None:
    self._geom_idx: int | None = None

  def bind(self, env: ManagerBasedRlEnv) -> list[tuple[float, float]]:
    event_cfg = env.cfg.events.get(self.name)
    if event_cfg is None:
      raise RuntimeError(f"FootFrictionSpec: no event '{self.name}' on env.cfg.events")
    params = event_cfg.params
    ranges = params["ranges"]

    asset_cfg = params["asset_cfg"]
    asset = env.scene[asset_cfg.name]
    pattern = asset_cfg.geom_names
    if isinstance(pattern, str):
      pat = re.compile(pattern)
      matches = [i for i, name in enumerate(asset.geom_names) if pat.match(name)]
    elif isinstance(pattern, (list, tuple)) and pattern:
      name_set = set(pattern)
      matches = [i for i, name in enumerate(asset.geom_names) if name in name_set]
    else:
      raise ValueError(f"FootFrictionSpec: '{self.name}' has empty geom_names")
    if not matches:
      raise ValueError(
        f"FootFrictionSpec: no geoms in '{asset_cfg.name}' match {pattern!r}"
      )
    self._geom_idx = self._global_entity_id(asset, "geom", matches[0])
    return [ranges[0] if isinstance(ranges, dict) else ranges]

  def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    if self._geom_idx is None:
      raise RuntimeError("FootFrictionSpec.read() called before bind()")
    return env.sim.model.geom_friction[:, self._geom_idx : self._geom_idx + 1, 0]


class CoMOffsetSpec(SimContextSpec):
  name = "base_com"
  dim = 3

  def __init__(self) -> None:
    self._body_id: int | None = None

  def bind(self, env: ManagerBasedRlEnv) -> list[tuple[float, float]]:
    event_cfg = env.cfg.events.get(self.name)
    if event_cfg is None:
      raise RuntimeError(f"CoMOffsetSpec: no event '{self.name}' on env.cfg.events")
    params = event_cfg.params
    ranges = params["ranges"]
    asset_cfg = params["asset_cfg"]
    if len(asset_cfg.body_names) != 1:
      raise ValueError(
        f"CoMOffsetSpec: expected exactly 1 body_name, got {asset_cfg.body_names!r}"
      )
    body_name = asset_cfg.body_names[0]
    asset = env.scene[asset_cfg.name]
    self._body_id = self._global_entity_id(
      asset, "body", asset.body_names.index(body_name)
    )
    return [ranges[i] for i in range(self.dim)]

  def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    if self._body_id is None:
      raise RuntimeError("CoMOffsetSpec.read() called before bind()")
    return env.sim.model.body_ipos[:, self._body_id, :]


class EncoderBiasSpec(SimContextSpec):
  name = "encoder_bias"
  dim = 0  # set in bind() from asset.num_joints

  def __init__(self) -> None:
    self._asset_name: str | None = None

  def bind(self, env: ManagerBasedRlEnv) -> list[tuple[float, float]]:
    event_cfg = env.cfg.events.get(self.name)
    if event_cfg is None:
      raise RuntimeError(f"EncoderBiasSpec: no event '{self.name}' on env.cfg.events")
    params = event_cfg.params
    asset_cfg = params["asset_cfg"]
    self._asset_name = asset_cfg.name
    self.dim = env.scene[self._asset_name].num_joints
    return [params["bias_range"]] * self.dim

  def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    if self._asset_name is None:
      raise RuntimeError("EncoderBiasSpec.read() called before bind()")
    return env.scene[self._asset_name].data.encoder_bias


class SimContext(nn.Module):
  def __init__(self, specs: list[SimContextSpec], ranges: list[tuple[float, float]]):
    super().__init__()
    self.specs = list(specs)
    stats = [_mean_scale(lo, hi) for lo, hi in ranges]
    self.register_buffer("mean", torch.tensor([m for m, _ in stats]))
    self.register_buffer("scale", torch.tensor([s for _, s in stats]))

  @classmethod
  def from_env(cls, env: ManagerBasedRlEnv) -> SimContext:
    """Discover, build, bind, and validate from the env's active DR events.

    Spec order here is the column layout of the normalized context: the replay
    buffer's ``sim_ctx`` field and the Dynamics input.
    """
    candidates: tuple[type[SimContextSpec], ...] = (
      FootFrictionSpec,
      CoMOffsetSpec,
      EncoderBiasSpec,
    )
    specs = [c() for c in candidates if c.name in env.cfg.events]
    if not specs:
      raise RuntimeError(
        "No sim_ctx specs derived from env.cfg.events. "
        "Enable at least one DR event (foot_friction, base_com, encoder_bias) "
        "in the task cfg."
      )
    return cls(specs, [r for s in specs for r in s.bind(env)]).to(env.device)

  @property
  def output_dim(self) -> int:
    return sum(spec.dim for spec in self.specs)

  def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Normalized per-env context ``(num_envs, output_dim)``, in spec order."""
    return (torch.cat([s.read(env) for s in self.specs], -1) - self.mean) / self.scale

  def find(self, name: str) -> slice | None:
    """The columns spec ``name`` occupies in a normalized context."""
    offset = 0
    for spec in self.specs:
      if spec.name == name:
        return slice(offset, offset + spec.dim)
      offset += spec.dim
    return None

  def denormalize(self, sim_ctx: torch.Tensor, columns: slice) -> torch.Tensor:
    """Raw values of ``columns`` in the normalized context ``sim_ctx``."""
    return sim_ctx[..., columns] * self.scale[columns] + self.mean[columns]
