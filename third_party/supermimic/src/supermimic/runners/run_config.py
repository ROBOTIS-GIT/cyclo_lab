"""Rebuild and validate a runner config from a wandb run's serialized config."""

from __future__ import annotations

import dataclasses
import types
from typing import Any, Union, get_args, get_origin, get_type_hints

from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg


def load_runner_cfg(run_config: dict[str, Any]) -> SuperMimicRunnerCfg:
  """Reconstruct ``SuperMimicRunnerCfg`` from ``run_config["runner_cfg"]``.

  JSON round-tripping turns tuples into lists and erases the ``policy`` union's
  variant; both are recovered structurally. Unknown keys are ignored and missing
  fields take today's default with a warning, so runs written by older configs
  still load; a missing or empty ``runner_cfg`` raises.
  """
  raw = run_config.get("runner_cfg")
  if not raw:
    raise ValueError("run has no serialized runner_cfg; cannot rebuild its config")
  defaulted: list[str] = []
  cfg = _rebuild(SuperMimicRunnerCfg, raw, defaulted, "")
  if defaulted:
    print(f"[WARN] run cfg missing fields, using current defaults: {defaulted}")
  return cfg


def check_representation_matches(task_id: str, representation_name: str) -> None:
  """The task fixes the env's obs layout; a run trained with the other
  representation would rebuild — and fail to load — at the wrong shapes."""
  expected = "maxcoord" if "MaxCoord" in task_id else "gencoord"
  if representation_name != expected:
    raise ValueError(
      f"{task_id} is a {expected} task but the run was trained with "
      f"{representation_name}; pass a matching task id"
    )


def _rebuild(cls: type, raw: dict[str, Any], defaulted: list[str], prefix: str) -> Any:
  hints = get_type_hints(cls)
  kwargs: dict[str, Any] = {}
  for field in dataclasses.fields(cls):
    if not field.init:
      continue
    if field.name not in raw:
      defaulted.append(f"{prefix}{field.name}")
      continue
    kwargs[field.name] = _coerce(
      hints[field.name], raw[field.name], defaulted, f"{prefix}{field.name}."
    )
  return cls(**kwargs)


def _coerce(hint: Any, value: Any, defaulted: list[str], prefix: str) -> Any:
  origin = get_origin(hint)
  if origin is tuple and isinstance(value, list):
    # Flat tuple fields only: a tuple nested inside another hint stays a list.
    return tuple(value)
  is_union = origin is Union or origin is types.UnionType
  candidates = get_args(hint) if is_union else (hint,)
  variants = [c for c in candidates if _is_cfg_type(c)]
  if variants and isinstance(value, dict):
    return _rebuild(_pick_variant(variants, value), value, defaulted, prefix)
  return value


def _is_cfg_type(hint: Any) -> bool:
  return isinstance(hint, type) and dataclasses.is_dataclass(hint)


def _pick_variant(variants: list[type], raw: dict[str, Any]) -> type:
  """Best field overlap with the serialized keys; ties go to the smallest
  variant, which is the exact match when a subclass merely extends it."""

  def score(variant: type) -> tuple[int, int]:
    names = {f.name for f in dataclasses.fields(variant)}
    return (len(names & raw.keys()), -len(names))

  return max(variants, key=score)
