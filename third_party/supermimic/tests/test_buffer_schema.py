"""``actor_obs`` is stored only where it is read back: the history warm-start.
Schema and writes share the ``_stores_actor_obs`` predicate, which PpoRunner
inherits from BaseRunner."""

from types import SimpleNamespace

from supermimic.runners.ppo_runner import PpoRunner
from supermimic.runners.supermimic_runner import BaseRunner


class _Runner(BaseRunner):
  """A runner holding only what ``_buffer_schema`` reads."""

  def __init__(self, actor_obs_mode: str, history_length: int) -> None:
    self._representation = SimpleNamespace(
      actor_obs_mode=actor_obs_mode, full_state_dim=71, proprio_obs_dim=lambda n: 64 + n
    )
    self._num_actions = 29
    self._sim_context = SimpleNamespace(output_dim=3)
    self.cfg = SimpleNamespace(history_length=history_length)


def test_actor_obs_stored_only_for_proprio_with_history():
  assert "actor_obs" not in _Runner("proprio", 1)._buffer_schema()
  assert "actor_obs" in _Runner("proprio", 4)._buffer_schema()
  assert "actor_obs" not in _Runner("rep_state", 4)._buffer_schema()
  assert PpoRunner._buffer_schema is BaseRunner._buffer_schema
  assert PpoRunner._stores_actor_obs is BaseRunner._stores_actor_obs
