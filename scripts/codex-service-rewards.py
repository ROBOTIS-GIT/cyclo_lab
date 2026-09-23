"""Apply launch-only service reward weights without accepting executable config."""
import json
import math
from pathlib import Path

OMY = {
    'end_effector_position_tracking': (-.5, 0),
    'end_effector_position_tracking_fine_grained': (0, .24),
    'end_effector_orientation_tracking': (-.24, 0),
}
LOCOMOTION = {
    'track_lin_vel_xy_exp': (0, 2), 'track_ang_vel_z_exp': (0, 2),
    'flat_orientation_l2': (-10, 0), 'action_rate_l2': (-.2, 0),
}
MIMIC = {
    'reference_anchor_position': (0, 1), 'reference_anchor_orientation': (0, 1),
    'reference_body_position': (0, 2), 'reference_body_orientation': (0, 2),
}
BOUNDS = {
    'Cyclo-Reach-OMY-v0': OMY,
    'Cyclo-Velocity-Flat-K1-Rev1-v0': LOCOMOTION,
    'Cyclo-Mimic-K1-Rev1-Dance1': MIMIC,
    'Cyclo-Mimic-K1-Rev1-Dance2': MIMIC,
}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate reward configuration key')
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError('Reward weight must be finite')


def apply_reward_file(config, task_id, filename):
    """Validate the whole bounded request before modifying any RewardTermCfg."""
    path = Path(filename)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 32768:
        raise ValueError('Invalid service reward file')
    payload = json.loads(path.read_text(), object_pairs_hook=unique_object, parse_constant=invalid_constant)
    if (not isinstance(payload, dict) or set(payload) != {'task_id', 'reward_terms'}
            or payload['task_id'] != task_id or task_id not in BOUNDS):
        raise ValueError('Reward configuration task mismatch')
    terms = payload['reward_terms']
    if not isinstance(terms, list) or len(terms) > len(BOUNDS[task_id]):
        raise ValueError('Invalid reward terms')
    changes, seen = [], set()
    for term in terms:
        if not isinstance(term, dict) or set(term) != {'term_id', 'weight'}:
            raise ValueError('Invalid reward term')
        name, weight = term['term_id'], term['weight']
        if not isinstance(name, str) or name not in BOUNDS[task_id] or name in seen:
            raise ValueError('Unknown or duplicate reward term')
        low, high = BOUNDS[task_id][name]
        if type(weight) not in (int, float) or not low <= weight <= high or not math.isfinite(weight):
            raise ValueError('Reward weight outside approved bounds')
        configured = getattr(config.rewards, name, None)
        if configured is None or not hasattr(configured, 'weight'):
            raise ValueError('Reward term unavailable in this runtime')
        seen.add(name)
        changes.append((configured, float(weight)))
    for configured, weight in changes:
        configured.weight = weight
    return len(changes)
