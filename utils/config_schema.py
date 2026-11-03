"""Schema definition and validation for EP²-MPC YAML configuration files."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check(ok: bool, msg: str, errors: List[str]) -> None:
    if not ok:
        errors.append(msg)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _validate_field(path: str, val: Any, rules: Dict[str, Any],
                    errors: List[str]) -> None:
    ftype = rules.get('type')
    nullable = rules.get('nullable', False)

    if val is None:
        if not nullable:
            errors.append(f"'{path}' must not be null")
        return

    if ftype == 'int':
        if not (isinstance(val, int) and not isinstance(val, bool)):
            errors.append(
                f"'{path}' must be an integer, got {type(val).__name__}")
            return
        val_num: Any = val
    elif ftype == 'float':
        if not _is_number(val):
            errors.append(
                f"'{path}' must be a number, got {type(val).__name__}")
            return
        val_num = val
    elif ftype == 'bool':
        _check(isinstance(val, bool),
               f"'{path}' must be a boolean, got {type(val).__name__}", errors)
        return
    elif ftype == 'str':
        _check(isinstance(val, str),
               f"'{path}' must be a string, got {type(val).__name__}", errors)
        choices = rules.get('choices')
        if choices and isinstance(val, str):
            _check(val in choices,
                   f"'{path}' must be one of {choices!r}, got {val!r}", errors)
        return
    elif ftype == 'list':
        _check(isinstance(val, list),
               f"'{path}' must be a list, got {type(val).__name__}", errors)
        length = rules.get('length')
        if length is not None and isinstance(val, list):
            _check(len(val) == length,
                   f"'{path}' must have {length} elements, got {len(val)}", errors)
        return
    else:
        return

    # Numeric range checks (int / float only)
    mn = rules.get('min')
    mx = rules.get('max')
    excl_min = rules.get('excl_min', False)
    if mn is not None:
        if excl_min:
            _check(val_num > mn,
                   f"'{path}' must be > {mn}, got {val_num}", errors)
        else:
            _check(val_num >= mn,
                   f"'{path}' must be >= {mn}, got {val_num}", errors)
    if mx is not None:
        _check(val_num <= mx,
               f"'{path}' must be <= {mx}, got {val_num}", errors)


def _validate_section(cfg: dict, section: str, fields: Dict[str, Any],
                      errors: List[str]) -> None:
    if section not in cfg:
        errors.append(f"Missing required section: '{section}'")
        return
    sec = cfg[section]
    if not isinstance(sec, dict):
        errors.append(
            f"Section '{section}' must be a mapping, got {type(sec).__name__}")
        return
    for field, rules in fields.items():
        required = rules.get('required', True)
        if field not in sec:
            if required:
                errors.append(f"'{section}.{field}' is required but missing")
            continue
        _validate_field(f"{section}.{field}", sec[field], rules, errors)


# ---------------------------------------------------------------------------
# Per-section field rules
# ---------------------------------------------------------------------------

_ROBOT_FIELDS: Dict[str, Any] = {
    'mass':       {'type': 'float', 'min': 0, 'excl_min': True},
    'wheel_radius': {'type': 'float', 'min': 0, 'excl_min': True},
    'wheelbase':  {'type': 'float', 'min': 0, 'excl_min': True},
    'max_v':      {'type': 'float', 'min': 0, 'excl_min': True},
    'max_omega':  {'type': 'float', 'min': 0, 'excl_min': True},
    'max_dv':     {'type': 'float', 'min': 0, 'excl_min': True},
    'max_domega': {'type': 'float', 'min': 0, 'excl_min': True},
}

_LIDAR_FIELDS: Dict[str, Any] = {
    'n_rays':    {'type': 'int',   'min': 1},
    'max_range': {'type': 'float', 'min': 0, 'excl_min': True},
    'noise_std': {'type': 'float', 'min': 0},
}

_ENERGY_FIELDS: Dict[str, Any] = {
    'c1': {'type': 'float', 'min': 0, 'excl_min': True},
    'c2': {'type': 'float', 'min': 0, 'excl_min': True},
    'c3': {'type': 'float', 'min': 0, 'excl_min': True},
    'c4': {'type': 'float', 'min': 0, 'excl_min': True},
    'c5': {'type': 'float', 'min': 0, 'excl_min': True},
}

_MPC_FIELDS: Dict[str, Any] = {
    'dt':             {'type': 'float', 'min': 0, 'excl_min': True},
    'N_min':          {'type': 'int',   'min': 1},
    'N_max':          {'type': 'int',   'min': 1},
    'N_default':      {'type': 'int',   'min': 1},
    'ipopt_max_iter': {'type': 'int',   'min': 1},
    'ipopt_tol':      {'type': 'float', 'min': 0, 'excl_min': True},
    'warm_start':     {'type': 'bool'},
}

_ADAPTIVE_HORIZON_FIELDS: Dict[str, Any] = {
    'beta_d':            {'type': 'float', 'min': 0, 'excl_min': True},
    'beta_e':            {'type': 'float', 'min': 0, 'excl_min': True},
    'd_thresh_default':  {'type': 'float', 'min': 0, 'excl_min': True},
    'dP_thresh_default': {'type': 'float', 'min': 0, 'excl_min': True},
    'w_power':           {'type': 'float', 'min': 0, 'excl_min': True},
}

_DEFAULT_WEIGHTS_FIELDS: Dict[str, Any] = {
    'w_p':     {'type': 'float', 'min': 0, 'excl_min': True},
    'w_dP':    {'type': 'float', 'min': 0, 'excl_min': True},
    'w_E':     {'type': 'float', 'min': 0, 'excl_min': True},
    'w_omega': {'type': 'float', 'min': 0, 'excl_min': True},
    'w_d':     {'type': 'float', 'min': 0, 'excl_min': True},
    'd_safe':  {'type': 'float', 'min': 0, 'excl_min': True},
    'w_goal':  {'type': 'float', 'min': 0, 'excl_min': True},
    'w_t':     {'type': 'float', 'min': 0, 'excl_min': True},
    'w_du':    {'type': 'float', 'min': 0, 'excl_min': True},
}

_EVOLUTION_FIELDS: Dict[str, Any] = {
    'pop_size':                    {'type': 'int',   'min': 1},
    'generations':                 {'type': 'int',   'min': 1},
    'seed':                        {'type': 'int',   'nullable': True},
    'hv_reference':                {'type': 'list',  'length': 4},
    'stagnation_window':           {'type': 'int',   'min': 1},
    'stagnation_threshold':        {'type': 'float', 'min': 0, 'excl_min': True},
    'injection_count':             {'type': 'int',   'min': 0},
    'sbx_eta':                     {'type': 'float', 'min': 0, 'excl_min': True},
    'mutation_eta':                {'type': 'float', 'min': 0, 'excl_min': True},
    'cosine_similarity_threshold': {'type': 'float', 'min': 0, 'max': 1},
    'llm_params_mode':             {'type': 'str',   'choices': ['llm', 'random']},
    'use_llm_params':              {'type': 'bool'},
}

_LLM_FIELDS: Dict[str, Any] = {
    'ollama_bin':                 {'type': 'str'},
    'model_strategy':             {'type': 'str'},
    'model_params':               {'type': 'str'},
    'timeout':                    {'type': 'float', 'min': 0, 'excl_min': True},
    'max_retries':                {'type': 'int',   'min': 1},
    'availability_cache_ttl_sec': {'type': 'int',   'min': 0},
    'base_url':                   {'type': 'str', 'required': False},
    'base_url_strategy':          {'type': 'str', 'required': False},
    'base_url_params':            {'type': 'str', 'required': False},
}

_EVALUATION_FIELDS: Dict[str, Any] = {
    'n_trials':          {'type': 'int',   'min': 1},
    'n_seeds':           {'type': 'int',   'min': 1},
    'timeout_per_trial': {'type': 'float', 'min': 0, 'excl_min': True},
}

_SECTIONS: Dict[str, Dict[str, Any]] = {
    'robot':            _ROBOT_FIELDS,
    'lidar':            _LIDAR_FIELDS,
    'energy_model':     _ENERGY_FIELDS,
    'mpc':              _MPC_FIELDS,
    'adaptive_horizon': _ADAPTIVE_HORIZON_FIELDS,
    'default_weights':  _DEFAULT_WEIGHTS_FIELDS,
    'evolution':        _EVOLUTION_FIELDS,
    'llm':              _LLM_FIELDS,
    'evaluation':       _EVALUATION_FIELDS,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_config(cfg: Any) -> None:
    """Validate the top-level EP²-MPC configuration dictionary.

    Parameters
    ----------
    cfg:
        The configuration object returned by ``yaml.safe_load``.

    Raises
    ------
    ValueError
        If *cfg* is not a mapping, any required section or field is absent,
        a field has the wrong type, a value is outside its allowed range, or
        a cross-field constraint is violated.  All violations are collected
        and reported together in a single exception.
    """
    if not isinstance(cfg, dict):
        raise ValueError(
            f"Config must be a YAML mapping, got {type(cfg).__name__}")

    errors: List[str] = []

    for section, fields in _SECTIONS.items():
        _validate_section(cfg, section, fields, errors)

    # Cross-field constraints: N_min < N_max and N_min <= N_default <= N_max
    mpc = cfg.get('mpc') or {}
    if isinstance(mpc, dict):
        n_min = mpc.get('N_min')
        n_max = mpc.get('N_max')
        n_def = mpc.get('N_default')
        if all(isinstance(v, int) for v in [n_min, n_max, n_def]
               if v is not None):
            if n_min is not None and n_max is not None and n_min >= n_max:
                errors.append(
                    f"'mpc.N_min' ({n_min}) must be < 'mpc.N_max' ({n_max})")
            if n_def is not None and n_min is not None and n_def < n_min:
                errors.append(
                    f"'mpc.N_default' ({n_def}) must be >= 'mpc.N_min' ({n_min})")
            if n_def is not None and n_max is not None and n_def > n_max:
                errors.append(
                    f"'mpc.N_default' ({n_def}) must be <= 'mpc.N_max' ({n_max})")

    # Top-level scalar field
    if 'results_dir' not in cfg:
        errors.append("Missing required field: 'results_dir'")
    elif not isinstance(cfg['results_dir'], str):
        errors.append(
            f"'results_dir' must be a string, "
            f"got {type(cfg['results_dir']).__name__}")

    if errors:
        bullet_list = "\n  - ".join(errors)
        raise ValueError(
            f"Config validation failed with {len(errors)} error(s):\n"
            f"  - {bullet_list}")
