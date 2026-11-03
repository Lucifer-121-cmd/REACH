"""Tests for utils.config_schema and utils.config_loader."""
import copy
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_schema import validate_config
from utils.config_loader import load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_real_config() -> dict:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for candidate in [
        os.path.join(repo_root, 'configs', 'global_config.yaml'),
        os.path.join(repo_root, 'config.yaml'),
    ]:
        if os.path.isfile(candidate):
            with open(candidate) as fh:
                return yaml.safe_load(fh)
    raise FileNotFoundError("No config file found")


# ---------------------------------------------------------------------------
# Tests — real configs
# ---------------------------------------------------------------------------

def test_real_config_passes_validation():
    """The shipped YAML configs must pass schema validation without errors."""
    cfg = _load_real_config()
    validate_config(cfg)  # must not raise


def test_load_config_returns_non_empty_dict():
    result = load_config()
    assert isinstance(result, dict)
    assert result  # non-empty


def test_load_config_accepts_config_path_keyword():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(repo_root, 'configs', 'global_config.yaml')
    result = load_config(config_path=cfg_path)
    assert isinstance(result, dict)
    assert result


def test_load_config_accepts_legacy_path_keyword():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(repo_root, 'configs', 'global_config.yaml')
    result = load_config(path=cfg_path)
    assert isinstance(result, dict)
    assert result


# ---------------------------------------------------------------------------
# Tests — missing required section / field
# ---------------------------------------------------------------------------

def test_missing_section_raises():
    cfg = _load_real_config()
    del cfg['robot']
    with pytest.raises(ValueError, match="Missing required section: 'robot'"):
        validate_config(cfg)


def test_missing_field_raises():
    cfg = _load_real_config()
    del cfg['mpc']['dt']
    with pytest.raises(ValueError, match="mpc.dt"):
        validate_config(cfg)


def test_missing_results_dir_raises():
    cfg = _load_real_config()
    del cfg['results_dir']
    with pytest.raises(ValueError, match="results_dir"):
        validate_config(cfg)


# ---------------------------------------------------------------------------
# Tests — wrong type
# ---------------------------------------------------------------------------

def test_wrong_type_int_field_raises():
    cfg = _load_real_config()
    cfg['mpc']['N_min'] = "eight"
    with pytest.raises(ValueError, match="N_min"):
        validate_config(cfg)


def test_wrong_type_float_field_raises():
    cfg = _load_real_config()
    cfg['robot']['mass'] = "heavy"
    with pytest.raises(ValueError, match="mass"):
        validate_config(cfg)


def test_wrong_type_bool_field_raises():
    cfg = _load_real_config()
    cfg['mpc']['warm_start'] = "yes"
    with pytest.raises(ValueError, match="warm_start"):
        validate_config(cfg)


def test_wrong_type_results_dir_raises():
    cfg = _load_real_config()
    cfg['results_dir'] = 123
    with pytest.raises(ValueError, match="results_dir"):
        validate_config(cfg)


# ---------------------------------------------------------------------------
# Tests — out-of-range values
# ---------------------------------------------------------------------------

def test_negative_mass_raises():
    cfg = _load_real_config()
    cfg['robot']['mass'] = -1.0
    with pytest.raises(ValueError, match="mass"):
        validate_config(cfg)


def test_zero_n_rays_raises():
    cfg = _load_real_config()
    cfg['lidar']['n_rays'] = 0
    with pytest.raises(ValueError, match="n_rays"):
        validate_config(cfg)


def test_zero_pop_size_raises():
    cfg = _load_real_config()
    cfg['evolution']['pop_size'] = 0
    with pytest.raises(ValueError, match="pop_size"):
        validate_config(cfg)


# ---------------------------------------------------------------------------
# Tests — cross-field constraints
# ---------------------------------------------------------------------------

def test_n_min_ge_n_max_raises():
    cfg = _load_real_config()
    cfg['mpc']['N_min'] = 20
    cfg['mpc']['N_max'] = 10
    with pytest.raises(ValueError, match="N_min"):
        validate_config(cfg)


def test_n_min_eq_n_max_raises():
    cfg = _load_real_config()
    cfg['mpc']['N_min'] = 12
    cfg['mpc']['N_max'] = 12
    with pytest.raises(ValueError, match="N_min"):
        validate_config(cfg)


def test_n_default_below_n_min_raises():
    cfg = _load_real_config()
    cfg['mpc']['N_min'] = 10
    cfg['mpc']['N_max'] = 24
    cfg['mpc']['N_default'] = 5
    with pytest.raises(ValueError, match="N_default"):
        validate_config(cfg)


def test_n_default_above_n_max_raises():
    cfg = _load_real_config()
    cfg['mpc']['N_min'] = 8
    cfg['mpc']['N_max'] = 20
    cfg['mpc']['N_default'] = 25
    with pytest.raises(ValueError, match="N_default"):
        validate_config(cfg)


# ---------------------------------------------------------------------------
# Tests — nullable / optional
# ---------------------------------------------------------------------------

def test_nullable_seed_none_is_valid():
    cfg = _load_real_config()
    cfg['evolution']['seed'] = None
    validate_config(cfg)  # must not raise


def test_integer_seed_is_valid():
    cfg = _load_real_config()
    cfg['evolution']['seed'] = 42
    validate_config(cfg)  # must not raise


# ---------------------------------------------------------------------------
# Tests — string choices
# ---------------------------------------------------------------------------

def test_invalid_llm_params_mode_raises():
    cfg = _load_real_config()
    cfg['evolution']['llm_params_mode'] = 'invalid_mode'
    with pytest.raises(ValueError, match="llm_params_mode"):
        validate_config(cfg)


def test_valid_llm_params_mode_random():
    cfg = _load_real_config()
    cfg['evolution']['llm_params_mode'] = 'random'
    validate_config(cfg)  # must not raise


# ---------------------------------------------------------------------------
# Tests — non-dict config
# ---------------------------------------------------------------------------

def test_non_dict_config_raises():
    with pytest.raises(ValueError, match="Config must be a YAML mapping"):
        validate_config("not a dict")


def test_none_config_raises():
    with pytest.raises(ValueError, match="Config must be a YAML mapping"):
        validate_config(None)
