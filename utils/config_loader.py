"""Shared YAML configuration loader with schema validation for EP²-MPC."""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None, **kwargs) -> dict:
    """Load and validate the EP²-MPC YAML configuration.

    Searches for a configuration file in this order:

    1. *config_path* if provided.
    2. ``configs/global_config.yaml`` relative to the repository root.
    3. ``config.yaml`` relative to the repository root.

    Parameters
    ----------
    config_path:
        Optional explicit path to a YAML configuration file.
    **kwargs:
        Backward-compatibility shim. Supports legacy ``path=...`` keyword.

    Returns
    -------
    dict
        Validated configuration dictionary.

    Raises
    ------
    ValueError
        If the loaded configuration fails schema validation.
    """
    legacy_path = kwargs.pop('path', None)
    if kwargs:
        unknown = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unexpected keyword argument(s): {unknown}")
    if config_path is None and legacy_path is not None:
        config_path = legacy_path

    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed; returning empty config.")
        return {}

    from utils.config_schema import validate_config

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates: list = []
    if config_path:
        candidates.append(config_path)
    candidates += [
        os.path.join(repo_root, 'configs', 'global_config.yaml'),
        os.path.join(repo_root, 'config.yaml'),
    ]

    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, encoding='utf-8') as fh:
                    cfg = yaml.safe_load(fh)
            except Exception as exc:
                logger.warning("Could not read %s: %s", path, exc)
                continue
            validate_config(cfg)
            logger.debug("Loaded and validated config from %s", path)
            return cfg

    logger.warning("No config file found; returning empty defaults.")
    return {}
