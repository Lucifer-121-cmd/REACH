"""Three navigation scenarios for EP²-MPC evaluation.

Scenarios are loaded from YAML files in configs/scenarios/ when available,
falling back to built-in defaults only when the YAML file is absent.
"""
import logging
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Dict

logger = logging.getLogger(__name__)


@dataclass
class ScenarioConfig:
    """Configuration for a navigation scenario."""
    name: str
    start: np.ndarray          # [x, y, theta]
    goal: np.ndarray           # [x, y]
    obstacles: List            # list of (cx, cy, radius) tuples
    bounds: Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)
    description: str = ""
    #: REACH maps: terrain multiplier regions on c1/c4. ``None`` on
    #: every legacy scenario, which keeps the energy model -- and every
    #: tracked number -- bit-identical to the campaigns already measured.
    terrain: Optional[Any] = None     # env.terrain.TerrainField (duck-typed
                                      # here to avoid a hard import cycle)
    #: REACH: the return-to-dock point. Defaults to the start pose position,
    #: which on the out-and-back maps is the dock. Legacy navigation maps
    #: leave it None and their goal is the mission target.
    dock: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        # deepcopy-able and cheap: nothing to normalise. Terrain validation
        # happens in TerrainField's constructor, which is the loud place.
        pass


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def _yaml_scenario_path(name: str) -> str:
    """Return expected YAML path for a scenario, relative to repo root."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, 'configs', 'scenarios', f'{name}.yaml')


def load_scenario_from_yaml(yaml_path: str) -> ScenarioConfig:
    """Load a ScenarioConfig from a YAML file.

    Expected YAML keys: name, description, bounds, start, goal, obstacles.
    Each obstacle entry is a list [cx, cy, radius].
    """
    import yaml
    with open(yaml_path, encoding='utf-8') as fh:
        data = yaml.safe_load(fh)

    obstacles = [tuple(obs) for obs in data['obstacles']]
    bounds = tuple(data['bounds'])          # (xmin, ymin, xmax, ymax)
    start = np.array(data['start'], dtype=float)
    goal = np.array(data['goal'], dtype=float)

    terrain = None
    if data.get('terrain'):
        from env.terrain import TerrainField
        terrain = TerrainField.from_dict(data['terrain'], bounds=bounds)
    dock = (np.array(data['dock'], dtype=float) if data.get('dock') is not None
            else None)

    return ScenarioConfig(
        name=data['name'],
        start=start,
        goal=goal,
        obstacles=obstacles,
        bounds=bounds,
        description=data.get('description', ''),
        terrain=terrain,
        dock=dock,
    )


# ---------------------------------------------------------------------------
# Built-in fallback scenario factories
# ---------------------------------------------------------------------------

def _create_corridor_scenario() -> ScenarioConfig:
    """Straight corridor: 20 m × 4 m with 3 staggered circular obstacles."""
    obstacles = [
        (5.0,  1.0, 0.4),   # lower
        (10.0, 3.0, 0.4),   # upper
        (15.0, 1.0, 0.4),   # lower
    ]
    return ScenarioConfig(
        name='corridor',
        start=np.array([0.5, 2.0, 0.0]),
        goal=np.array([19.5, 2.0]),
        obstacles=obstacles,
        bounds=(0.0, 0.0, 20.0, 4.0),
        description='Straight corridor with 3 staggered obstacles',
    )


def _create_s_curve_scenario() -> ScenarioConfig:
    """S-curve slalom: 25 m × 3.5 m with 2 single-sided blockers."""
    obstacles = [
        (6.0,  3.7, 1.7),   # Gate 1: top blocker, bottom edge=1.8m, blocks center
        (14.0, 0.3, 1.7),   # Gate 2: bottom blocker, top edge=2.2m, blocks center
    ]
    return ScenarioConfig(
        name='s_curve',
        start=np.array([0.5, 3.5, 0.0]),
        goal=np.array([19.5, 0.5]),
        obstacles=obstacles,
        bounds=(0.0, 0.0, 20.0, 4.0),
        description='S-shaped slalom: 2 alternating gate blockers force true S-path',
    )


def _create_chicane_scenario() -> ScenarioConfig:
    """Chicane: two inward-facing gates forcing energy-intensive S-path. 20x4m arena"""
    obstacles = [
        # Gate 1 (x=7.0): bottom-mounted, gap at TOP (y=2.5 to 4.0)
        (7.0, 1.0, 1.5),
        # Gate 2 (x=13.0): top-mounted, gap at BOTTOM (y=0.0 to 1.5)
        (13.0, 3.0, 1.5),
    ]
    return ScenarioConfig(
        name='chicane',
        start=np.array([0.5, 2.0, 0.0]),
        goal=np.array([19.5, 2.0]),
        obstacles=obstacles,
        bounds=(0.0, 0.0, 20.0, 4.0),
        description='Chicane: two inward-facing gates forcing energy-intensive S-path. 20x4m arena',
    )


_FALLBACK_REGISTRY: Dict[str, callable] = {
    'corridor': _create_corridor_scenario,
    's_curve':  _create_s_curve_scenario,
    'chicane':   _create_chicane_scenario,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_scenario(name: str) -> ScenarioConfig:
    """Return a ScenarioConfig by name.

    Loads ``configs/scenarios/<name>.yaml`` when the file exists, and falls back
    to the built-in definition only when it does **not** exist. A YAML file that
    exists but fails to parse raises -- the previous bare ``except Exception:
    pass`` silently substituted different geometry.

    Args:
        name: one of ``'corridor'``, ``'s_curve'``, ``'chicane'``

    Returns:
        ScenarioConfig instance

    Raises:
        ValueError: unknown scenario name.
        Exception: whatever ``load_scenario_from_yaml`` raises for a malformed
            or unreadable YAML file, re-raised with the path attached.
    """
    name_lower = name.lower().replace('-', '_')
    if name_lower not in _FALLBACK_REGISTRY:
        available = list(_FALLBACK_REGISTRY.keys())
        raise ValueError(f"Unknown scenario '{name}'. Available: {available}")

    yaml_path = _yaml_scenario_path(name_lower)
    if os.path.isfile(yaml_path):
        try:
            return load_scenario_from_yaml(yaml_path)
        except Exception as exc:
            raise type(exc)(
                f"Failed to load scenario YAML {yaml_path}: {exc}. Refusing to "
                f"fall back to the built-in definition, which may specify "
                f"different geometry."
            ) from exc

    logger.warning(
        "No YAML at %s; using the built-in '%s' definition. Verify that "
        "configs/scenarios/ is present -- built-ins are not the tracked geometry "
        "of record.", yaml_path, name_lower,
    )
    return _FALLBACK_REGISTRY[name_lower]()


def list_scenarios() -> List[str]:
    """Return names of all available scenarios."""
    return list(_FALLBACK_REGISTRY.keys())
