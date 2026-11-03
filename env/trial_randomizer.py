"""3-source perturbation for trial randomization in EP²-MPC evaluation."""
import numpy as np
from typing import Optional, List, Tuple
import copy

from env.scenarios import ScenarioConfig


class TrialRandomizer:
    """Applies 3-source perturbation to scenario configurations.
    
    Sources:
    1. Start pose: ±15cm position, ±10° heading
    2. Obstacle positions: ±5cm per obstacle
    3. Initial velocity: v₀∈[0,0.08], ω₀∈[-0.05,0.05]
    """

    def __init__(
        self,
        pos_std: float = 0.15,
        heading_std_deg: float = 10.0,
        obs_pos_std: float = 0.05,
        v0_range: Tuple[float, float] = (0.0, 0.08),
        omega0_range: Tuple[float, float] = (-0.05, 0.05),
    ):
        self.pos_std = pos_std
        self.heading_std = np.deg2rad(heading_std_deg)
        self.obs_pos_std = obs_pos_std
        self.v0_range = v0_range
        self.omega0_range = omega0_range

    def randomize_start(
        self, base_start: np.ndarray, seed: Optional[int] = None
    ) -> np.ndarray:
        """Perturb start pose: ±pos_std position, ±heading_std heading.
        
        Returns:
            perturbed [x, y, theta]
        """
        rng = np.random.default_rng(seed)
        dx = rng.uniform(-self.pos_std, self.pos_std)
        dy = rng.uniform(-self.pos_std, self.pos_std)
        dtheta = rng.uniform(-self.heading_std, self.heading_std)
        return np.array([
            base_start[0] + dx,
            base_start[1] + dy,
            base_start[2] + dtheta if len(base_start) > 2 else dtheta
        ])

    def randomize_obstacles(
        self, base_obstacles: List, seed: Optional[int] = None
    ) -> List:
        """Perturb obstacle positions by ±obs_pos_std in x and y.
        
        Obstacle radius is kept unchanged.
        """
        rng = np.random.default_rng(seed)
        perturbed = []
        for obs in base_obstacles:
            obs_list = list(obs)
            obs_list[0] += rng.uniform(-self.obs_pos_std, self.obs_pos_std)  # cx
            obs_list[1] += rng.uniform(-self.obs_pos_std, self.obs_pos_std)  # cy
            perturbed.append(tuple(obs_list))
        return perturbed

    def randomize_initial_velocity(
        self, seed: Optional[int] = None
    ) -> Tuple[float, float]:
        """Sample random initial velocity (v₀, ω₀).
        
        v₀ ∈ [v0_min, v0_max], ω₀ ∈ [omega0_min, omega0_max]
        
        Returns:
            (v0, omega0)
        """
        rng = np.random.default_rng(seed)
        v0 = rng.uniform(self.v0_range[0], self.v0_range[1])
        omega0 = rng.uniform(self.omega0_range[0], self.omega0_range[1])
        return float(v0), float(omega0)

    def randomize_trial(
        self,
        scenario_config: ScenarioConfig,
        seed: Optional[int] = None
    ) -> Tuple[ScenarioConfig, float, float]:
        """Apply all 3 perturbation sources to a scenario.
        
        Uses sequential seeding to ensure reproducibility while varying each source.
        
        Args:
            scenario_config: base scenario
            seed: master random seed (None = random)
        
        Returns:
            (perturbed_scenario, v0, omega0)
        """
        rng = np.random.default_rng(seed)
        # Generate sub-seeds
        seed_start = int(rng.integers(0, 2**31))
        seed_obs = int(rng.integers(0, 2**31))
        seed_vel = int(rng.integers(0, 2**31))

        new_start = self.randomize_start(scenario_config.start, seed=seed_start)
        new_obstacles = self.randomize_obstacles(scenario_config.obstacles, seed=seed_obs)
        v0, omega0 = self.randomize_initial_velocity(seed=seed_vel)

        perturbed = ScenarioConfig(
            name=scenario_config.name,
            start=new_start,
            goal=scenario_config.goal.copy(),
            obstacles=new_obstacles,
            bounds=scenario_config.bounds,
            description=f"{scenario_config.description} [perturbed seed={seed}]"
        )
        return perturbed, v0, omega0
