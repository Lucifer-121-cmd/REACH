"""Pioneer 3-DX robot simulation environment.

Includes first-order motor lag dynamics and path distance tracking.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.lidar_sim import LidarSim
from mpc.energy_model import compute_power, load_default_coefficients


class PioneerEnv:
    """Pioneer 3-DX simulation with motor lag and cumulative energy.

    Dynamics include first-order velocity lag::

        v_{k+1} = v_k + alpha_v   * (v_cmd   - v_k)
        w_{k+1} = w_k + alpha_w   * (w_cmd   - w_k)
        x_{k+1} = x_k + v_k * cos(theta_k) * dt
        y_{k+1} = y_k + v_k * sin(theta_k) * dt
        theta_{k+1} = theta_k + w_k * dt
        E_{k+1} = E_k + P(v_k, w_k) * dt

    Where alpha_v = 1 - exp(-dt / tau_v) etc.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.dt = self.config.get('dt', 0.1)
        self.v_max = self.config.get('max_v', 1.2)
        self.omega_max = self.config.get('max_omega', 3.14159)
        self.mass = self.config.get('mass', 9.0)
        self.wheel_radius = self.config.get('wheel_radius', 0.095)
        self.wheelbase = self.config.get('wheelbase', 0.330)
        self.coeffs = self.config.get('energy_coeffs', load_default_coefficients())
        self.collision_radius = self.config.get('collision_radius', 0.25)

        # Motor lag
        self.tau_v = self.config.get('tau_v', 0.3)
        self.tau_omega = self.config.get('tau_omega', 0.2)
        self.alpha_v = 1.0 - np.exp(-self.dt / self.tau_v)
        self.alpha_omega = 1.0 - np.exp(-self.dt / self.tau_omega)

        # LiDAR
        # The scan RNG must be seeded per trial. LidarSim.seed() existed
        # from the start but nothing ever called it, so with noise_std = 0.02 the
        # scan stream came from OS entropy and NO trial was bit-reproducible --
        # not across machines, not even across two runs on this one.
        lidar_cfg = self.config.get('lidar', {})
        self.lidar = LidarSim(
            n_rays=lidar_cfg.get('n_rays', 36),
            max_range=lidar_cfg.get('max_range', 8.0),
            noise_std=lidar_cfg.get('noise_std', 0.02),
            seed=lidar_cfg.get('seed'),
        )
        self._lidar_seed = lidar_cfg.get('seed')

        # Internal state
        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0
        self._v = 0.0
        self._omega = 0.0
        self._cumulative_energy = 0.0
        self._path_distance = 0.0
        self._step_count = 0
        self._trajectory: List[Tuple[float, float, float]] = []

        # Scenario
        self.obstacles: List = []
        self.bounds: Optional[Tuple[float, float, float, float]] = None
        self._in_collision = False

        # Terrain multiplier field (None on every legacy scenario,
        # which keeps their energy -- and every tracked number -- unchanged).
        self._terrain = None

        # Plant/model mismatch and actuation noise. Both default
        # OFF. When on, the PLANT runs with perturbed coefficients and noisy
        # commands while the controller plans with the nominal model -- the
        # condition under which a reserve constraint finally defends against
        # something real (without mismatch, E_k is accumulated
        # with the same P used to plan and the receding-horizon element
        # protects against nothing).
        self._power_mismatch = 0.0          # fractional scale on all c_i
        self._actuation_sigma = (0.0, 0.0)  # (v_cmd, omega_cmd) Gaussian sigma
        self._actuation_rng = np.random.default_rng(None)

    # ------------------------------------------------------------------
    # Reset / step
    # ------------------------------------------------------------------

    def reset(self, start_pose: Optional[np.ndarray] = None,
              lidar_seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset to *start_pose* = [x, y, theta].

        Args:
            start_pose: [x, y, theta] to reset to.
            lidar_seed: reseed the scan RNG so this trial is
                bit-reproducible. Pass the trial seed. Omitting it leaves the
                stream running, which is only correct for a single episode.
        """
        if start_pose is not None:
            self._x = float(start_pose[0])
            self._y = float(start_pose[1])
            self._theta = float(start_pose[2]) if len(start_pose) > 2 else 0.0
        else:
            self._x = self._y = self._theta = 0.0

        if lidar_seed is not None:
            self.lidar.seed(int(lidar_seed))
            self._lidar_seed = int(lidar_seed)

        self._v = 0.0
        self._omega = 0.0
        self._cumulative_energy = 0.0
        self._path_distance = 0.0
        self._step_count = 0
        self._trajectory = [(self._x, self._y, self._theta)]
        self._in_collision = False
        return self.get_state()

    @property
    def lidar_seed(self) -> Optional[int]:
        """Seed the scan RNG was last reset with (None = unseeded, not reproducible)."""
        return self._lidar_seed

    def step(self, v_cmd: float, omega_cmd: float
             ) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """Apply motor command and advance by dt.

        Motor lag: actual velocity tracks commanded with first-order dynamics.
        Position integrates the *actual* (lagged) velocity.

        Returns (state, reward, done, info).
        """
        v_cmd = float(np.clip(v_cmd, -self.v_max, self.v_max))
        omega_cmd = float(np.clip(omega_cmd, -self.omega_max, self.omega_max))

        # Actuation noise, applied AFTER clipping so the plant can
        # still be driven slightly past the nominal limits -- a real motor
        # does not know the controller's saturation either.
        if self._actuation_sigma != (0.0, 0.0):
            sv, sw = self._actuation_sigma
            v_cmd += float(self._actuation_rng.normal(0.0, sv))
            omega_cmd += float(self._actuation_rng.normal(0.0, sw))

        # Store previous position for path distance
        x_prev, y_prev = self._x, self._y

        # Motor lag dynamics
        self._v += self.alpha_v * (v_cmd - self._v)
        self._omega += self.alpha_omega * (omega_cmd - self._omega)

        # Kinematic integration (uses actual velocity, not command)
        self._x += self._v * np.cos(self._theta) * self.dt
        self._y += self._v * np.sin(self._theta) * self.dt
        self._theta += self._omega * self.dt
        self._theta = float(np.arctan2(np.sin(self._theta),
                                       np.cos(self._theta)))

        # Energy -- terrain- and mismatch-scaled at the CURRENT position
        # (the plant pays for the floor it is standing on).
        P = compute_power(self._v, self._omega,
                          self._plant_coeffs(self._x, self._y))
        self._cumulative_energy += P * self.dt

        # Path distance
        self._path_distance += np.hypot(self._x - x_prev, self._y - y_prev)

        self._step_count += 1
        self._trajectory.append((self._x, self._y, self._theta))
        self._in_collision = self.check_collision()

        state = self.get_state()
        reward = -P * self.dt
        done = self._in_collision
        info = {'power': P, 'step': self._step_count}
        return state, reward, done, info

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Return current robot state as a dictionary of named fields."""
        return {
            'x': self._x,
            'y': self._y,
            'theta': self._theta,
            'v': self._v,
            'omega': self._omega,
            'cumulative_energy': self._cumulative_energy,
            'path_distance': self._path_distance,
            'step': self._step_count,
        }

    def get_state_vector(self) -> np.ndarray:
        """6-D state: [x, y, theta, v, omega, E]."""
        return np.array([self._x, self._y, self._theta,
                         self._v, self._omega, self._cumulative_energy])

    def get_lidar_scan(self) -> np.ndarray:
        """Perform a LiDAR scan from the current robot pose and return range measurements."""
        pose = np.array([self._x, self._y, self._theta])
        return self.lidar.scan(pose, self.obstacles)

    # ------------------------------------------------------------------
    # Collision & goal
    # ------------------------------------------------------------------

    def check_collision(self) -> bool:
        """Return True if the robot is currently colliding with an obstacle or boundary."""
        pos = np.array([self._x, self._y])
        for obs in self.obstacles:
            if len(obs) >= 3:
                dist = np.hypot(pos[0] - obs[0], pos[1] - obs[1])
                if dist < obs[2] + self.collision_radius:
                    return True
            else:
                if np.hypot(pos[0] - obs[0], pos[1] - obs[1]) < self.collision_radius:
                    return True
        if self.bounds is not None:
            xmin, ymin, xmax, ymax = self.bounds
            m = self.collision_radius
            if (self._x < xmin + m or self._x > xmax - m
                    or self._y < ymin + m or self._y > ymax - m):
                return True
        return False

    def get_min_obstacle_distance(self) -> float:
        """Return the minimum range from the current LiDAR scan (closest obstacle distance)."""
        return float(self.lidar.min_range(self.get_lidar_scan()))

    def is_goal_reached(self, goal: np.ndarray, tolerance: float = 0.2) -> bool:
        """Return True if the robot is within *tolerance* metres of *goal*."""
        return np.hypot(self._x - goal[0], self._y - goal[1]) < tolerance

    def set_scenario(self, obstacles: List,
                     bounds: Optional[Tuple] = None,
                     terrain: Optional[Any] = None):
        """Set the obstacle list, optional map bounds, and optional terrain.

        ``terrain`` is an ``env.terrain.TerrainField``; when given, the
        plant's power at each step scales ``c1``/``c4`` by the field's
        multipliers at the robot's position, so the min-energy route through
        heterogeneous terrain is genuinely different from the shortest one.
        Legacy callers omit it and get the nominal model.
        """
        self.obstacles = obstacles
        self.bounds = bounds
        self._terrain = terrain

    # -- plant/model mismatch hooks ------------------------------------------

    def set_power_mismatch(self, delta: float) -> None:
        """Scale the PLANT's energy coefficients by ``1 + delta``.

        The controller still plans with the nominal coefficients; that gap
        is the point. ``delta = 0.15`` means the robot really consumes 15%
        more power than the controller believes. The base coefficients are
        never mutated -- a second call replaces, not compounds.
        """
        self._power_mismatch = float(delta)

    def set_actuation_noise(self, sigma_v: float, sigma_omega: float,
                            seed: Optional[int] = None) -> None:
        """Per-step Gaussian noise on (v_cmd, omega_cmd), seeded per
        trial for reproducibility."""
        self._actuation_sigma = (float(sigma_v), float(sigma_omega))
        self._actuation_rng = np.random.default_rng(seed)

    def _plant_coeffs(self, x: float, y: float) -> Dict[str, float]:
        """Effective plant coefficients at (x, y): nominal × terrain ×
        mismatch. One place decides, so the env's energy and any test that
        recomputes it agree by construction."""
        c = dict(self.coeffs)
        if self._terrain is not None:
            m1, m4 = self._terrain.multipliers(x, y)
            c['c1'] *= m1
            c['c4'] *= m4
        if self._power_mismatch:
            for k in ('c1', 'c2', 'c3', 'c4', 'c5'):
                c[k] *= (1.0 + self._power_mismatch)
        return c

    def get_path_distance(self) -> float:
        """Total path distance traveled so far (metres)."""
        return self._path_distance

    def get_trajectory(self) -> np.ndarray:
        """Return trajectory as (n, 3) array of [x, y, theta]."""
        return np.array(self._trajectory)
