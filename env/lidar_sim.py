"""36-ray LiDAR simulation for Pioneer 3-DX."""
import numpy as np
from typing import List, Optional, Tuple, Union


class LidarSim:
    """Simulates a 36-ray LiDAR sensor on the Pioneer 3-DX.
    
    Rays are evenly spaced over [0, 2π). Obstacles can be circles or rectangles.
    """

    def __init__(self, n_rays: int = 36, max_range: float = 8.0, noise_std: float = 0.02,
                 seed: Optional[int] = None):
        self.n_rays = n_rays
        self.max_range = max_range
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed)
        # Ray angles relative to robot heading: 0 = forward
        self.angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)

    def scan(self, robot_pose: np.ndarray, obstacles: List) -> np.ndarray:
        """Perform a LiDAR scan.
        
        Args:
            robot_pose: [x, y, theta] robot pose
            obstacles: list of obstacle descriptions
                       Circle: (cx, cy, radius)
                       Rectangle: (cx, cy, half_w, half_h) - axis-aligned
        
        Returns:
            ranges: array of n_rays range measurements (with Gaussian noise)
        """
        x_r, y_r, theta_r = robot_pose[0], robot_pose[1], robot_pose[2]
        ranges = np.full(self.n_rays, self.max_range)

        for i, rel_angle in enumerate(self.angles):
            ray_angle = theta_r + rel_angle
            dx = np.cos(ray_angle)
            dy = np.sin(ray_angle)
            ray_origin = np.array([x_r, y_r])
            ray_dir = np.array([dx, dy])

            min_t = self.max_range
            for obs in obstacles:
                t = self._intersect(ray_origin, ray_dir, obs)
                if t is not None and 0.0 < t < min_t:
                    min_t = t
            ranges[i] = min_t

        # Add Gaussian noise
        if self.noise_std > 0:
            noise = self._rng.normal(0.0, self.noise_std, self.n_rays)
            ranges = np.clip(ranges + noise, 0.01, self.max_range)

        return ranges

    def _intersect(self, origin: np.ndarray, direction: np.ndarray, obstacle) -> Optional[float]:
        """Compute ray-obstacle intersection distance. Returns None if no hit."""
        obs = tuple(obstacle)
        if len(obs) == 3:
            return self._ray_circle(origin, direction, obs[0], obs[1], obs[2])
        elif len(obs) == 4:
            return self._ray_rectangle(origin, direction, obs[0], obs[1], obs[2], obs[3])
        return None

    def _ray_circle(
        self, origin: np.ndarray, direction: np.ndarray,
        cx: float, cy: float, radius: float
    ) -> Optional[float]:
        """Ray-circle intersection using quadratic formula."""
        oc = origin - np.array([cx, cy])
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - radius * radius
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return None
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        if t1 > 1e-6:
            return float(t1)
        if t2 > 1e-6:
            return float(t2)
        return None

    def _ray_rectangle(
        self, origin: np.ndarray, direction: np.ndarray,
        cx: float, cy: float, half_w: float, half_h: float
    ) -> Optional[float]:
        """Ray-AABB intersection using slab method."""
        inv_dx = 1.0 / direction[0] if abs(direction[0]) > 1e-12 else np.inf
        inv_dy = 1.0 / direction[1] if abs(direction[1]) > 1e-12 else np.inf

        tx1 = (cx - half_w - origin[0]) * inv_dx
        tx2 = (cx + half_w - origin[0]) * inv_dx
        ty1 = (cy - half_h - origin[1]) * inv_dy
        ty2 = (cy + half_h - origin[1]) * inv_dy

        t_min = max(min(tx1, tx2), min(ty1, ty2))
        t_max = min(max(tx1, tx2), max(ty1, ty2))

        if t_max < 0 or t_min > t_max:
            return None
        t = t_min if t_min > 1e-6 else t_max
        if t > 1e-6:
            return float(t)
        return None

    def min_range(self, scan: np.ndarray) -> float:
        """Return minimum range from a scan."""
        return float(np.min(scan))

    def seed(self, seed: int):
        """Reseed the noise RNG for reproducibility."""
        self._rng = np.random.default_rng(seed)

    def get_obstacle_distances(self, scan: np.ndarray) -> np.ndarray:
        """Return the full range array (alias for scan output)."""
        return scan

    def get_forward_range(self, scan: np.ndarray) -> float:
        """Return the forward-facing range (ray index 0)."""
        return float(scan[0])
