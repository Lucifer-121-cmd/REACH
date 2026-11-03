"""Randomised out-and-back REACH maps with a train/held-out split.

Why these maps exist: on corridor/s_curve/chicane the mission is a 20 m drag
race -- start at one end, goal at the other, no decision about *how far* to
go. REACH's claim is about the turn-back point: maximise outbound reach
subject to retaining enough energy to return to the dock. That needs:

* a **dock** the robot departs from and must return to (scenario ``goal`` IS
  the dock -- the geodesic field built to ``goal`` then measures
  energy-to-dock everywhere, which is exactly the reserve certificate's
  currency);
* **long routes** where the round-trip cost to the far end exceeds the
  budget, so the reserve constraint -- not the wall -- decides the turn-back
  point (rejection-sampled: ``2·E*(far end) >= reserve_headroom × B_max``);
* **terrain heterogeneity** (regions on c1/c4) so the min-energy route
  differs from the shortest route with 2–3× authority instead of the
  homogeneous 0.35%;
* **straight-line blocked**: without it, a sampled map is corridor's drag
  race again and the geodesic detour the certificate prices never happens;
* **≤ MAX_OBS = 6 obstacles**, so the NLP parameter structure -- and every
  timing comparison against the legacy maps -- stays unchanged.

The generator is deterministic from a seed. Splits never share seeds: the
held-out maps are not a re-labelling of the training maps, which is what a
generalisation test actually requires (train/held-out split from disjoint
seed ranges).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from env.scenarios import ScenarioConfig
from env.terrain import TerrainField, TerrainRegion
from mpc.energy_model import load_default_coefficients
from planning.energy_dijkstra import build_field

logger = logging.getLogger(__name__)

MAX_OBS = 6            # NLP structure cap
R_BODY = 0.25
DOCK_CLEARANCE = 1.5   # dock this far from any inflated obstacle (m)
#: Surface-to-surface gap between obstacles. 0.6 m (the first draft) left
#: only ~0.1 m of body-centre clearance after 2×R_BODY inflation -- a
#: knife-edge channel for a 0.05 m planning grid, and the closed-loop smoke
#: test clipped an obstacle in exactly such a gap. 1.8 m leaves ≥ 0.55 m of
#: centre clearance BEYOND the planner's 0.35 m hard-clearance buffer
#: (hard_clearance was raised 0.20 → 0.35 on 2026-09-12 after a return leg
#: rode the bound into the top wall; the gap follows it): passable with
#: margin, still a real detour.
OBSTACLE_GAP = 1.8
DEFAULT_B_MAX = 300.0  # J; the largest budget the REACH experiments will run
RESERVE_HEADROOM = 1.3 # round-trip-to-far-end must exceed B_max by this factor


@dataclass
class MapSpec:
    """Everything the rejection sampler enforces, in one place."""
    seed: int
    split: str                  # 'train' | 'heldout'
    length: float = 30.0
    width: float = 6.0
    n_obstacles: Optional[int] = None     # None → sampled in [2, MAX_OBS]
    n_terrain: Optional[int] = None       # None → sampled in [1, 3]
    m_range: Tuple[float, float] = (1.3, 3.0)
    b_max: float = DEFAULT_B_MAX
    # filled during generation:
    rejections: int = 0
    rejection_reasons: Optional[Dict[str, int]] = None


def _segment_obstacle_clearance(p0, p1, cx, cy) -> float:
    """Distance from obstacle centre (cx, cy) to segment p0→p1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L2 = float(d @ d)
    t = 0.0 if L2 == 0 else float(np.clip((np.array([cx, cy]) - p0) @ d / L2, 0, 1))
    return float(np.hypot(*(p0 + t * d - np.array([cx, cy]))))


def _sample_map(spec: MapSpec, rng: np.random.Generator) -> ScenarioConfig:
    """One raw sample: dock, obstacles, terrain. No rejection logic here."""
    L, W = spec.length, spec.width
    dock = np.array([1.0, W / 2.0])
    n_obs = spec.n_obstacles if spec.n_obstacles is not None else int(rng.integers(2, MAX_OBS + 1))
    n_terr = spec.n_terrain if spec.n_terrain is not None else int(rng.integers(1, 4))

    obstacles: List[Tuple[float, float, float]] = []
    far_x = L - 2.0
    for _ in range(n_obs):
        for _ in range(200):        # placement attempts, then give up (caller rejects)
            cx = float(rng.uniform(4.0, far_x))
            cy = float(rng.uniform(1.0, W - 1.0))
            r = float(rng.uniform(0.5, 1.5))
            ok = (math.hypot(cx - dock[0], cy - dock[1]) > DOCK_CLEARANCE + r + R_BODY)
            ok &= all(math.hypot(cx - ox, cy - oy) > r + orad + OBSTACLE_GAP
                      for ox, oy, orad in obstacles)
            ok &= (cx - r > 3.0) and (cx + r < L - 1.0)   # keep dock zone & far end open
            if ok:
                obstacles.append((cx, cy, r))
                break

    regions: List[TerrainRegion] = []
    for _ in range(n_terr):
        for _ in range(200):        # placement attempts; overlap = retry, not crash
            # Terrain bands: full-height or partial, never covering the dock zone
            x0 = float(rng.uniform(3.0, L - 6.0))
            span = float(rng.uniform(4.0, 10.0))
            x1 = min(x0 + span, L - 1.0)
            if rng.random() < 0.5:
                y0, y1 = 0.0, W                        # full-height band
            else:
                y0 = float(rng.uniform(0.0, W - 2.0))
                y1 = min(y0 + float(rng.uniform(2.0, W - y0)), W)
            m = float(rng.uniform(*spec.m_range))
            cand = TerrainRegion(x0, y0, x1, y1, m1=m, m4=m)
            # non-overlap is a TerrainField invariant; sample until it holds
            if all(a.x0 >= cand.x1 or cand.x0 >= a.x1
                   or a.y0 >= cand.y1 or cand.y0 >= a.y1 for a in regions):
                regions.append(cand)
                break
        # unable to place this region non-overlapping: fewer regions, not an
        # invalid field. A map with zero regions is terrain-free and still
        # subject to every rejection criterion.

    terrain = TerrainField(regions, bounds=(0.0, 0.0, L, W)) if regions else None
    return ScenarioConfig(
        name=f"reach_{spec.split}_{spec.seed}",
        start=np.array([dock[0], dock[1], 0.0]),
        goal=dock.copy(),                 # the geodesic field measures energy-to-dock
        obstacles=obstacles,
        bounds=(0.0, 0.0, L, W),
        description=(f"generated out-and-back REACH map (seed {spec.seed}, "
                     f"split {spec.split}); goal IS the dock"),
        terrain=terrain,
        dock=dock.copy(),
    )


def _check_map(sc: ScenarioConfig, spec: MapSpec,
               coeffs: Dict[str, float]) -> Tuple[bool, str]:
    """Every rejection criterion, named. Returns (ok, reason)."""
    L, W = sc.bounds[2], sc.bounds[3]
    dock = np.asarray(sc.dock)

    if not sc.obstacles:
        return False, "no_obstacles"
    # 1. start/goal clearance: dock clear of every inflated obstacle
    for cx, cy, r in sc.obstacles:
        if math.hypot(cx - dock[0], cy - dock[1]) < DOCK_CLEARANCE + r + R_BODY:
            return False, "dock_clearance"
    # 2. straight-line blocked: the +x ray from the dock must be obstructed
    far = np.array([L - 1.0, dock[1]])
    if all(_segment_obstacle_clearance(dock, far, cx, cy) > r + R_BODY
           for cx, cy, r in sc.obstacles):
        return False, "straight_line_open"
    # 3. route feasibility: the terrain-weighted field to the dock must be
    #    finite at the far end (an open route exists both ways)
    try:
        f = build_field(sc, coeffs=coeffs, terrain=sc.terrain)
    except ValueError:
        return False, "field_unbuildable"
    if not math.isfinite(f.dstar(float(far[0]), float(far[1]))):
        return False, "far_end_unreachable"
    # 4. the reserve must decide the turn-back: round-trip energy to the far
    #    end above b_max by the headroom factor. On a 'J' field D IS joules;
    #    on a metric field scale by lambda_min.
    e_far = f.dstar(float(far[0]), float(far[1]))
    if f.denomination == "m":
        e_far *= f.lambda_min
    if 2.0 * e_far < RESERVE_HEADROOM * spec.b_max:
        return False, "turnback_not_interior"
    # 5. obstacles must not merge into a wall: the free graph must let the
    #    geodesic thread between SOME pair of obstacles (cheap proxy: the
    #    geodesic path from far end to dock is longer than the straight line
    #    by a detour but the field was finite -- already guaranteed by 3;
    #    here we only require at least 2 obstacles so a detour exists at all)
    if len(sc.obstacles) < 2:
        return False, "no_detour_possible"
    return True, "ok"


def generate_map(spec: MapSpec,
                 coeffs: Optional[Dict[str, float]] = None,
                 max_attempts: int = 500) -> Tuple[ScenarioConfig, MapSpec]:
    """Rejection-sample one map. Returns (scenario, spec-with-counters)."""
    coeffs = coeffs or load_default_coefficients()
    spec.rejection_reasons = {}
    for attempt in range(max_attempts):
        # reseed per attempt so each rejection resamples EVERYTHING; the
        # attempt counter is part of the seed, so a map is reproducible from
        # (seed, attempt) alone
        cand = _sample_map(spec, np.random.default_rng(spec.seed * 10_000 + attempt))
        ok, reason = _check_map(cand, spec, coeffs)
        if ok:
            spec.rejections = attempt
            return cand, spec
        spec.rejection_reasons[reason] = spec.rejection_reasons.get(reason, 0) + 1
    raise RuntimeError(
        f"seed {spec.seed}: no valid map in {max_attempts} attempts; "
        f"rejection reasons: {spec.rejection_reasons}. Loosen the MapSpec "
        "(length, obstacle radii, terrain range) rather than raising "
        "max_attempts -- a map that only passes one time in ten thousand is "
        "not a map the generator understands.")


def scenario_to_yaml_dict(sc: ScenarioConfig) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "name": sc.name,
        "description": sc.description,
        "bounds": list(sc.bounds),
        "start": [float(v) for v in sc.start],
        "goal": [float(v) for v in sc.goal],
        "obstacles": [[float(c), float(y), float(r)] for c, y, r in sc.obstacles],
    }
    if sc.terrain is not None:
        d["terrain"] = sc.terrain.to_dict()
    if sc.dock is not None:
        d["dock"] = [float(v) for v in sc.dock]
    return d


def generate_split(out_dir: str, *, n_train: int, n_heldout: int,
                   seed0: int = 20260911,
                   coeffs: Optional[Dict[str, float]] = None,
                   **spec_kwargs) -> Dict[str, Any]:
    """Generate both splits into ``out_dir`` and write a manifest.

    Train and held-out maps come from DISJOINT seed ranges, so no held-out
    map is a re-labelled training map (Reviewer 9's generalisation question
    is answered by the split, not by the count).
    """
    import yaml
    from utils.safe_write import safe_write_json
    os.makedirs(out_dir, exist_ok=True)
    manifest: Dict[str, Any] = {"seed0": seed0, "splits": {}, "specs": {}}
    for split, n, offset in (("train", n_train, 0), ("heldout", n_heldout, 100_000)):
        split_dir = os.path.join(out_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        entries = []
        for i in range(n):
            spec = MapSpec(seed=seed0 + offset + i, split=split, **spec_kwargs)
            sc, spec = generate_map(spec, coeffs=coeffs)
            path = os.path.join(split_dir, f"reach_{split}_{spec.seed}.yaml")
            with open(path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(scenario_to_yaml_dict(sc), fh, sort_keys=False)
            entries.append({"seed": spec.seed, "file": os.path.basename(path),
                            "rejections": spec.rejections,
                            "n_obstacles": len(sc.obstacles),
                            "terrain": (sc.terrain.describe()
                                        if sc.terrain else None)})
            logger.info("%s: %s (rejections=%d, %d obstacles, %s)",
                        split, entries[-1]["file"], spec.rejections,
                        len(sc.obstacles), entries[-1]["terrain"])
        manifest["splits"][split] = entries
    # the full spec is part of the record: a map without its sampling box is
    # not reproducible from the seed alone
    manifest["specs"] = {
        "length": spec_kwargs.get("length", 30.0),
        "width": spec_kwargs.get("width", 6.0),
        "b_max": spec_kwargs.get("b_max", DEFAULT_B_MAX),
        "m_range": list(spec_kwargs.get("m_range", (1.3, 3.0))),
        "seed0": seed0,
        "train_seed_range": [seed0, seed0 + n_train - 1],
        "heldout_seed_range": [seed0 + 100_000, seed0 + 100_000 + n_heldout - 1],
    }
    mpath = os.path.join(out_dir, "manifest.json")
    safe_write_json(mpath, manifest)
    return manifest


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default=None,
                    help="write to <runs>/<run-id>/reach_maps/ on /data")
    ap.add_argument("--out-dir", default=None,
                    help="explicit output directory (alternative to --run-id)")
    ap.add_argument("--n-train", type=int, default=4)
    ap.add_argument("--n-heldout", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=20260911)
    ap.add_argument("--length", type=float, default=30.0)
    ap.add_argument("--width", type=float, default=6.0)
    ap.add_argument("--b-max", type=float, default=DEFAULT_B_MAX)
    args = ap.parse_args(argv)

    if args.out_dir:
        out = args.out_dir
    elif args.run_id:
        from utils.paths import run_dir
        out = run_dir(args.run_id, "reach_maps")
    else:
        raise SystemExit("give --run-id or --out-dir (there is no default: "
                         "generated maps are campaign artefacts, not repo files)")
    m = generate_split(out, n_train=args.n_train, n_heldout=args.n_heldout,
                       seed0=args.seed0, length=args.length,
                       width=args.width, b_max=args.b_max)
    n = {s: len(v) for s, v in m["splits"].items()}
    print(f"generated {n} into {out} (manifest.json records seeds, "
          f"rejections and terrain per map)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
