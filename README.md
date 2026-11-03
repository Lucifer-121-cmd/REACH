# REACH — Compliance by Construction

Energy-budgeted reach maximization in NMPC via a quantile-certified
return-cost constraint. A differential-drive mobile robot must travel as
far as its mission requires and still return: we enforce budget compliance
by construction — a hard terminal reserve constraint whose margin is sized
by a per-state 95th-percentile certificate of a named fallback policy's
realized excess return cost.

## Headline

**Reach arm: 0/480 hard-budget breaches** across nominal, held-out, and
±15% mismatch regimes vs priced-heuristic NMPC arms breaching **92/960**
(two-sided Fisher p = 1.8×10⁻¹⁷). The same certificate tightening that
raises the estimators' breaches (73→92 of 960 priced; 11→53 of 360 for a
plan-then-track oracle) leaves the constrained arms at zero.

## How it works

The controller is a mission-length NMPC (N=60, dt=0.5s) on a 6-state
differential-drive model with a terminal reserve constraint:

```
E_N + a_q + D_J(x_N) ≤ B
```

where `D_J` is a terrain-weighted energy-to-dock geodesic field and `a_q`
is a per-map intercept certified as the 95th percentile of a fallback
policy's realized excess return cost — a coverage construction, not a
regression fit.

## Repository structure

```
mpc/          NMPC controllers (reach, EP²-MPC, standard, adaptive horizon)
env/          Pioneer 3-DX simulator (LiDAR, terrain, trial randomizer)
evaluation/   Trial runner, baselines, paired statistics
planning/     Energy-aware Dijkstra (GATE-R1) + reach-map generation
evolution/    NSGA-II machinery (genome-of-record provenance)
scripts/      Pipeline stages 02–35 + run_campaign + figure generators
utils/        Paths, safe_write, provenance, config schema, preflight
results/      Campaign data: trials, certificates, maps, timing, figures
paper/        Manuscript sources (IEEE T-CST + Elsevier RAS versions)
tests/        262 tests (254 passed, 7 skipped, 1 collection item)
```

## Key results

| claim | value | source |
|---|---|---|
| Constrained arm breaches | **0/480** | `results/reach/main3_v2_const50/` |
| Priced-arm breaches (deployed) | **92/960** | pooled from all regimes |
| Fisher two-sided p | **1.8×10⁻¹⁷** | exact test on pooled counts |
| Old-cert breaches | 73/960 | `results/reach/main2_v2_const50/` |
| Oracle realisation | **64.5%** | `results/reach/oracle/oracle_results.json` |
| Certificate shrink (dial) | **29% mean** | `results/reach_maps_adaptive/` |
| Paired reach gain | **+0.63 m** (p=2.8×10⁻⁷) | `results/reach/p3_validation/` |
| Tracking deadline | **99.97%** of 46,991 solves | `results/gate_e_timing/` |
| Missions total | **6,420** | §VI inventory |

## Requirements

- Python 3.11+, CasADi 3.7.2, NumPy, SciPy
- IPOPT (bundled with CasADi)
- matplotlib (figures only)

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full campaign (150 workers recommended)
python scripts/run_campaign.py --run-id <id> --stage main

# Individual stages
python scripts/02_run_baselines.py --scenario corridor
python scripts/04_ablation.py --scenario corridor --study all
python scripts/05_statistical_eval.py --scenario corridor
```

**Warning:** several scripts write to fixed output paths and have
destroyed data when re-run. Read the guardrails in each script's
docstring before running.

## Tests

```bash
pytest tests/ -q
```

254 passed, 7 skipped (device/symlink guards that require the
lab-server `/data` mount).

## License

This project is for academic use. Contact the authors for reuse
permissions.
