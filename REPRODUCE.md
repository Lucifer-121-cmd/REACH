# REPRODUCE — REACH paper results

Commands to reproduce every number in the manuscript (all tiers), shortest-first.
The repo ships all raw campaign artefacts (`results/`), so verification of the
headline numbers needs **no simulation**.

## 0. Environment

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python 3.11
pip install -r requirements.txt                      # CasADi, NumPy, SciPy, matplotlib
python -c "import casadi; print(casadi.__version__)" # >= 3.7
```

No GPU. Solver: IPOPT (bundled with CasADi). Simulation output root defaults to
a `/data` path — override for your machine:

```bash
export REACH_NMPC_RUNS_DIR="$PWD/runs-data"   # created on demand
```

Everything below is run from the repo root.

## Tier A — verify shipped results (minutes, no simulation)

```bash
pytest tests/                                  # full suite, all must pass
```

Re-derive the headline tables from the shipped JSONL sinks (same aggregator
the paper used):

```bash
python scripts/21a_reach_analyze.py --run-id main3_adaptive_20260915
python scripts/21a_reach_analyze.py --run-id main2_final_20260914
```

Headline numbers to check against the manuscript (`results/reach/`):

| claim | source artefact |
|---|---|
| constrained arm 0/480 breaches | `main2_final_20260914` + `main3_adaptive_20260915`, `reach_trials_*.jsonl` |
| priced arms 92/960, Fisher p = 1.8e-17 | same runs (const@50 re-run: `...sweep5_20260914`) |
| tightening experiment: priced 73→92, oracle 11→53 | `main2_*` vs `main3_*`, `oracle/oracle_results.json` |
| dial: −29% intercepts, +0.63 m reach, Wilcoxon p = 2.75e-7 | `certificates_adaptive_p3.json`, `p3_validation/p3_validation.json` |
| hybrid boundary pair 0/60 vs 10/60 | `hybrid_validation/` |
| timing: 0.03% over 100 ms | `gate_e_timing/gate_e_timing_s_curve.json`, `reach/timing/reach_planning_timing.json` |

## Tier B — regenerate figures from shipped artefacts (minutes)

```bash
python scripts/27_reach_figures.py --run-id run_20260910T174433Z   # Figs 1–6, 8–11
python scripts/33_oracle_trajectory_figure.py                      # Fig 7
```

Every figure is written with a `.provenance.json` sidecar naming the exact
input files. Renderer output lands under `runs-data/.../figures_reach/`;
the manuscript copies in `paper/figures/` are byte-identical to these.

## Tier C — full re-run (hours to ~1 day on an 8-core desktop)

Fixed seeds and pre-declared grids throughout; a fresh run-id replays the
whole pipeline:

```bash
RID=repro_$(date +%Y%m%d)

# 1. maps: 9 train + 3 heldout out-and-back maps with terrain J-fields
python planning/reach_maps.py --run-id $RID                # deterministic from a fixed seed

# 2. certificates: per-map intercept a_q, quantile coverage 0.950 (pre-declared)
python scripts/20_reach_certificate_terrain.py --run-id $RID

# 3. campaigns — pilot first (minutes; validates the install end-to-end)
python scripts/21_reach_campaign.py --run-id $RID --stage pilot
# then the paper grids (4 arms x budget grid x seeds; hours each):
python scripts/21_reach_campaign.py --run-id $RID --stage main       # train maps
python scripts/21_reach_campaign.py --run-id $RID --stage sweep      # const-price box
python scripts/21_reach_campaign.py --run-id $RID --stage heldout    # unseen maps
python scripts/21_reach_campaign.py --run-id $RID --stage mismatch   # ±15% power mismatch
# useful flags: --max-workers 8  |  --n-seeds 5  |  --dry-run  |  --skip-preflight

# 4. analysis: sweep selection (rule fixed before the data) + headline tables
python scripts/21a_reach_analyze.py --run-id $RID

# 5. experiments backing the remaining tables/figures (each self-contained,
#    reads the run-id's own campaign outputs):
python scripts/24_reach_oracle.py --run-id $RID          # plan-then-track oracle (Fig 7 / oracle)
python scripts/25_descent_check.py --run-id $RID         # descent audit of the fit
python scripts/26_p3_validation.py --run-id $RID         # the dial: terrain-adaptive refit (Tables 3–4, Figs 8–9)
python scripts/29_threshold_baseline.py --run-id $RID    # threshold-family sweep (Fig 6)
python scripts/30_p5_margin.py --run-id $RID             # P5 margin ladder
python scripts/31_hybrid_validation.py --run-id $RID     # plan-then-track hybrid (Table 5)
python scripts/32_tau_match_control.py --run-id $RID     # tau-matched control
python scripts/22_reach_stepchange.py --run-id $RID --budget 280   # step-change B=280
python scripts/22_reach_stepchange.py --run-id $RID --budget 320   # step-change B=320 (robustness)
python scripts/23_reach_timing.py --run-id $RID          # two-tier compute benchmark (Fig 11)

# 6. figures + final tables
python scripts/27_reach_figures.py --run-id $RID
python scripts/33_oracle_trajectory_figure.py --run-id $RID
```

`scripts/21_reach_campaign.py --help` lists every flag. Each stage writes a
`stage_manifest.json` recording arm definitions, seeds, and per-task results;
JSONL sinks append, so an interrupted stage resumes without losing data.

## Determinism notes

- Seeds are assigned by the campaign planner (`--seed0`, default 31000), never
  by the worker; re-running a stage with the same run-id and flags repeats it.
- Maps regenerate deterministically (fixed generator seed in
  `planning/reach_maps.py`).
- The tracking tier runs a wall-clock deadline (100 ms); timing numbers are
  machine-dependent — breach/violation/collision counts are not.
- IPOPT is configured from fixed option blocks (`mpc/`); no hidden state
  persists across missions.

## Layout

```
results/reach/          all campaign JSONL sinks + manifests + certificates
results/reach_maps(_adaptive)/   generated maps + J-fields (+ adaptive intercepts)
results/figures_reach/  manuscript figures + provenance sidecars
mpc/                    ReachMPC controllers (reach / EP^2 / baselines)
planning/               geodesic J-field + map generator + Dijkstra
evaluation/             mission runner, arm definitions, statistics
```
