# REPRODUCE — REACH paper results

Commission-for-commission everything the numbers in the paper rest on is shipped
in `results/`, so **verification needs no simulation**. Three tiers, shortest
first. Every command runs from the repo root as-is.

## 0. Environment

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python 3.11
pip install -r requirements.txt                      # pins: casadi 3.7.2, numpy 2.4.3, scipy 1.17.1
```

No GPU; IPOPT+MUMPS are bundled inside the CasADi wheel (no system solver).
Simulation output root is set with:

```bash
export REACH_NMPC_RUNS_DIR="$PWD/runs-data"   # created on demand
```

## Tier A — verify the paper's numbers from the shipped raw sinks (minutes)

Full test suite:

```bash
pytest tests/
```

The pre-declared **hard-budget breach rule** is `energy_spent > B + 0.5 J` — a
single definition, fixed before running any campaign. Re-derive every headline
breach count with it directly from the shipped JSONL sinks:

```bash
python3 - <<'PY'
import json
from collections import defaultdict

def counts(paths):
    n, b = defaultdict(int), defaultdict(int)
    for p in paths:
        for line in open(p):
            d = json.loads(line)
            n[d['arm']] += 1
            b[d['arm']] += d['energy_spent'] > d['B'] + 0.5
    return n, b

RES = 'results/reach'
old_cert = [f'{RES}/main2_v2_const50/reach_trials_train.jsonl',     # const@50 re-run
            f'{RES}/heldout2_final_20260914/reach_trials_heldout.jsonl',
            f'{RES}/mismatch2_20260914/reach_trials_train.jsonl']
deployed = [f'{RES}/main3_v2_const50/reach_trials_train.jsonl',     # const@50 merged mains
            f'{RES}/heldout3_adaptive_20260915/reach_trials_heldout.jsonl',
            f'{RES}/mismatch3_adaptive_20260915/reach_trials_train.jsonl']

for name, paths in (('old certificate ', old_cert), ('deployed certificate', deployed)):
    n, b = counts(paths)
    cn, cb = n['const_weight'], b['const_weight']       # old-cert runs carry const@50 in main2_v2..
    dn, db = n['dual'], b['dual']
    print(f"{name}: const {cb}/{cn}  dual {db}/{dn}  priced {cb+db}/{cn+dn}  reach {b['reach']}/{n['reach']}")
PY
```

For the plan-then-track oracle breach counts:

```bash
python3 - <<'PY'
import json
for f, name in (('oracle/prefix2_oracle_20260915.json', 'old cert '),
                ('oracle/oracle_results.json',         'deployed ')):
    d = json.load(open('results/reach/' + f))
    t = d['trials']
    b = sum(1 for x in t if x['energy_spent'] > x['B'] + 0.5)
    print(f"{name}: oracle breaches {b}/{d['n_missions']}")
PY
```

Expected output (matches the paper exactly):

| tier | const_weight | dual | priced pooled | reach (constrained) |
|---|---|---|---|---|
| old certificate | 37/480 | 36/480 | **73/960** | 0/480 |
| deployed certificate | 52/480 | 40/480 | **92/960** | 0/480 |
| plan-then-track oracle | — | — | **11 → 53 / 360** | — |

For the full per-cell aggregate view (dock/violation/collision/reach per
arm × map × budget), point the aggregator at the shipped sinks:

```bash
python scripts/21a_reach_analyze.py --run-id run_20260910T174433Z \
  --main-sink   results/reach/main3_v2_const50/reach_trials_train.jsonl \
  --heldout-sink results/reach/heldout3_adaptive_20260915/reach_trials_heldout.jsonl
```

(If your checkout lacks the runs tree, `21a` still prints the tables straight
from these flag paths; it also writes `reach_analysis.json` under
`$REACH_NMPC_RUNS_DIR/run_20260910T174433Z/reach/`.)

## Tier B — regenerate the paper's figures from the shipped artefacts (minutes)

The figure scripts read a runs-style layout. Rebuild it with symlinks — the
shipped `results/` tree plays the role of the run:

```bash
export REACH_NMPC_RUNS_DIR="$PWD/runs-data"
mkdir -p runs-data/run_20260910T174433Z
cd runs-data/run_20260910T174433Z
ln -s ../../results/reach           reach
ln -s ../../results/reach_maps      reach_maps
ln -s ../../results/reach_maps_adaptive reach_maps_adaptive
ln -s ../../results/figures_reach   figures_reach
ln -s ../../results/statistics       statistics
cd ../..

python scripts/27_reach_figures.py --run-id run_20260910T174433Z   # Figs 1–6, 8–11
python scripts/33_oracle_trajectory_figure.py --run-id run_20260910T174433Z   # Fig 7
# sanity check a single figure first if you like:  --only enforce  (stem names: enforce, frontier, stepchange, timing, power, conservatism, gamma, map)
```

Output lands in `runs-data/run_20260910T174433Z/figures_reach/`, each figure
with a `.provenance.json` sidecar naming its exact inputs. The crops in
`paper/figures/` are byte-identical to the renderers' outputs.

## Tier C — full re-run (hours to ~1 day on an 8-core desktop)

Fixed seeds and pre-declared grids throughout; a fresh run-id replays the whole
pipeline. The main grid's const arm runs at the sweep-selected price 50.0
(driver `scripts/21` sets it in the main stage; that override is what the
paper's headline numbers use).

```bash
RID=repro_$(date +%Y%m%d)

# 1. maps: 9 train + 3 heldout out-and-back maps with terrain J-fields
python planning/reach_maps.py --run-id $RID            # deterministic from a fixed seed

# 2. certificates: per-map intercept a_q, quantile coverage 0.950 (pre-declared)
python scripts/20_reach_certificate_terrain.py --run-id $RID

# 3. campaigns — pilot first (minutes; validates the install end-to-end)
python scripts/21_reach_campaign.py --run-id $RID --stage pilot
python scripts/21_reach_campaign.py --run-id $RID --stage main        # mains @ const@50
python scripts/21_reach_campaign.py --run-id $RID --stage sweep       # const-price box
python scripts/21_reach_campaign.py --run-id $RID --stage heldout     # unseen maps
python scripts/21_reach_campaign.py --run-id $RID --stage mismatch    # ±15% power mismatch
# flags: --max-workers 8 | --n-seeds 5 | --dry-run | --skip-preflight | --aggregate-only

# 4. analysis: sweep selection + headline tables
python scripts/21a_reach_analyze.py --run-id $RID

# 5. experiments backing the remaining tables/figures:
python scripts/24_reach_oracle.py --run-id $RID          # plan-then-track oracle
python scripts/25_descent_check.py --run-id $RID         # descent audit of the fit
python scripts/26_p3_validation.py --run-id $RID         # the dial (Tables 3–4, Figs 8–9)
python scripts/29_threshold_baseline.py --run-id $RID    # threshold family (Fig 6)
python scripts/30_p5_margin.py --run-id $RID             # P5 margin ladder
python scripts/31_hybrid_validation.py --run-id $RID     # hybrid (Table 5)
python scripts/32_tau_match_control.py --run-id $RID     # tau-matched control
python scripts/22_reach_stepchange.py --run-id $RID --budget 280   # step-change B=280
python scripts/22_reach_stepchange.py --run-id $RID --budget 320   # step-change B=320 robustness
python scripts/23_reach_timing.py --run-id $RID          # two-tier compute benchmark (Fig 11)

# 6. figures
python scripts/27_reach_figures.py --run-id $RID
python scripts/33_oracle_trajectory_figure.py --run-id $RID
```

Every stage writes a `stage_manifest.json` recording arm definitions, seeds,
and per-task results; JSONL sinks append, so an interrupted stage resumes
without losing data. `scripts/21_reach_campaign.py --help` lists all flags.

## Determinism notes

- Seeds come from the campaign planner (`--seed0`, default 31000), never from
  worker state or wall-clock — re-running a stage with the same run-id and
  flags repeats the same campaign plan.
- Maps regenerate deterministically (fixed seed inside `planning/reach_maps.py`).
- The tracking tier runs a wall-clock 100 ms deadline: **timing numbers are
  machine-dependent**; breach/violation/collision counts are not.
- IPOPT is configured from fixed option blocks in `mpc/`; nothing persists
  across missions.
- Across different BLAS builds/hardware, IPOPT's last-digit floating-point
  drift can flip an occasional mission that sits exactly on a decision
  boundary; the paper's aggregate effects (73→92, 92-of-960 vs 0/480, +0.63 m
  dial) are orders of magnitude larger than such drift.

## Layout

```
results/reach/              all main/heldout/mismatch/oracle/threshold/hybrid/p3/p5
                            JSONL sinks + stage manifests + certificates
results/reach_maps(_adaptive)/   generated maps + J-fields (+ adaptive intercepts)
results/figures_reach/      manuscript figures + .provenance.json sidecars
results/gate_e_timing/      tracking-tier deadline benchmark (Fig 11)
mpc/                        ReachMPC controllers (reach / EP^2 / baselines)
planning/                   geodesic J-field + map generator + Dijkstra
evaluation/                 mission runner, arm definitions, statistics
```
