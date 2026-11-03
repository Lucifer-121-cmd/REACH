# results/ — the quotable campaign data

This directory mirrors the **quotable subset** of the paper's validation
campaign from the lab server: `results/` here ≡
`runs/run_20260910T174433Z/` there (campaign of record, started
2026-09-10). Every number in `paper/reach_tcst.tex` traces through
`paper/truth_paper.md` §10 (provenance map) to a file in this tree.

## Contents

| path | what it is |
|---|---|
| `reach/` | The REACH experiments. `main3_adaptive_20260915`, `heldout3_adaptive_20260915`, `mismatch3_adaptive_20260915`, `sweep6_adaptive_20260915` are the **deployed** (adaptive-certificate) runs the headline tables quote; `main2/heldout2/mismatch2/sweep5` are the fixed-certificate historical comparison (input of the conservatism paired analysis). Plus: `certificates_*.json` (quantile fits + descent checks), `campaign_manifest_*.json`, `oracle/` (reach ceiling), `p3_validation/`, `p5_margin/` (mismatch-margin ladder), `hybrid_validation/`, `stepchange/` (B=280 price-50 re-run + `stepchange_results_B320.json` B=320), `threshold_baseline/` (return-cost baseline), `timing/` (planning timing), `reach_analysis.json` |
| `reach_maps/`, `reach_maps_adaptive/` | Trained energy-return maps (fixed and terrain-adaptive $\pi_{fb}$ fits) with their certificates |
| `figures_reach/` | The manuscript's 10 figures as rendered (script `27_reach_figures.py`), each with its `.provenance.json` sidecar (inputs, sha256, git state, argv). The copies used by the manuscript live in `paper/figures/`. Includes `fig_oracle_traj.pdf` (the oracle-decomposition trajectory figure, bit-identity verified) |
| `statistics/` | Comparator paired-campaign summaries (excluded from the REACH paper; preserved for a separate submission) |
| `reach/tau_match_control/` | The matched-τ control closing derivation-verify F-1 (truth_paper.md §1): planning model τ matched to the plant — 0/240 breaches, reach 9.43→9.22 m, docking 10.0→13.8% |
| `gate_e_timing/` | Reserved-core per-solve timing (REACH_ROADMAP Move 6) backing `fig_timing_survival` |
| `genome_reevolved/` | The comparator arm's **genome of record**: gen059 NSGA-II winners per scenario (pure NSGA-II, LLM disabled; searched 2026-09-06, 59 generations). Layout matches what `utils/checkpoint_utils.load_best_evolved_params` expects — pass `--evolution-root results/genome_reevolved` to reproduce the comparator arm |

## What is deliberately NOT here

- **Raw per-trial JSONL** (~575 MB): the comparator/baseline campaign trials and the
  timing trials. They live only on the lab server (`/data`); every quotable
  number is derivable from the aggregated files above. They can be
  regenerated with `scripts/run_campaign.py` + `scripts/02_run_baselines.py`
  (see REPRODUCE.md §8). The original price-1.0 step-change results are
  preserved as `stepchange_results_price1_OLD.json` and `stepchange_results_B320_price1_OLD.json`.
- **Quarantined / never-quote material**: the pre-fix (24 J placeholder
  certificate) runs, the sink sweeps, and the unpinned/rounded-intercept
  validation files. `paper/truth_paper.md` §10 names them so they are never
  quoted; they remain on the lab server only.
