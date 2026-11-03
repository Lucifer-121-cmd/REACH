#!/usr/bin/env bash
# Campaign driver. Runs the bulk-safe half of RUNBOOK.md §4 unattended.
#
# Designed to be launched inside tmux and survive a lost SSH/VS Code connection.
# Every stage writes its own log under the run directory on /data, so nothing is
# lost if the terminal goes away and progress is inspectable at any point.
#
#   Usage:  scripts/launch_campaign.sh <RUN_ID> <WORKERS>
#
# What it deliberately does NOT run: the TIMING stages (RUNBOOK §4 step 5).
# As of 2026-09-05 the box IS quiet (load 1.2, 99.5% idle, 0 foreign python), so
# these are runnable for the first time -- but they take ~11.3 h of exclusive
# machine time and must not be started as a side effect of a bulk campaign:
#   CPUS=$(python -m utils.preflight --mode timing --suggest-cpus 4)
#   python -m utils.preflight --mode timing --cpus $CPUS
#   python scripts/13_horizon_isolation.py timing --run-id <RUN_ID> --cpus $CPUS ...

set -uo pipefail

RUN_ID="${1:?usage: launch_campaign.sh <RUN_ID> <WORKERS>}"
WORKERS="${2:?usage: launch_campaign.sh <RUN_ID> <WORKERS>}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate

LOGDIR="$(python -c "from utils.paths import run_dir; print(run_dir('$RUN_ID','logs'))")"
SUMMARY="$LOGDIR/SUMMARY.txt"
: > "$SUMMARY"

say() { printf '%s  %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$SUMMARY"; }

# Run one stage. Records exit code and elapsed time; never aborts the script, so
# a late failure cannot discard the stages that already succeeded.
stage() {
  local name="$1"; shift
  local log="$LOGDIR/${name}.log"
  local t0 rc el
  say "START  $name"
  t0=$(date +%s)
  nice -n 19 "$@" >"$log" 2>&1
  rc=$?
  el=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ]; then
    say "OK     $name  (${el}s)  log: ${name}.log"
  else
    say "FAIL   $name  rc=$rc  (${el}s)  log: ${name}.log"
    tail -n 15 "$log" | sed 's/^/           | /' | tee -a "$SUMMARY"
  fi
  printf '%s\t%s\t%s\n' "$name" "$rc" "$el" >> "$LOGDIR/stage_results.tsv"
  return $rc
}

say "=========================================================="
say "REACH-nmpc campaign  run-id=$RUN_ID  workers=$WORKERS"
say "host=$(hostname)  git=$(git rev-parse --short HEAD)$([ -n "$(git status --porcelain)" ] && echo '-dirty')"
say "output=$(python -c "from utils.paths import run_dir; print(run_dir('$RUN_ID'))")"
say "        -> $(python -c "import os;from utils.paths import run_dir; print(os.path.realpath(run_dir('$RUN_ID')))")"
say "loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
say "=========================================================="

# ---- 1. GATE-R1: free, arithmetic, decides REACH vs REALTIME --------------
stage "01_geodesic_fields" python -m planning.energy_dijkstra --run-id "$RUN_ID"
stage "02_gate_r1_reach"   python scripts/14_gate_r1_reach.py --run-id "$RUN_ID"
GATE_R1_RC=$?
stage "03_vfb_decomposition" python prototypes/audit_vfb_decomposition.py \
      --cells --stride 4 --out "runs/$RUN_ID/gate_r1/vfb_decomposition.json"
if [ "$GATE_R1_RC" -ne 0 ]; then
  say "GATE-R1 did NOT pass. PLAN.md §3 says switch to REALTIME rather than"
  say "engineering around it. Continuing with the measurement stages anyway --"
  say "they are needed under both routes -- but do not start REACH work."
fi

# ---- 2. bulk stages, fanned out -------------------------------------------
stage "04_preflight_bulk" python -m utils.preflight --mode bulk --max-workers "$WORKERS"
for S in gate_r2 ep2 baselines; do
  stage "05_dryrun_$S" python scripts/run_campaign.py --run-id "$RUN_ID" \
        --stage "$S" --max-workers "$WORKERS" --dry-run
done
for S in gate_r2 ep2 baselines; do
  stage "06_campaign_$S" python scripts/run_campaign.py --run-id "$RUN_ID" \
        --stage "$S" --max-workers "$WORKERS"
done

# ---- 3. GATE-E solvability: load-insensitive, single process ---------------
stage "07_gate_e_solvability" python scripts/13_horizon_isolation.py solvability \
      --run-id "$RUN_ID" --scenario s_curve --n-trials 50

# ---- 4. statistics --------------------------------------------------------
stage "08_statistics" python scripts/05_statistical_eval.py --scenario all \
      --run-id "$RUN_ID"
# 05_statistical_eval reports total_trials=250 but runs every inferential test on
# ONE master seed (n=50), which PLAN.md's own power analysis calls underpowered
# 4.7-5.8x for the 0.16 J target effect. The campaign already wrote 250
# seed-aligned pairs per cell, so this re-does the tests properly at n=250.
# Analysis only: reads the JSONL, no solver, seconds.
stage "08b_paired_n250" python scripts/16_paired_campaign_tests.py \
      --run-id "$RUN_ID"

# ---- 5. C1 re-runs: the geometry fix invalidated these inputs -------------
stage "09_trajectories" python scripts/collect_trajectories.py --run-id "$RUN_ID"
stage "10_ablation" python scripts/04_ablation.py --scenario all --study all \
      --run-id "$RUN_ID"

# ---- 5a. aggregate baseline JSONs -----------------------------------------
# `--stage baselines` above writes per-trial JSONL for the two campaign arms.
# The tables and figures readers want the 5-arm AGGREGATE, baseline_<scen>.json,
# which only 02_run_baselines.py writes (it is the one caller of
# collect_all_baselines, which adds energy_penalized_mpc and the N=24/N=50 arms).
# Serial per scenario, so run the three maps concurrently -- they write different
# files into the same directory.
for S in corridor s_curve chicane; do
  stage "10b_baselines02_$S" python scripts/02_run_baselines.py --scenario "$S" \
        --n_trials 50 --run-id "$RUN_ID" &
done
wait
say "OK     10b_baselines02 (all three maps)"

# ---- 6. figures and tables ------------------------------------------------
stage "11_figures" python scripts/07_figures.py \
      --statistics-dir  "runs/$RUN_ID/statistics" \
      --ablation-dir    "runs/$RUN_ID/ablation" \
      --evolution-root  results \
      --trajectory-file "runs/$RUN_ID/trajectories/trajectory_data.json" \
      --run-id "$RUN_ID" --allow-missing
# FOUR explicit sources, same as the figures. --evolution-root is the TRACKED
# results/ on purpose: Pareto and HV are search-space quantities C1 does not
# invalidate. Everything else must come from this run, or the EP2 row and the
# baseline rows describe different code.
stage "12_tables" python summarize_paper_tables.py \
      --statistics-dir "runs/$RUN_ID/statistics" \
      --baselines-dir  "runs/$RUN_ID/baselines" \
      --ablation-dir   "runs/$RUN_ID/ablation" \
      --evolution-root results \
      --run-id "$RUN_ID"

say "=========================================================="
say "DONE. Stage results:"
column -t "$LOGDIR/stage_results.tsv" 2>/dev/null | sed 's/^/  /' | tee -a "$SUMMARY" \
  || cat "$LOGDIR/stage_results.tsv" | tee -a "$SUMMARY"
# "DONE" means the driver reached the end, NOT that every stage passed: stage()
# deliberately never aborts, so a late failure cannot discard earlier successes.
# The first campaign printed DONE with four stages broken (baselines rc=1,
# trajectories rc=1, ablation rc=2, figures rc=1). Read the codes, not the banner.
BAD=$(awk -F'\t' '$2 != 0 {print $1" rc="$2}' "$LOGDIR/stage_results.tsv")
if [ -n "$BAD" ]; then
  say ""
  say "STAGES THAT FAILED -- do not treat this run as complete:"
  printf '%s\n' "$BAD" | sed 's/^/           ! /' | tee -a "$SUMMARY"
else
  say ""
  say "All stages exited 0."
fi
say ""
say "TIMING STAGES WERE NOT RUN -- see the header for the reserved-core commands."
say "=========================================================="
