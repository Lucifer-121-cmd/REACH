"""Defect #7 re-run: the main-grid const_weight arm at price 50.

WHY (2026-10-04 raw-data audit, found pre-submission):
    The campaign driver's main stage omitted the price0=50 override for
    const_weight (present in the heldout/mismatch blocks), so both the
    old-cert and deployed-cert main-grid campaigns ran the arm at the
    evaluator fallback price0=1.0 -- 100x below the w_reach crossover
    (~100) that the driver's own comments call a straw-man box. The
    recorded rows prove it: all 240 const_weight rows in main2_final and
    main3_adaptive carry price0=1.0, while heldout/mismatch rows carry
    50.0. This script re-runs the 240 main-grid const_weight missions
    per certificate at the correct price0=50.0.

WHAT THIS SCRIPT DOES:
    1. Builds the same 240 (map, budget, seed) tasks as the main stage,
       const_weight only, with price0=50.0.
    2. Runs them with the same run_reach_trial path (deterministic,
       same seeds/config as the original campaign).
    3. Writes the rows to <run>/reach/main_const50_<tag>/reach_trials_train.jsonl.
    4. Prints a summary so the new numbers are immediately visible.

Usage:
    python scripts/34_defect7_const_price50_rerun.py \\
        --run-id run_20260910T174433Z \\
        --maps-root reach_maps  --tag oldcert    # old certificate
    python scripts/34_defect7_const_price50_rerun.py \\
        --run-id run_20260910T174433Z \\
        --maps-root reach_maps_adaptive  --tag adaptive  # deployed cert

NOTE: the maps-root resolves through utils.paths.run_dir, which points
to the run tree; the adaptive root's map YAMLs there are absolute
symlinks into the pre-rename /data layout (broken since 2026-10-01).
Use --maps-root-override results/reach_maps[_adaptive] to read the
repo's dereferenced copies.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BUDGET_FRACTIONS = (0.30, 0.45, 0.60, 0.75)
SEED0 = 31000
N_SEEDS = 10



def _run_one(t):
    """Module-level worker: one const_weight@50 mission (picklable)."""
    from env.scenarios import load_scenario_from_yaml
    from evaluation.reach_evaluator import (ReachArm, run_reach_trial)
    from utils.config_loader import load_config
    sc = load_scenario_from_yaml(t['map_file'])
    arm = ReachArm('const_weight', use_constraint=False, price0=50.0,
                   turnback_rule='energy')
    cfg = load_config(None)
    res = run_reach_trial(sc, arm, B=t['B'], seed=t['seed'],
                           config=cfg, replan_every_s=2.5,
                           a_q=t['a_q'])
    rec = res.to_record()
    rec['split'] = 'train'
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--maps-root', default=None,
                    help='certificate root name inside the run tree '
                         '(e.g. reach_maps or reach_maps_adaptive)')
    ap.add_argument('--maps-root-override', default=None,
                    help='absolute or repo-relative path to the maps root '
                         '(use results/reach_maps[_adaptive] for the '
                         'dereferenced repo copies)')
    ap.add_argument('--tag', required=True,
                    help='suffix for the output dir (e.g. oldcert, adaptive)')
    ap.add_argument('--max-workers', type=int, default=64)
    args = ap.parse_args(argv)

    from utils.paths import run_dir as _run_dir
    from utils.config_loader import load_config
    from env.scenarios import load_scenario_from_yaml
    from evaluation.reach_evaluator import (ReachArm, run_reach_trial,
                                             emit_reach_trial)

    if args.maps_root_override:
        maps_root = args.maps_root_override
    else:
        maps_root = _run_dir(args.run_id, args.maps_root, create=False)

    # load the six training maps + their certificates
    certs = {c['map']: float(c['a_q'])
             for c in json.load(open(os.path.join(maps_root, 'train',
                                                  'certificates.json')))}
    import glob
    map_files = sorted(glob.glob(os.path.join(maps_root, 'train',
                                              'reach_train_*.yaml')))
    assert len(map_files) == 6, f'expected 6 training maps, got {len(map_files)}'

    # E_rt per map, from the original campaign rows (B_min / 0.30)
    _E_RT = {}
    _orig = os.path.join(_run_dir(args.run_id, 'reach', create=False),
                         'main3_adaptive_20260915',
                         'reach_trials_train.jsonl')
    from collections import defaultdict as _dd
    _Bs = _dd(set)
    for _l in open(_orig):
        _r = json.loads(_l)
        if _r['arm'] == 'reach':
            _Bs[_r['map_name']].add(_r['B'])
    for _m, _b in _Bs.items():
        _E_RT[_m] = sorted(_b)[0] / 0.30
    print(f'[setup] E_rt per map: { {k: round(v,1) for k,v in _E_RT.items()} }')

    # build the tasks (const_weight only, price 50, same grid as the main stage)
    tasks = []
    for mf in map_files:
        sc = load_scenario_from_yaml(mf)
        a_q = certs[sc.name]
        # E_rt from the original campaign rows (the yamls don't carry it;
        # the driver computes B = round(frac * E_rt, 1) and the original
        # rows record the exact B values, so E_rt = B_min / 0.30)
        E_rt = _E_RT.get(sc.name)
        if E_rt is None:
            raise SystemExit(f'map {sc.name} has no E_rt')
        for frac in BUDGET_FRACTIONS:
            B = round(frac * E_rt, 1)
            for s in range(SEED0, SEED0 + N_SEEDS):
                tasks.append({'map_file': mf, 'map_name': sc.name,
                              'B': B, 'seed': s, 'a_q': a_q})
    print(f'[setup] {len(tasks)} const_weight tasks at price 50, '
          f'maps root: {maps_root}')

    out_dir = os.path.join(_run_dir(args.run_id, 'reach', create=True),
                           f'main_const50_{args.tag}')
    os.makedirs(out_dir, exist_ok=True)
    sink = os.path.join(out_dir, 'reach_trials_train.jsonl')

    cfg = load_config(None)
    arm = ReachArm('const_weight', use_constraint=False, price0=50.0,
                   turnback_rule='energy')

    # run (serial would take too long; use concurrent futures)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import dataclasses


    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(_run_one, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            rec = fut.result()
            results.append(rec)
            done += 1
            if done % 40 == 0:
                print(f'  [{done}/{len(tasks)}] '
                      f'({time.perf_counter()-t0:.0f}s)')

    # sort by (map, B, seed) for stable merging
    results.sort(key=lambda r: (r['map_name'], r['B'], r['seed']))

    # write
    with open(sink, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(f'\n[done] {len(results)} rows -> {sink} '
          f'({time.perf_counter()-t0:.0f}s)')

    # summary: breaches, docked, reach
    n = len(results)
    b = sum(1 for r in results if r['energy_spent'] > r['B'] + 0.5)
    d = sum(1 for r in results if r['outcome'] == 'docked')
    reach = sum(r['reach_m'] for r in results) / n
    print(f'\n=== SUMMARY (const_weight @ price 50, {args.tag}) ===')
    print(f'  breaches: {b}/{n}')
    print(f'  docked:   {100*d/n:.1f}%')
    print(f'  reach:    {reach:.2f} m')
    print(f'  price0 in all rows: {set(r["price0"] for r in results)}')

    # per-budget breakdown
    from collections import defaultdict
    by = defaultdict(lambda: [0, 0])
    for r in results:
        key = r['B']
        by[key][0] += 1
        by[key][1] += int(r['energy_spent'] > r['B'] + 0.5)
    for B in sorted(by):
        print(f'  B={B}: {by[B][1]}/{by[B][0]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
