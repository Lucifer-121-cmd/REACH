"""Analyse the REACH campaign: sweep selection + headline tables.

Two jobs, in the order the paper needs them:

1. SWEEP SELECTION (rule fixed before the data, stated in
   script 21's docstring): the smallest const_weight price with ZERO
   reserve violations across all training maps and budgets; ties broken by
   highest mean reach. A price that violates even once is not a comparator
   -- the hard-constraint arm is compared against an arm that is SUPPOSED
   to respect the same budget.

2. HEADLINE: reach-vs-budget for all arms with violation and collision
   rates reported JOINTLY per cell (script 21's aggregate already emits
   the cells; this renders the tables and the discriminating deltas).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_aggregate(reach_dir: str, sink: str) -> List[Dict[str, Any]]:
    rows = [json.loads(l) for l in open(sink) if l.strip()]
    cells: Dict[tuple, Dict[str, Any]] = {}
    for r in rows:
        key = (r['map_name'], r['arm'], r['B'], r.get('split', 'train'),
               r.get('price0', 0.0))
        c = cells.setdefault(key, {'n': 0, 'docked': 0, 'stranded': 0,
                                   'collision': 0, 'timeout': 0,
                                   'violations': 0, 'reach_sum': 0.0,
                                   'reach_n': 0, 'margin_min': math.inf,
                                   'price': r.get('price0', 0.0),
                                   'a_q': r.get('a_q'),
                                   'a_q_source': r.get('a_q_source')})
        c['n'] += 1
        c[r['outcome']] += 1
        if r['min_margin_J'] < 0:
            c['violations'] += 1
        if r['reach_m'] is not None:
            c['reach_sum'] += r['reach_m']
            c['reach_n'] += 1
        c['margin_min'] = min(c['margin_min'], r['min_margin_J'])
    out = []
    for (m, a, B, split, _p), c in sorted(cells.items()):
        n = c['n']
        out.append({'map': m, 'arm': a, 'B': B, 'split': split, 'n': n,
                    'price0': c['price'], 'a_q': c['a_q'],
                    'a_q_source': c['a_q_source'],
                    'dock_rate': c['docked'] / n,
                    'stranded_rate': c['stranded'] / n,
                    'collision_rate': c['collision'] / n,
                    'timeout_rate': c['timeout'] / n,
                    'violation_rate': c['violations'] / n,
                    'reach_mean': (c['reach_sum'] / c['reach_n']
                                   if c['reach_n'] else None),
                    'margin_min_J': (None if c['margin_min'] == math.inf
                                     else c['margin_min'])})
    return out


def sweep_selection(sweep_cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The fixed rule: smallest price with zero violations; ties -> reach."""
    prices = sorted({c['price0'] for c in sweep_cells})
    per_price = {}
    for p in prices:
        cells = [c for c in sweep_cells if c['price0'] == p]
        viol = sum(c['violation_rate'] > 0 for c in cells)
        per_price[p] = {
            'n_cells': len(cells),
            'cells_with_violations': viol,
            'max_violation_rate': max((c['violation_rate'] for c in cells),
                                      default=0.0),
            'mean_reach': (sum(c['reach_mean'] or 0 for c in cells) / len(cells)
                           if cells else None),
            'zero_violations': viol == 0,
        }
    selected = None
    for p in prices:
        if per_price[p]['zero_violations']:
            selected = p
            break
    return {'per_price': {str(p): v for p, v in per_price.items()},
            'selected_price': selected,
            'rule': ('smallest price with ZERO reserve violations across all '
                     'training maps and budgets; ties broken by highest mean '
                     'reach. FIXED BEFORE THE DATA (script 21 docstring)')}


def headline(cells: List[Dict[str, Any]]) -> List[str]:
    """The headline table: per (map, arm, B) the joint rates + reach."""
    lines = []
    lines.append(f"{'map':<22}{'arm':<14}{'B':>6}{'n':>3}{'dock':>6}"
                 f"{'viol':>6}{'coll':>6}{'reach':>7}{'min_marg':>9}")
    for c in cells:
        lines.append(
            f"{c['map']:<22}{c['arm']:<14}{c['B']:>6.0f}{c['n']:>3}"
            f"{c['dock_rate']:>6.2f}{c['violation_rate']:>6.2f}"
            f"{c['collision_rate']:>6.2f}"
            f"{(c['reach_mean'] or 0):>7.2f}"
            f"{(c['margin_min_J'] if c['margin_min_J'] is not None else float('nan')):>9.1f}")
    return lines


def mismatch_table(sink: str) -> None:
    """Mismatch: hard-budget compliance per arm per delta — the pre-registered
    comparison (prediction: the constrained arm's advantage WIDENS)."""
    rows = [json.loads(l) for l in open(sink) if l.strip()]
    from collections import defaultdict
    agg = defaultdict(lambda: dict(n=0, dock=0, over=0, coll=0, reach=0.0, reach_n=0))
    for r in rows:
        a = agg[(r['arm'], r['power_mismatch'])]
        a['n'] += 1
        a['dock'] += r['outcome'] == 'docked'
        a['over'] += r['energy_spent'] > r['B'] + 0.5
        a['coll'] += r['outcome'] == 'collision'
        if r['reach_m'] is not None: a['reach'] += r['reach_m']; a['reach_n'] += 1
    print(f"{'delta':>6} {'arm':<14}{'n':>4}{'dock%':>7}{'BREACH%':>8}{'coll%':>7}{'reach':>7}")
    for (arm, d), a in sorted(agg.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        print(f"{d*100:>5.0f}% {arm:<14}{a['n']:>4}{100*a['dock']/a['n']:>7.1f}"
              f"{100*a['over']/a['n']:>8.1f}{100*a['coll']/a['n']:>7.1f}"
              f"{a['reach']/a['reach_n']:>7.2f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--sweep-sink', default=None,
                    help='JSONL of the sweep stage (default: reach/sweep_sink_*/reach_trials_train.jsonl)')
    ap.add_argument('--main-sink', default=None)
    ap.add_argument('--heldout-sink', default=None)
    args = ap.parse_args(argv)

    from utils.paths import run_dir as _run_dir
    from utils.safe_write import safe_write_json
    reach = _run_dir(args.run_id, 'reach', create=False)

    def _find(pattern: str) -> str:
        import glob
        hits = sorted(glob.glob(os.path.join(reach, pattern)))
        if not hits:
            return ''
        return hits[-1]

    report: Dict[str, Any] = {'run_id': args.run_id}

    sweep_sink = args.sweep_sink or _find('sweep_sink_*/reach_trials_train.jsonl')
    if sweep_sink:
        cells = _load_aggregate(reach, sweep_sink)
        sel = sweep_selection(cells)
        report['sweep'] = sel
        print("== const_weight sweep (selection rule fixed in advance) ==")
        for p, v in sel['per_price'].items():
            mark = '  <== SELECTED' if float(p) == (sel['selected_price'] or -1) else ''
            print(f"  price {p:>5}: {v['cells_with_violations']}/{v['n_cells']} cells violate, "
                  f"max rate {v['max_violation_rate']:.2f}, mean reach "
                  f"{v['mean_reach'] or 0:.2f}{mark}")
        if sel['selected_price'] is None:
            print("  NO price achieves zero violations -- report the "
                  "smallest viol rate and say so; do not relax the rule "
                  "after seeing the data")
    else:
        print("(no sweep sink found; skipping selection)")

    for name, sink_arg, default_pat in (
            ('main', args.main_sink, 'main_sink_*/reach_trials_train.jsonl'),
            ('heldout', args.heldout_sink,
             'reach_trials_heldout.jsonl')):
        sink = sink_arg or _find(default_pat)
        if not sink:
            print(f"(no {name} sink found; skipping)")
            continue
        cells = _load_aggregate(reach, sink)
        report[name] = cells
        print(f"\n== {name} ({len(cells)} cells) ==")
        for ln in headline(cells):
            print(ln)

    out = os.path.join(reach, 'reach_analysis.json')
    safe_write_json(out, report)
    print(f"\nwritten: {out}")

    # R5 mismatch stage, if present (pre-registered comparison)
    import glob as _glob
    mm = sorted(_glob.glob(os.path.join(reach,
                                        'mismatch_*/reach_trials_train.jsonl')))
    if mm:
        print("\n== R5 mismatch (pre-registered: the advantage WIDENS) ==")
        mismatch_table(mm[-1])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
