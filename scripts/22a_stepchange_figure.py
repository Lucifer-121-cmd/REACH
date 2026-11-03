"""Figure: step-change speed profiles — anticipation vs reaction.

One panel: outbound speed vs x for the three arms, with the terrain step
at x=15 shaded. The mechanism claim is visual: the constrained arm's speed
falls BEFORE the step (Ebar prices it ahead); the priced arms hold speed
into it and pay. Uses the per-step (x, v) profiles recorded by
scripts/22_reach_stepchange.py.

Rendered from stepchange_results.json; every figure carries a provenance
sidecar.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STEP_X = 15.0
ARMS = ('reach', 'dual', 'const_weight')
COLORS = {'reach': '#1a6faa', 'dual': '#d1495b', 'const_weight': '#edae49'}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', required=True,
                    help='stepchange_results.json from script 22')
    ap.add_argument('--out', required=True, help='output PDF path')
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from utils.provenance import write_sidecar

    data = json.load(open(args.results))
    trials = data['trials']

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.axvspan(STEP_X, 30.0, color='0.92', zorder=0)
    ax.text(0.5 * (STEP_X + 30.0), 1.32, 'terrain m=3',
            ha='center', va='bottom', fontsize=8, color='0.35')

    for arm in ARMS:
        rows = [t for t in trials if t['arm'] == arm]
        if not rows:
            continue
        # mean outbound speed binned by x (outbound = before each trial's
        # turn-back time; simplest robust cut: use the profile up to max_x)
        bins = np.arange(0.0, 26.0, 0.5)
        sums = np.zeros(len(bins) - 1)
        counts = np.zeros(len(bins) - 1)
        for t in rows:
            tb = t['turnback_time'] or 1e9
            for x, v, tt in zip(t['x_profile'], t['v_profile'],
                                np.arange(len(t['x_profile'])) * 0.1):
                if tt >= tb or x >= 25.0:
                    continue                      # outbound only
                b = int(x / 0.5)
                if 0 <= b < len(sums):
                    sums[b] += v
                    counts[b] += 1
        means = np.divide(sums, counts, out=np.zeros_like(sums),
                          where=counts > 0)
        centers = 0.5 * (bins[:-1] + bins[1:])
        label = {'reach': 'REACH (terminal reserve)',
                 'dual': 'dual ascent (reactive price)',
                 'const_weight': 'const weight (price %.0f)'
                                 % (rows[0].get('price0') or 1.0)}[arm]
        ax.plot(centers, means, color=COLORS[arm], lw=2.0, label=label)

    ax.axvline(STEP_X, color='0.4', lw=0.8, ls='--')
    ax.set_xlabel('x (m)   — dock at x = 1')
    ax.set_ylabel('outbound speed (m/s)')
    ax.set_xlim(1, 25)
    ax.set_ylim(0, 1.4)
    ax.legend(fontsize=8, loc='lower left', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out)
    write_sidecar(args.out, run_id=data.get('run_id', '?'),
                  inputs=[args.results])
    print(f"saved {args.out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
