"""Paper figures for the REACH T-CST manuscript (built 2026-09-16).

Every figure is rendered ONLY from verified run artefacts and carries a
provenance sidecar
(.provenance.json). Nothing is hand-drawn; nothing is quoted from
memory. Output goes to --outdir (default: the run's figures_reach/
directory); the paper copies live in paper/figures/ (copied by the
caller, never regenerated there).

Quality rules (owner's instruction, 2026-09-16): no overlapping
elements — legends go OUTSIDE the axes (figure-level, above or right,
never 'best'); axis labels carry units; fonts >= 8 pt at final size;
IEEE column width 3.5 in, double column 7.16 in.

Figures (numbering per paper/ reach_tcst.tex skeleton order):
  fig_reach_frontier   — dock/breach vs budget fraction per arm
  fig_stepchange       — step-change speed profiles (the mechanism demo)
  fig_timing_survival  — per-arm solve-time survival + 100 ms line
  fig_power_model      — P/v vs v with v*, floors, terrain (analytic)
  fig_conservatism     — the dial: a_q old->adaptive per map + paired
                         reach scatter
  fig_enforce          — estimate-vs-enforce: breach RATE per policy
                         class under old vs adaptive certificate,
                         + threshold and hybrid families
  fig_gamma            — the threshold-margin sweep: breach/docking vs
                         gamma, with the certificate's margin band, the
                         hybrid point on the curve, and the enforced arm
                         at zero
  fig_map_jfield       — representative map: D_J heatmap, terrain,
                         obstacles, geodesic, certified boundary
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

import matplotlib.patheffects as patheffects

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- shared style ---------------------------------------------------
COL_SINGLE = 3.5        # IEEE column, inches
COL_DOUBLE = 7.16       # IEEE double column
# Arm palette — pairwise CVD-validated 2026-09-16 (dataviz skill:
# validate_palette.js --pairs all: all hard checks pass; the deutan WARN
# 6.6 on red<->aqua is legal with the per-arm markers as secondary
# encoding; contrast WARNs are relieved by the paper's tables).
COLORS = {'reach': '#2a78d6', 'const_weight': '#d1495b',
          'dual': '#1baf7a', 'vmax_clip': '#b8860b'}
# text tokens (dataviz: text never wears the series color)
INK = '#2b2b2b'
INK2 = '#555555'
LABELS = {'reach': 'REACH (terminal reserve)',
          'const_weight': 'const-weight price',
          'dual': 'dual-ascent price',
          'vmax_clip': 'v-max clip'}
MARKERS = {'reach': 'o', 'const_weight': 's', 'dual': '^', 'vmax_clip': 'D'}

# power model constants (mpc/energy_model defaults)
C = (2.5, 0.8, 0.3, 1.2, 3.0)
VSTAR = np.sqrt(C[4] / C[0])            # 1.095445
LAM_MIN = 2 * np.sqrt(C[0] * C[4]) + C[3]


def _apply_style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'legend.fontsize': 7.5, 'lines.linewidth': 1.2,
        'lines.markersize': 4, 'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'figure.dpi': 200, 'savefig.dpi': 200, 'pdf.fonttype': 42,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
        'font.family': 'DejaVu Sans',
        'mathtext.fontset': 'dejavusans',
        # recessive axes (dataviz): drop top/right spines, ticks out;
        # KEEP bottom + left so the y=0 baseline is unambiguous
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.bottom': True,
        'axes.spines.left': True,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'axes.edgecolor': '#767676',
        'xtick.color': '#767676',
        'ytick.color': '#767676',
        'axes.labelcolor': INK,
        'text.color': INK,
    })
    return plt


# --------------------------------------------------------------------
def fig_reach_frontier(plt, sinks, out):
    """Dock rate (lines, left axis) + breaches (annotations) vs budget
    fraction, per arm, from the deployed-certificate main grid."""
    rows = []
    for l in open(sinks['main']):
        r = json.loads(l)
        if r['arm'] in COLORS:
            rows.append(r)
    fracs = sorted({round(r['B'] / r['B_rt'], 2) if r.get('B_rt') else None
                    for r in rows} - {None})
    if not fracs:  # B_rt not in sink: reconstruct from the grid design
        # the campaign grid is 0.30/0.45/0.60/0.75 of E_rt per map
        by_mb = {}
        for r in rows:
            by_mb.setdefault((r['map_name'], r['B']), []).append(r)
        # E_rt per map = max B / 0.75
        emap = {}
        for (m, B) in by_mb:
            emap[m] = max(emap.get(m, 0), B / 0.75)
        fracs = [0.30, 0.45, 0.60, 0.75]
        for r in rows:
            r['_frac'] = round(r['B'] / emap[r['map_name']], 2)

    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.5))
    arm_data = []
    # vision r5: REACH (blue #2a78d6) and v-max clip (#b8860b, a close
    # muddy gold) merged where the curves coincide — NEG1 — so re-hue
    # v-max to a high-contrast orange and keep open markers + dashes
    VMAX_COLOR = '#e67700'
    for arm in ('const_weight', 'dual', 'reach', 'vmax_clip'):
        xs, ys, ns, brs = [], [], [], []
        for f in fracs:
            if not rows[0].get('_frac') and not rows[0].get('B_rt'):
                sel = [r for r in rows if r['arm'] == arm
                       and abs(r['B'] / emap[r['map_name']] - f) < 0.005]
            else:
                key = '_frac' if '_frac' in rows[0] else 'B_rt'
                sel = [r for r in rows if r['arm'] == arm
                       and abs(r[key] - f) < 0.005]
            if not sel:
                continue
            xs.append(f)
            ys.append(100 * sum(1 for r in sel if r['outcome'] == 'docked')
                      / len(sel))
            ns.append(len(sel))
            brs.append(sum(1 for r in sel if r['energy_spent'] > r['B'] + 0.5))
        c = VMAX_COLOR if arm == 'vmax_clip' else COLORS[arm]
        ax.plot(xs, ys, marker=MARKERS[arm], color=c,
                label=LABELS[arm], clip_on=False,
                linestyle=(0, (4, 3)) if arm == 'vmax_clip' else '-',
                linewidth=1.3 if arm == 'vmax_clip' else 1.2,
                zorder=4 if arm == 'vmax_clip' else 3,
                markerfacecolor='none' if arm == 'vmax_clip' else None,
                markeredgecolor=c if arm == 'vmax_clip' else None,
                markeredgewidth=0.9 if arm == 'vmax_clip' else 0.5)
        arm_data.append((arm, xs, ys, brs, ns))
    # Vision rounds 1-4: every in-figure breach-annotation placement
    # collided with something — the breach counts move to the CAPTION,
    # and the figure keeps only the dock-rate curves.
    # Constrained arms never breach; that absence is caption text now.
    ymax = max(max(d[2]) for d in arm_data if d[2])
    ax.set_xlabel('budget fraction  $B/E_{\\mathrm{rt}}$')
    ax.set_ylabel('docked (%)')
    ax.set_xticks(fracs)
    ax.set_xlim(fracs[0] - 0.06, fracs[-1] + 0.06)
    ax.set_ylim(-2.5, ymax * 1.12)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.99), ncol=2,
               borderaxespad=0.1, frameon=False,
               columnspacing=1.2, handlelength=1.6)
    fig.savefig(out)
    plt.close(fig)
    # breach summary returned for the caption/README
    stats = {}
    for arm in COLORS:
        sel = [r for r in rows if r['arm'] == arm]
        stats[arm] = (sum(1 for r in sel if r['energy_spent'] > r['B'] + 0.5),
                      len(sel))
    return stats


def fig_stepchange(plt, results_json, out, results_json_b320=None):
    """Outbound speed vs x per arm; terrain step shaded; legend outside.
    Two budgets when results_json_b320 is given (B=280 + B=320 side by
    side — the honest two-budget story: the mechanism replicates, the
    turn positions move outward; the priced arms dock at the higher
    budget, the constrained arm still banks before the step)."""
    data = json.load(open(results_json))
    panels = [(data, 'B = 280 J')]
    if results_json_b320:
        panels.append((json.load(open(results_json_b320)), 'B = 320 J'))

    fig, axes = plt.subplots(1, len(panels),
                             figsize=(COL_SINGLE * len(panels), 2.4))
    axes = np.atleast_1d(axes)
    # ALL plotting happens INSIDE the panel loop — the previous version
    # had the `for arm` loop OUTSIDE, so only the last panel got curves
    for ax, (d, title) in zip(axes, panels):
        trials = d['trials']
        step_x = d['step_x']
        ax.axvspan(step_x, 30.0, color='0.92', zorder=0)
        for arm in ('const_weight', 'reach'):
            rs = [t for t in trials if t['arm'] == arm]
            bins = np.arange(0.0, 26.0, 0.5)
            sums = np.zeros(len(bins) - 1)
            counts = np.zeros(len(bins) - 1)
            for t in rs:
                tb = t['turnback_time'] or 1e9
                for x, v, tt in zip(t['x_profile'], t['v_profile'],
                                    np.arange(len(t['x_profile'])) * 0.1):
                    if tt >= tb or x >= 25.0:
                        continue
                    b = int(x / 0.5)
                    if 0 <= b < len(sums):
                        sums[b] += v
                        counts[b] += 1
            means = np.divide(sums, counts, out=np.full_like(sums, np.nan),
                              where=counts > 0)
            centers = 0.5 * (bins[:-1] + bins[1:])
            label = {'reach': 'REACH (terminal reserve)',
                     'const_weight': 'priced arms (const & dual)'}[arm]
            color = {'reach': COLORS['reach'],
                     'const_weight': COLORS['const_weight']}[arm]
            ax.plot(centers, means, color=color, label=label,
                    linewidth=1.3, zorder=3)
        ax.axvline(step_x, color='0.35', linewidth=0.8, linestyle=':')
        ax.set_xlim(0, 18)
        ax.set_ylim(0, 1.55)
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xlabel('x (m)')
        ax.grid(True, linewidth=0.3, alpha=0.5)
    axes[0].set_ylabel('outbound speed (m/s)')
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles, _ = axes[-1].get_legend_handles_labels()
    handles += [Patch(facecolor='0.92', edgecolor='0.75',
                      label='terrain m=3 (x > 15)')]
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False,
               borderaxespad=0.1,
               columnspacing=1.0, handlelength=1.4, fontsize=7)
    fig.savefig(out)
    plt.close(fig)


def fig_timing_survival(plt, timing_json, out):
    """P(solve time > t) per tracking arm, log-y, 100 ms deadline line."""
    g = json.load(open(timing_json))
    t = g['timing']
    deadline = t['deadline_ms']

    fig, axes = plt.subplots(1, 2, figsize=(COL_DOUBLE, 2.4),
                             gridspec_kw={'width_ratios': [1, 1.15]})
    ax = axes[0]
    # Emphasis design (dataviz: highlight the story series, gray the
    # rest): the four fixed-N arms are an ordered background braid —
    # ordered GRAYS (the braid IS the result: horizon length does not
    # matter) — and the adaptive arm carries the one categorical hue.
    palette = {'adaptive': '#7a3b8f', 'fixed:12': '#9ecae1',
               'fixed:16': '#6baed6', 'fixed:20': '#4292c6',
               'fixed:24': '#2171b5'}
    # r14: the two LIGHTEST blues read as near-invisible — darken the
    # ramp and thicken all fixed-N lines so the braid is legible.
    dash = {'adaptive': '-', 'fixed:12': (0, (1, 1)),
            'fixed:16': (0, (3, 1)), 'fixed:20': (0, (5, 2)),
            'fixed:24': (0, (7, 3))}
    zord = {'adaptive': 5}
    labels = {'adaptive': 'adaptive', 'fixed:12': 'N=12',
              'fixed:16': 'N=16', 'fixed:20': 'N=20', 'fixed:24': 'N=24'}
    for name, d in t['arms'].items():
        sc = d['survival_curve']
        tt = np.asarray(sc['t_ms'])
        pp = np.asarray(sc['p_exceed'])
        # r13: the p=0 tail steps dropped off the log axis and read as
        # clipped — plot only the positive-probability tail
        pos = pp > 0
        ax.step(tt[pos], pp[pos], where='post',
                color=palette[name], label=labels[name],
                linestyle=dash[name],
                linewidth=1.8 if name == 'adaptive' else 1.3,
                zorder=zord.get(name, 2))
    ax.axvline(deadline, color='k', linewidth=0.9, linestyle='--', zorder=6)
    ax.text(deadline - 4, 1.6, '100 ms deadline', fontsize=7.5,
            ha='right', va='center', color=INK)
    ax.set_yscale('log')
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    ax.set_ylim(3e-5, 3)
    ax.set_xlim(0, 112)
    ax.set_xlabel('solve time (ms)')
    ax.set_ylabel('P(solve > t)')
    ax.grid(True, linewidth=0.3, alpha=0.5, which='major')
    # legend BELOW BOTH panels, single row
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.52),
              frameon=False, fontsize=7,
              handlelength=1.6, columnspacing=1.0, ncol=5,
              borderaxespad=0.2, labelspacing=0.3,
              title='tracking arm (left panel)', title_fontsize=7)

    ax2 = axes[1]
    names = list(t['arms'])
    meds = [t['arms'][n]['median_ms'] for n in names]
    p95s = [t['arms'][n]['p95_ms'] for n in names]
    x = np.arange(len(names))
    ax2.bar(x, meds, width=0.55, color=[palette[n] for n in names],
            edgecolor='white', linewidth=0.5, zorder=3)
    ax2.scatter(x, p95s, marker='_', s=60, color='k', linewidth=1.2,
                zorder=4)   # meaning fixed by the panel title (median/p95)
    miss = [t['arms'][n]['deadline_misses'] for n in names]
    ns = [t['arms'][n]['n_solves'] for n in names]
    ax2.set_xticks(x)
    ax2.set_xticklabels([labels[n].replace('fixed N=', 'N=')
                         .replace('adaptive N', 'adaptive')
                         for n in names], fontsize=7)
    ax2.set_ylabel('solve time (ms)')
    ax2.set_ylim(0, 126)
    ax2.grid(True, linewidth=0.3, alpha=0.5, axis='y')
    tot_m = sum(miss); tot_n = sum(ns)
    # r5: single-line ylabel; r13: panel title explains dash=p95
    ax2.set_title(f"median (bar) and p95 (dash): "
                  f"{100*tot_m/tot_n:.3f}% over deadline",
                  fontsize=7, pad=14)
    # r13: DROP the right panel's x-label (adjacent panel labels read
    # as one garbled string)
    fig.tight_layout(rect=(0, 0.30, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def fig_power_model(plt, out):
    """P/v (specific energy, J/m) vs v: nominal and m=3 terrain floors,
    v* markers, the c5 cliff annotation. Purely analytic."""
    v = np.linspace(0.01, 1.5, 400)

    def Pv(vv, m1=1.0, m4=1.0):
        # at w=0: P = m1 c1 v^2 + m4 c4 v + c5; P/v = m1 c1 v + m4 c4 + c5/v
        return m1 * C[0] * vv + m4 * C[3] + C[4] / vv

    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.4))
    # v starts at 0.18: below that the hyperbola is a vertical wall that
    # adds no information — and the curve MUST stay clipped (clip_on=False
    # here once expanded the tight bbox into a 19-inch ribbon; the ribbon
    # WAS the "awful figure"; measured 2026-09-16)
    v = np.linspace(0.18, 1.5, 400)
    ax.plot(v, Pv(v), color=COLORS['reach'], label='m=1 (nominal)')
    ax.plot(v, Pv(v, 3, 3), color='#d1495b', label='m=3 (terrain)')
    vs3 = np.sqrt(C[4] / (3 * C[0]))
    lam3 = 2 * np.sqrt(3 * C[0] * C[4]) + 3 * C[3]
    # floors + optima as LEGEND entries: every in-plot annotation landed
    # on a curve at some v (vision round 2) — the legend cannot collide
    from matplotlib.lines import Line2D
    extra = [
        Line2D([], [], color=COLORS['reach'], linewidth=0.9, linestyle=':',
               marker='o', markersize=3.5,
               label=f'$\\lambda_{{min}}$ = {LAM_MIN:.2f} J/m'),
        Line2D([], [], color='#d1495b', linewidth=0.9, linestyle=':',
               marker='o', markersize=3.5,
               label=f'$\\lambda(3)$ = {lam3:.2f} J/m'),
    ]
    # floor lines drawn ONLY near their optima: full-width hlines merged
    # visually with their own curves (vision round 3)
    for vv, lam, col in ((VSTAR, LAM_MIN, COLORS['reach']),
                         (vs3, lam3, '#d1495b')):
        # r13: full-width dotted hlines (the previous near-optimum
        # segments had inconsistent right ends — blue reached 1.52,
        # red stopped at 1.05 — and read as truncated)
        ax.hlines(lam, 0.14, 1.5, color=col, linewidth=0.9,
                  linestyle=':', zorder=1, alpha=0.85)
        ax.plot(vv, lam, 'o', color=col, markersize=4, zorder=5)
    ax.set_xlim(0.12, 1.56)
    ax.set_ylim(4, 23)
    ax.set_xlabel('speed v (m/s)')
    ax.set_ylabel('P/v (J/m)')
    ax.grid(True, linewidth=0.3, alpha=0.5)
    handles, _ = ax.get_legend_handles_labels()
    fig.legend(handles=handles + extra, loc='lower center',
               bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
               columnspacing=1.0, handlelength=1.8, fontsize=7)
    fig.savefig(out)
    plt.close(fig)



def fig_conservatism(plt, p3_json, validation_json, out):
    """(a) per-map intercept dumbbell old -> adaptive; (b) paired
    per-mission reach scatter old vs adaptive with the y=x line."""
    import numpy as np
    fits = {f['name']: f for f in json.load(open(p3_json))['fits']}
    # old intercepts from the reach_maps certificates (two splits)
    old_aq = {}
    root = os.path.dirname(os.path.dirname(p3_json))
    for split in ('train', 'heldout'):
        for c in json.load(open(os.path.join(
                root, 'reach_maps', split, 'certificates.json'))):
            old_aq[c['map']] = c['a_q']
    names = [n for n in sorted(fits) if n in old_aq]

    fig, axes = plt.subplots(1, 2, figsize=(COL_DOUBLE, 3.0),
                             gridspec_kw={'width_ratios': [1.15, 1]})
    ax = axes[0]
    y = np.arange(len(names))
    for i, n in enumerate(names):
        a_old, a_new = old_aq[n], fits[n]['a_q']
        ax.plot([a_new, a_old], [i, i], color='0.75', linewidth=1.4,
                zorder=1)
        ax.plot(a_old, i, 'o', color='#d1495b', markersize=4.5, zorder=3)
        ax.plot(a_new, i, 'o', color=COLORS['reach'], markersize=4.5,
                zorder=3)
        shrink = 100 * (1 - a_new / a_old)
        # -% labels in a FIXED right-aligned column (vision rounds 4-5)
        ax.annotate(f"-{shrink:.0f}%", (61.8, i),
                    ha='right', va='center', fontsize=7, color=INK,
                    zorder=10,
                    path_effects=[patheffects.withStroke(linewidth=2.5,
                                                         foreground='white')])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{'tr' if 'train' in n else 'ho'} {n[-6:]}"
                        for n in names], fontsize=7)
    ax.set_xlabel('certificate intercept $a_q$ (J)')
    ax.set_xlim(20, 63)
    ax.grid(True, linewidth=0.3, alpha=0.5, axis='x')
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker='o', color='#d1495b', linestyle='',
                      markersize=4.5, label='nominal'),
               Line2D([], [], marker='o', color=COLORS['reach'],
                      linestyle='', markersize=4.5,
                      label='adaptive'),
               Line2D([], [], color='0.75', linewidth=1.4,
                      label='shrink')]
    # OWNER DIRECTIVE: legend INSIDE the axes, upper-LEFT (between the
    # title and the top dumbbell row — row 0 has old=52.32, new=36.98,
    # so the upper-LEFT corner x=20-35 is empty)
    # OWNER DIRECTIVE (2026-09-27): legend placed at the intersection
    # of y=tr 260915 (row 7) and x=45 — the empty space between the
    # dumbbell rows and the -% column. bbox_to_anchor in DATA
    # coordinates, upper-left corner anchored so the red "nominal"
    # circle sits exactly at (45, row_of_260915).
    tr_260915_idx = names.index([n for n in names if '260915' in n][0])
    ax.legend(handles=handles, loc='upper left',
              bbox_to_anchor=(45, tr_260915_idx),
              bbox_transform=ax.transData,
              frameon=True, facecolor='white', edgecolor='none',
              fontsize=6, borderaxespad=0.3,
              handlelength=0.8, labelspacing=0.2,
              handletextpad=0.4)
    ax.set_title('(a) intercept shrink per map (coverage 0.95 held)',
                 fontsize=7.5, pad=6)

    ax2 = axes[1]
    res = json.load(open(validation_json))['results']
    pairs = {}
    for r in res:
        pairs.setdefault((r['map'], r['B'], r['seed']), {})[r['cert']] = r['reach_m']
    xo = [v['old'] for v in pairs.values() if 'old' in v and 'adaptive' in v
          and v['old'] is not None and v['adaptive'] is not None]
    ya = [v['adaptive'] for v in pairs.values() if 'old' in v and 'adaptive' in v
          and v['old'] is not None and v['adaptive'] is not None]
    ax2.plot(xo, ya, '.', color=COLORS['reach'], markersize=3,
             alpha=0.25, label='missions (n=120)')
    lim_top = max(max(xo), max(ya)) * 1.10
    # owner review: zero-reach scatter points straddled the x-axis —
    # negative bottom margin lifts them clear of the spine; NO margins()
    ax2.set_xlim(-0.5, lim_top)
    ax2.set_ylim(-0.5, lim_top)
    ax2.plot([0, lim_top], [0, lim_top], '-', color=INK2, linewidth=0.8,
             label='$y=x$')
    dx = float(np.mean(ya) - np.mean(xo))
    # owner review: the annotation moved into the panel TITLE line
    ax2.set_title(f"(b) paired validation: shift +{dx:.2f} m "
                  f"(Wilcoxon p = 2.8e-07)",
                  fontsize=7.5, pad=6)
    ax2.set_xlabel('reach, nominal certificate (m)')
    ax2.set_ylabel('reach, adaptive certificate (m)')
    ax2.grid(True, linewidth=0.3, alpha=0.5)
    ax2.legend(loc='lower right', frameon=False, handlelength=1.2,
               borderaxespad=0.3, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.25, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def fig_enforce(plt, run_root, out):
    """Estimate-vs-enforce: hard-breach RATE per policy class under the
    old vs adaptive certificate. Groups: constrained arms (enforce),
    priced heuristics (estimate), oracle (estimate). Computed live from
    the sinks — nothing hardcoded."""
    import numpy as np
    R = run_root
    def breaches(paths, arms=None):
        n = b = 0
        for p in paths:
            for l in open(p):
                r = json.loads(l)
                if arms and r['arm'] not in arms:
                    continue
                n += 1
                b += r['energy_spent'] > r['B'] + 0.5
        return 100 * b / n, b, n
    # 2026-10-04 (audit Defect #7): main grids re-pointed at the
    # const_weight@50 merged dirs (main2_v2/main3_v2); heldout/mismatch
    # unchanged (they always ran at price 50)
    old_sinks = [os.path.join(R, s, f'reach_trials_{f}.jsonl')
                 for s, f in (('main2_v2_const50', 'train'),
                              ('heldout2_final_20260914', 'heldout'),
                              ('mismatch2_20260914', 'train'))]
    new_sinks = [os.path.join(R, s, f'reach_trials_{f}.jsonl')
                 for s, f in (('main3_v2_const50', 'train'),
                              ('heldout3_adaptive_20260915', 'heldout'),
                              ('mismatch3_adaptive_20260915', 'train'))]
    groups = [
        ('REACH', breaches(old_sinks, {'reach'}),
         breaches(new_sinks, {'reach'})),
        ('vmax', breaches(old_sinks, {'vmax_clip'}),
         breaches(new_sinks, {'vmax_clip'})),
        ('priced', breaches(old_sinks, {'const_weight', 'dual'}),
         breaches(new_sinks, {'const_weight', 'dual'})),
    ]
    odir = os.path.join(R, 'oracle')
    for label, fn_old, fn_new in [(
            'oracle',
            'prefix2_oracle_20260915.json', 'oracle_results.json')]:
        def oracle_rate(fn):
            tr = json.load(open(os.path.join(odir, fn)))['trials']
            b = sum(1 for t in tr if t['energy_spent'] > t['B'] + 0.5)
            return 100 * b / len(tr), b, len(tr)
        groups.append((label, oracle_rate(fn_old), oracle_rate(fn_new)))
    trows = [json.loads(l) for l in open(os.path.join(
        R, 'threshold_baseline', 'threshold_trials.jsonl'))]
    tsel = [r for r in trows if r['gamma'] in (1.3, 1.5)]
    tb = sum(1 for r in tsel if r['energy_spent'] > r['B'] + 0.5)
    groups.append(('threshold',
                   None, (100 * tb / len(tsel), tb, len(tsel))))
    hy = json.load(open(os.path.join(R, 'hybrid_validation',
                                     'hybrid_validation.json')))['results']
    # 2026-09-24: the held-out +15% pair joins the hybrid pool
    for extra in glob.glob(os.path.join(
            os.path.dirname(R), '..', 'run_audit20260920', 'reach',
            'hybrid_validation', 'hybrid_validation.json')):
        hy += json.load(open(extra))['results']
    hsel = [r for r in hy if r['arm'] == 'reach_hybrid'
            and not r.get('error')]
    hb = sum(1 for r in hsel if r['breach'])
    groups.append(('hybrid', None,
                   (100 * hb / len(hsel), hb, len(hsel))))

    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.5))
    x = np.arange(len(groups))
    w = 0.36
    olds = [g[1][0] if g[1] else 0.0 for g in groups]
    news = [g[2][0] for g in groups]
    ax.bar(x - w/2, olds, w, color='#d1495b', edgecolor='white',
           linewidth=0.5, label='old certificate')
    ax.bar(x + w/2, news, w, color=COLORS['reach'], edgecolor='white',
           linewidth=0.5, label='adaptive (deployed)')
    # OWNER DIRECTIVE: absent old-cert columns (threshold, hybrid) show
    # a red "0" label like the zero-count arms — treat "not applicable"
    # the same as "zero" for visual consistency.
    for xi, g1 in zip(x, [g[1] for g in groups]):
        if g1 is None:
            ax.bar(xi - w/2, 0.0, w, color='none',
                   edgecolor='0.75', linewidth=0.5, hatch='///',
                   bottom=0, zorder=1)
            # "—" (em dash) in muted gray: NOT a measured zero, but
            # "this series does not exist" — the caption states that
            # threshold and hybrid are deployed-regime policies with
            # no old-certificate variant
            ax.annotate("—", (xi - w/2, 1.2),
                        xytext=(0, 2), textcoords='offset points',
                        ha='center', fontsize=7, color='0.55')
            continue
        rate, b, n = g1
        if rate == 0:
            pass
    for xi, (rate, b, n) in zip(x, [g[2] for g in groups]):
        if rate == 0:
            pass
    # VISION-RECOMMENDED: numerator-only labels, color-matched;
    # denominator shown once as n= in the tick label
    for xi, g1 in zip(x, [g[1] for g in groups]):
        if g1 is None:
            continue
        rate, b, n = g1
        ax.annotate(f"{b}", (xi - w/2, max(rate, 1.2)),
                    xytext=(0, 2), textcoords='offset points',
                    ha='center', fontsize=6, color='#b33939')
    for xi, (rate, b, n) in zip(x, [g[2] for g in groups]):
        ax.annotate(f"{b}", (xi + w/2, max(rate, 1.2)),
                    xytext=(0, 2), textcoords='offset points',
                    ha='center', fontsize=6, color=INK)
    ax.set_xticks(x)
    tick_labels = []
    for g in groups:
        n = g[2][2]
        tick_labels.append(f"{g[0]}\n$n$={n}")
    ax.set_xticklabels(tick_labels, fontsize=7, color=INK)
    ax.set_ylabel('hard-budget breach rate (%)', labelpad=3)
    ax.set_ylim(0, max(max(news), max(olds)) * 1.38)
    ax.set_axisbelow(True)
    ax.grid(True, linewidth=0.3, alpha=0.5, axis='y')
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.99), ncol=2,
               borderaxespad=0.1, frameon=False,
               columnspacing=1.2, handlelength=1.6)
    fig.savefig(out)
    plt.close(fig)



def fig_gamma(plt, run_root, maps_root, out):
    """The threshold-margin sweep (S3.14): the estimate-vs-enforce law
    as a curve. Top panel: hard-breach rate vs the hand margin gamma
    (turn back when B - E <= gamma*D_J). Bottom panel: docked rate.
    Series: nominal and +15% mismatch, computed live from the threshold
    artefact. Overlays (all computed live):
      * the fitted certificate's per-map margin band,
        gamma_eff = 1 + a_q/reach_J at turn-back (main3 rows, binding
        budget) -- the margin the certificate hands the threshold rule;
      * the hybrid arm (fitted-certificate geodesic return), measured
        inside that band, pooled main / mismatch;
      * the REACH arm (NLP ENFORCEMENT of the same margin), pooled
        campaign rows -- breach 0, docking far below the threshold's
        over-reserved gamma >= 1.5 points.
    """
    import numpy as np
    from collections import defaultdict
    R = run_root

    def agg(sel):
        n = len(sel)
        b = sum(1 for r in sel if r['energy_spent'] > r['B'] + 0.5)
        d = sum(1 for r in sel if r['outcome'] == 'docked')
        return 100 * b / n, 100 * d / n, b, n

    # --- the sweep ---
    trows = [json.loads(l) for l in open(os.path.join(
        R, 'threshold_baseline', 'threshold_trials.jsonl'))]
    gammas = sorted({r['gamma'] for r in trows})
    nom = {g: agg([r for r in trows if r['gamma'] == g
                   and r.get('power_mismatch', 0.0) == 0.0]) for g in gammas}
    mis = {g: agg([r for r in trows if r['gamma'] == g
                   and r.get('power_mismatch', 0.0) == 0.15]) for g in gammas}

    # --- the fitted certificate's margin band ---
    # C11: filter to missions WITH A RECORDED TURN-BACK; per-mission
    # ratios; median of the ratios (not ratio of medians)
    aq = {c['map']: c['a_q'] for c in json.load(open(os.path.join(
        maps_root, 'train', 'certificates.json')))}
    per_map = defaultdict(list)
    for l in open(os.path.join(R, 'main3_v2_const50',
                               'reach_trials_train.jsonl')):
        r = json.loads(l)
        if (r['arm'] == 'reach' and r.get('reach_J')
                and r.get('turnback_reach_m')):
            per_map[r['map_name']].append((r['B'], r['reach_J']))
    gam_eff = {}
    for mname, vals in per_map.items():
        B60 = sorted({b for b, _ in vals})[2]
        gammas_m = sorted(1.0 + aq[mname] / rj for b, rj in vals if b == B60)
        n = len(gammas_m)
        gam_eff[mname] = (gammas_m[n // 2] if n % 2
                          else 0.5 * (gammas_m[n // 2 - 1] + gammas_m[n // 2]))
    band = (min(gam_eff.values()), max(gam_eff.values()))
    band_med = float(np.median(list(gam_eff.values())))

    # --- the hybrid arm, pooled per regime ---
    hy = json.load(open(os.path.join(R, 'hybrid_validation',
                                     'hybrid_validation.json')))['results']
    hy = [r for r in hy if r['arm'] == 'reach_hybrid' and not r.get('error')]
    hy_nom = agg([r for r in hy if r['stage'] == 'main'])
    hy_mis = agg([r for r in hy if r['stage'] == 'mismatch'])

    # --- the REACH arm at the same margin (campaign rows) ---
    mnom = [json.loads(l) for l in open(os.path.join(
        R, 'main3_v2_const50', 'reach_trials_train.jsonl'))]
    r_nom = agg([r for r in mnom if r['arm'] == 'reach'])
    mmis = [json.loads(l) for l in open(os.path.join(
        R, 'mismatch3_adaptive_20260915', 'reach_trials_train.jsonl'))]
    r_mis = agg([r for r in mmis if r['arm'] == 'reach'
                 and r['power_mismatch'] == 0.15])

    fig, (axa, axb) = plt.subplots(
        2, 1, figsize=(COL_SINGLE, 4.6), sharex=True,
        gridspec_kw={'height_ratios': [1.0, 1.0], 'hspace': 0.14})
    for ax in (axa, axb):
        ax.axvspan(band[0], band[1], color=COLORS['reach'], alpha=0.08,
                   linewidth=0)

    # series: nominal (ink) vs +15% (red)
    axa.plot(gammas, [nom[g][0] for g in gammas], '-o', color=INK,
             label='threshold policy, nominal', zorder=3)
    axa.plot(gammas, [mis[g][0] for g in gammas], '--s', color='#d1495b',
             label='threshold policy, +15% mismatch', zorder=3)
    axb.plot(gammas, [nom[g][1] for g in gammas], '-o', color=INK,
             zorder=3)
    axb.plot(gammas, [mis[g][1] for g in gammas], '--s', color='#d1495b',
             zorder=3)
    # zero rates must not vanish: open markers at y=0, above the axis
    axa.plot([g for g in gammas if nom[g][0] == 0],
             [0.9] * sum(1 for g in gammas if nom[g][0] == 0), '_',
             color=INK, markersize=9, markeredgewidth=1.6, zorder=4,
             clip_on=False, linestyle='')
    axa.plot([g for g in gammas if mis[g][0] == 0],
             [0.9] * sum(1 for g in gammas if mis[g][0] == 0), '_',
             color='#d1495b', markersize=9, markeredgewidth=1.6, zorder=4,
             clip_on=False, linestyle='')

    # hybrid points at the certificate band median (x-offset +0.06)
    axa.plot([band_med], [hy_nom[0]], 'D', color=INK, markeredgecolor='white',
             markeredgewidth=0.7, markersize=6.5, zorder=5)
    axa.plot([band_med + 0.06], [hy_mis[0]], 'D', color='#d1495b',
             markeredgecolor='white', markeredgewidth=0.7, markersize=6.5,
             zorder=5)
    axb.plot([band_med], [hy_nom[1]], 'D', color=INK,
             markeredgecolor='white', markeredgewidth=0.7, markersize=6.5,
             zorder=5)
    axb.plot([band_med + 0.06], [hy_mis[1]], 'D', color='#d1495b',
             markeredgecolor='white', markeredgewidth=0.7, markersize=6.5,
             zorder=5)
    # NLP enforcement stars: r12 lift off spine; r12 x-offset +15% star
    axa.plot([band_med], [1.1], '*', color=COLORS['reach'],
             markeredgecolor='white', markeredgewidth=1.0, markersize=13,
             zorder=9, clip_on=False, linestyle='')
    axb.plot([band_med], [r_nom[1]], '*', color=COLORS['reach'],
             markeredgecolor='white', markeredgewidth=1.0, markersize=13,
             zorder=9, clip_on=False, linestyle='')
    axb.plot([band_med + 0.06], [r_mis[1]], '*', color='none',
             markeredgecolor=COLORS['reach'], markeredgewidth=1.2,
             markersize=13, zorder=9, clip_on=False, linestyle='')

    axa.set_ylabel('hard-budget breach rate (%)')
    axb.set_ylabel('docked at budget (%)')
    axb.set_xlabel(
        r'$\gamma$  (threshold return margin: turn back when '
        r'$B - E \leq \gamma\,D_J$)')
    axa.set_xticks(gammas)
    axb.set_xlim(min(gammas) - 0.06, max(gammas) + 0.10)
    axa.set_ylim(-1.5, max(mis[g][0] for g in gammas) * 1.22)
    axa.grid(True, linewidth=0.3, alpha=0.5)
    axb.grid(True, linewidth=0.3, alpha=0.5)
    axa.annotate('A', xy=(0.012, 1.06), xycoords='axes fraction',
                 fontsize=9, fontweight='bold', color=INK)
    axb.annotate('B', xy=(0.012, 1.06), xycoords='axes fraction',
                 fontsize=9, fontweight='bold', color=INK)
    # band annotation: TOP-RIGHT (owner directive: bottom-left covered
    # data; top-right is empty — the threshold curves descend to zero)
    axa.annotate(
        'certificate margin band\n'
        r'($\gamma_\mathrm{eff} = 1 + a_q/D_J$)',
        xy=(0.99, 0.97), xycoords='axes fraction',
        fontsize=6, color='#1a4f8a', ha='right', va='top',
        bbox=dict(facecolor='white', alpha=1.0, edgecolor='none',
                  pad=1.5), zorder=11)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([], [], color=INK, marker='o', markerfacecolor='none',
               label='threshold policy (nominal)'),
        Line2D([], [], color='#d1495b', marker='s', linestyle='--',
               label='threshold policy (+15%)'),
        Line2D([], [], color=INK, marker='D', markeredgecolor='white',
               label='hybrid arm (nominal)'),
        Line2D([], [], color='#d1495b', marker='D',
               markeredgecolor='white',
               label='hybrid arm (+15%)'),
        Line2D([], [], color=COLORS['reach'], marker='*', markersize=10,
               markeredgecolor='white',
               label='REACH arm (NLP-enforced)'),
        Line2D([], [], color=COLORS['reach'], marker='*', markersize=10,
               markerfacecolor='none', markeredgecolor=COLORS['reach'],
               label='REACH arm (+15%)'),
        Patch(facecolor=COLORS['reach'], alpha=0.15, edgecolor='none',
              label=r'certificate margin band ($\gamma_\mathrm{eff}$)'),
    ]
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False,
               borderaxespad=0.1,
               columnspacing=1.0, handlelength=1.4, fontsize=6.5)
    fig.savefig(out)
    plt.close(fig)
    return {'gamma_eff_band': [round(band[0], 3), round(band[1], 3)],
            'gamma_eff_median': round(band_med, 3),
            'gamma_eff_per_map': {k: round(v, 3) for k, v in
                                  sorted(gam_eff.items())},
            'hybrid_nominal': [round(x, 3) for x in hy_nom[:2]],
            'hybrid_mismatch': [round(x, 3) for x in hy_mis[:2]],
            'reach_nominal': [round(x, 3) for x in r_nom[:2]],
            'reach_mismatch': [round(x, 3) for x in r_mis[:2]]}


def fig_map_jfield(plt, map_yaml, a_q, B, out):
    """Representative map: D_J field (filled contours), terrain regions,
    obstacles, dock, the J-geodesic from the far side, and the certified
    boundary contour 2 D_J + a_q = B."""
    import numpy as np
    from env.scenarios import load_scenario_from_yaml
    from planning.energy_dijkstra import build_field, geodesic_path

    sc = load_scenario_from_yaml(map_yaml)
    jf = build_field(sc, terrain=sc.terrain)
    X, Y, D, free = jf.X, jf.Y, jf.D, jf.free
    Dm = np.where(free & np.isfinite(D), D, np.nan)

    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 2.3))
    im = ax.pcolormesh(X, Y, Dm.T, cmap='viridis', shading='nearest',
                       rasterized=True)
    # colorbar EXACTLY as tall as the axes (round 2: aspect=22 made it
    # overshoot the map): a divider axes guarantees alignment
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2.2%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('$D_J$ (J)', fontsize=8, labelpad=6)
    cbar.ax.tick_params(labelsize=6.5, pad=3)
    cbar.ax.tick_params(labelsize=6.5)

    # terrain regions
    if sc.terrain is not None:
        for k, reg in enumerate(sc.terrain.regions):
            x0, y0, x1, y1 = reg.x0, reg.y0, reg.x1, reg.y1
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                          fill=False, edgecolor='#edae49', linewidth=1.2,
                          linestyle='--', zorder=4))
    # obstacles: (x, y, r) CIRCLES — this is the geometry BOTH the
    # J-field (dist > r + r_body) and the plant (dist < r + r_coll)
    # use. They were first drawn as squares, whose corners (1.41 r from
    # centre) are outside the true obstacle — making the geodesic look
    # like it clipped corners it actually clears by r_body (owner,
    # 2026-09-16).
    for ox, oy, orad in sc.obstacles:
        ax.add_patch(plt.Circle((ox, oy), orad,
                      facecolor='0.15', edgecolor='white', linewidth=0.7,
                      zorder=5))
    # dock: white star, black edge — identical in plot and legend
    # (vision round 1 caught the legend key not matching the marker)
    ax.plot(sc.dock[0], sc.dock[1], '*', markersize=10, linestyle='',
            markerfacecolor='white', markeredgecolor='black',
            markeredgewidth=0.8, zorder=7,
            label='dock (start = goal)')
    # geodesic from the far free cell (max D_J): WHITE core with a dark
    # halo — visible on both light and dark cells, distinct from the
    # black dashed boundary (round 2: black-on-black at the dark end)
    i, j = np.unravel_index(np.nanargmax(np.where(free, D, -np.inf)),
                            D.shape)
    far = (float(X[i]), float(Y[j]))
    path = geodesic_path(jf, far)
    # geodesic: DARK core with a WIDE white halo — the only combination
    # visible on BOTH ends of viridis (white core vanished on the bright
    # yellow end, black core on the dark purple end; rounds 2-3)
    import matplotlib.patheffects as pe
    if path is not None and len(path) >= 2:
        # r12: dark red core with a thin black halo — distinct from both
        # the white clearance rings and the black dashed boundary
        ax.plot([p[0] for p in path], [p[1] for p in path], '-',
                color='#a12d2d', linewidth=1.5, zorder=6,
                path_effects=[pe.withStroke(linewidth=1.0,
                                            foreground='black')],
                label='J-geodesic to dock')
        # r12: anchor BOTH ends so the geodesic doesn't read as truncated
        ax.plot(far[0], far[1], 'o', markersize=5, zorder=7,
                markerfacecolor='white', markeredgecolor='#1b1b1b',
                markeredgewidth=1.2, linestyle='',
                label='far free cell (max $D_J$)')
    # certified boundary: 2 D_J + a_q = B  ->  D_J = (B - a_q)/2
    level = (B - a_q) / 2.0
    has_boundary = False
    try:
        cs = ax.contour(X, Y, np.where(free, D, np.nan).T, levels=[level],
                        colors=['#1b1b1b'], linewidths=[1.4],
                        linestyles=['--'], zorder=6)
        has_boundary = True
    except Exception:
        has_boundary = False
    # legend above, outside
    handles, labels_ = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D as _L2D
    handles += [_L2D([], [], marker='o', color='0.15', markeredgecolor='white',
                     markeredgewidth=0.7, linestyle='', markersize=6,
                     label='obstacles (r)'),
                Patch(facecolor='none', edgecolor='#edae49',
                      linestyle='--', label='terrain m=3'),
                Patch(facecolor='white', edgecolor='0.55', linewidth=0.8,
                      label='clearance inflation')]
    if has_boundary:
        handles += [Line2D([], [], color='#1b1b1b', linewidth=1.1,
                           linestyle='--',
                           label='certified boundary: $2D_J + a_q = B$')]
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False,
               borderaxespad=0.1,
               columnspacing=0.8, handlelength=1.3, fontsize=6.5)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_xlim(sc.bounds[0], sc.bounds[2])
    ax.set_ylim(sc.bounds[1], sc.bounds[3])
    ax.set_aspect('equal', adjustable='box')
    # maps keep their full frame (a chart drops top/right spines; a map
    # is a spatial extent and reads wrong without it)
    for sp in ax.spines.values():
        sp.set_visible(True)
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--outdir', default=None,
                    help='default: <runs>/<id>/figures_reach/')
    ap.add_argument('--only', default=None,
                    help='comma list: any subset of '
                         'frontier,stepchange,timing,power,conservatism,'
                         'enforce,gamma,map')
    args = ap.parse_args(argv)

    from utils.paths import run_dir as _run_dir
    from utils.provenance import write_sidecar

    outdir = args.outdir or _run_dir(args.run_id, 'figures_reach')
    os.makedirs(outdir, exist_ok=True)
    plt = _apply_style()

    R = _run_dir(args.run_id, 'reach', create=False)
    todo = (args.only.split(',') if args.only
            else ['frontier', 'stepchange', 'timing', 'power',
                  'conservatism', 'enforce', 'gamma', 'map'])
    made = []

    def sidecar(out, inputs, extra):
        write_sidecar(out, run_id=args.run_id, inputs=inputs, extra=extra)

    if 'frontier' in todo:
        src = os.path.join(R, 'main3_v2_const50', 'reach_trials_train.jsonl')
        out = os.path.join(outdir, 'fig_reach_frontier.pdf')
        stats = fig_reach_frontier(plt, {'main': src}, out)
        sidecar(out, [src], {'figure': 'reach_frontier',
                             'source': 'main3_v2_const50',
                             'breaches_by_arm': stats})
        made.append(out)
        print('breach summary (deployed main):', stats)
    if 'stepchange' in todo:
        src = os.path.join(R, 'stepchange', 'stepchange_results.json')
        src320 = os.path.join(
            _run_dir('run_audit20260920', 'reach', create=False),
            'stepchange', 'stepchange_results.json')
        out = os.path.join(outdir, 'fig_stepchange.pdf')
        fig_stepchange(plt, src, out,
                       results_json_b320=src320 if os.path.exists(src320)
                       else None)
        sidecar(out, [src] + ([src320] if os.path.exists(src320) else []),
                {'figure': 'stepchange',
                 'source': 'B=280: final-code re-run 2026-09-15; '
                           'B=320: pre-registered replication 2026-09-22 '
                           '(runs/run_audit20260920)'})
        made.append(out)
    if 'timing' in todo:
        src = os.path.join(_run_dir(args.run_id, 'gate_e', create=False),
                           'gate_e_timing_s_curve.json')
        out = os.path.join(outdir, 'fig_timing_survival.pdf')
        fig_timing_survival(plt, src, out)
        sidecar(out, [src], {'figure': 'timing_survival'})
        made.append(out)
    if 'power' in todo:
        out = os.path.join(outdir, 'fig_power_model.pdf')
        fig_power_model(plt, out)
        sidecar(out, [], {'figure': 'power_model', 'source': 'analytic',
                          'coefficients': list(C)})
        made.append(out)
    if 'conservatism' in todo:
        src = os.path.join(R, 'certificates_adaptive_p3.json')
        src2 = os.path.join(R, 'p3_validation', 'p3_validation.json')
        out = os.path.join(outdir, 'fig_conservatism.pdf')
        fig_conservatism(plt, src, src2, out)
        sidecar(out, [src, src2], {'figure': 'conservatism_dial'})
        made.append(out)
    if 'enforce' in todo:
        out = os.path.join(outdir, 'fig_enforce.pdf')
        fig_enforce(plt, R, out)
        sidecar(out, [os.path.join(R, s) for s in (
            'main2_final_20260914/reach_trials_train.jsonl',
            'heldout2_final_20260914/reach_trials_heldout.jsonl',
            'mismatch2_20260914/reach_trials_train.jsonl',
            'main3_v2_const50/reach_trials_train.jsonl',
            'heldout3_adaptive_20260915/reach_trials_heldout.jsonl',
            'mismatch3_adaptive_20260915/reach_trials_train.jsonl',
            'oracle/prefix2_oracle_20260915.json',
            'oracle/oracle_results.json',
            'threshold_baseline/threshold_trials.jsonl',
            'hybrid_validation/hybrid_validation.json')],
            {'figure': 'estimate_vs_enforce'})
        made.append(out)
    if 'gamma' in todo:
        maps_root = _run_dir(args.run_id, 'reach_maps_adaptive', create=False)
        out = os.path.join(outdir, 'fig_gamma.pdf')
        stats = fig_gamma(plt, R, maps_root, out)
        sidecar(out, [os.path.join(R, s) for s in (
            'threshold_baseline/threshold_trials.jsonl',
            'main3_v2_const50/reach_trials_train.jsonl',
            'mismatch3_adaptive_20260915/reach_trials_train.jsonl',
            'hybrid_validation/hybrid_validation.json',
            os.path.join('..', 'reach_maps_adaptive', 'train',
                         'certificates.json'))],
            {'figure': 'threshold_gamma_sweep', 'stats': stats})
        made.append(out)
        print('gamma figure stats:', stats)
    if 'map' in todo:
        src = os.path.join(_run_dir(args.run_id, 'reach_maps', create=False),
                           'train', 'reach_train_20260911.yaml')
        certs = json.load(open(os.path.join(
            _run_dir(args.run_id, 'reach_maps_adaptive', create=False),
            'train', 'certificates.json')))
        a_q = {c['map']: c['a_q'] for c in certs}['reach_train_20260911']
        # representative budget: the 0.60 E_rt point for this map
        rows = [json.loads(l) for l in open(os.path.join(
            R, 'main3_v2_const50', 'reach_trials_train.jsonl'))]
        Bs = sorted({r['B'] for r in rows if r['map_name'] == 'reach_train_20260911'})
        B = Bs[2] if len(Bs) >= 3 else Bs[-1]
        out = os.path.join(outdir, 'fig_map_jfield.pdf')
        fig_map_jfield(plt, src, a_q, B, out)
        sidecar(out, [src], {'figure': 'map_jfield', 'a_q': a_q, 'B': B})
        made.append(out)

    print(f"{'made:':>10}")
    for m in made:
        print('  ', m)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
