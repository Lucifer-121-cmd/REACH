"""fig_oracle_traj — the oracle gap made visible: one paired mission's
trajectories (2026-10-03, the T-CST revision figure for Sec. VII-C).

WHAT THE FIGURE SHOWS
---------------------
Sec. VII-C quantifies the estimate-vs-enforce law's cost: the
constrained arm realises 64.5% of the plan-then-track oracle's reach,
and the decomposition attributes the gap to EARLY TURN-BACKS (the arm
turns at 69% of its certified ceiling holding +34.7 J of certified
margin, the oracle rides the boundary to 92% and completes). This
figure shows ONE representative pair of missions as trajectories on
the map: both arms' outbound and return legs, their turn-back points,
and the certified boundary 2 D_J + a_q = B the oracle rides.

SELECTION RULE (declared, mechanical, computed live from the artefacts)
------------------------------------------------------------------------
Among training-map pairs of the deployed-certificate oracle campaign
(``reach/oracle/oracle_results.json``, whose rows carry the paired
``reach_arm_*`` fields) where the oracle docked and the constrained
arm did not collide, select the pair whose constrained-arm
reach-to-certified-ceiling fraction is CLOSEST TO THE CAMPAIGN-POOLED
arm fraction (0.596, the pooled main-grid figure). The picked pair is the
one whose closed-loop behaviour best represents the pooled
decomposition the section quotes.

TRAJECTORY PROVENANCE (the bit-identity contract)
--------------------------------------------------
The oracle and campaign artefacts record summary statistics only, so
this script RE-RUNS both missions of the selected pair with a
recording environment (a PioneerEnv subclass that logs the state
after every plant step; no dynamics, seeding, or logic is touched).
The re-runs must then reproduce the RECORDED rows exactly -- same
outcome, energy, reach, turn-back fields, to the last digit -- or the
script aborts with a loud error: the plotted trajectories are then
provably the recorded missions' trajectories, not lookalikes. This is
the same reproduction discipline the main-grid validation used to prove
the stack bit-identical.

Usage:
  python scripts/33_oracle_trajectory_figure.py --run-id <id> \
      [--maps-root reach_maps_adaptive] [--outdir <dir>]

Outputs: fig_oracle_traj.{pdf,png} + provenance sidecar in the run's
figures_reach/ (or --outdir).
"""

import argparse
import importlib.util
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COL_SINGLE = 3.5
INK = '#2b2b2b'
C_ARM = '#2a78d6'      # house palette: reach blue (script 27, CVD-checked)
C_ORA = '#d1495b'      # house palette: red (pairwise CVD-validated pair)
POOLED_ARM_FRACTION = 0.596   # pooled arm/ceiling fraction, main grid


def _load_oracle_module():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '24_reach_oracle.py')
    spec = importlib.util.spec_from_file_location('oracle_mod', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Recorder:
    """Patch env.pioneer_env.PioneerEnv with a subclass that logs the
    state after reset and after every plant step. Nothing else changes:
    the same RNG draws, the same dynamics, the same call order."""

    def __init__(self, dt):
        import env.pioneer_env as pe
        self._pe = pe
        self._orig = pe.PioneerEnv
        self.dt = dt
        self.traj = []      # list of (t, x, y, E)

        outer = self

        class Rec(self._orig):
            def reset(self, *a, **kw):
                out = super().reset(*a, **kw)
                st = self.get_state_vector()
                outer.traj.append((0.0, float(st[0]), float(st[1]),
                                   float(st[5])))
                return out

            def step(self, v, w):
                super().step(v, w)
                st = self.get_state_vector()
                outer.traj.append((len(outer.traj) * outer.dt,
                                   float(st[0]), float(st[1]),
                                   float(st[5])))

        self._cls = Rec
        pe.PioneerEnv = Rec

    def restore(self):
        self._pe.PioneerEnv = self._orig

    def split(self, t_turn):
        """(outbound, return) samples split at the recorded turn-back
        time; the turn-back sample itself joins BOTH legs (the robot is
        there when the phase flips)."""
        out, ret = [], []
        for s in self.traj:
            (out if s[0] <= t_turn else ret).append(s)
        if out and ret:
            out.append(ret[0])
        return out, ret


def _check(name, got, want, problems):
    if got != want:
        problems.append(f"{name}: re-run {got!r} != recorded {want!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', default='run_20260910T174433Z')
    ap.add_argument('--maps-root', default=None,
                    help='default: the run-id reach_maps_adaptive/ '
                         '(the deployed certificates)')
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--from-cache', action='store_true',
                    help='plot from the verified trajectory cache '
                         '(fig_oracle_traj.traj.json); the cached pair '
                         'and rows are re-checked against the fresh '
                         'artefacts before use')
    args = ap.parse_args(argv)

    from utils.paths import run_dir as _run_dir
    from utils.config_loader import load_config
    from utils.provenance import write_sidecar
    from env.scenarios import load_scenario_from_yaml
    from planning.energy_dijkstra import build_field
    from evaluation.reach_evaluator import run_reach_trial, ARMS

    reach_root = _run_dir(args.run_id, 'reach', create=False)
    maps_root = (args.maps_root or
                 _run_dir(args.run_id, 'reach_maps_adaptive', create=False))
    outdir = args.outdir or _run_dir(args.run_id, 'figures_reach')
    os.makedirs(outdir, exist_ok=True)

    oracle_json = os.path.join(reach_root, 'oracle', 'oracle_results.json')
    main_jsonl = os.path.join(reach_root,
                              'main3_adaptive_20260915',
                              'reach_trials_train.jsonl')

    # ---- 1. the declared selection, computed live --------------------
    trials = json.load(open(oracle_json))['trials']
    pool = [t for t in trials if t['split'] == 'train'
            and t['outcome'] == 'docked'
            and t['reach_arm_outcome'] in ('stranded', 'docked')]
    sel = min(pool, key=lambda t: abs(
        t['reach_arm_reach_m'] / t['certified_ceiling_m']
        - POOLED_ARM_FRACTION))
    print(f"[select] pool={len(pool)} pairs -> map={sel['map']} "
          f"seed={sel['seed']} B={sel['B']} "
          f"arm/ceiling={sel['reach_arm_reach_m']/sel['certified_ceiling_m']:.3f} "
          f"(pooled target {POOLED_ARM_FRACTION})")

    # the paired campaign row + the deployed certificate
    row = None
    for line in open(main_jsonl):
        r = json.loads(line)
        if (r['arm'] == 'reach' and r['map_name'] == sel['map']
                and r['seed'] == sel['seed'] and r['B'] == sel['B']):
            row = r
            break
    assert row is not None, "paired campaign row not found"
    certs = {c['map']: float(c['a_q']) for c in
             json.load(open(os.path.join(maps_root, 'train',
                                         'certificates.json')))}
    a_q = certs[sel['map']]
    scenario_path = os.path.join(maps_root, 'train', sel['map'] + '.yaml')
    sc = load_scenario_from_yaml(scenario_path)
    jfield = build_field(sc, terrain=sc.terrain)   # boundary contour

    cache_path = os.path.join(outdir, 'fig_oracle_traj.traj.json')
    if args.from_cache:
        c = json.load(open(cache_path))
        assert (c['map'] == sel['map'] and c['seed'] == sel['seed']
                and c['B'] == sel['B'] and c['a_q'] == a_q
                and c['arm_row'] == row and c['oracle_row'] == sel), \
            'cache does not match the freshly selected pair'
        print('[cache] pair + recorded rows verified -> '
              'skipping the solver re-runs')
        arm_out, arm_ret = c['arm_out'], c['arm_ret']
        ora_out, ora_ret = c['ora_out'], c['ora_ret']

        class _R:
            pass
        res = _R()
        res.turnback_time = c['arm_res']['turnback_time']
        res.reach_m = c['arm_res']['reach_m']
        res.margin_at_turnback_J = c['arm_res']['margin_at_turnback_J']
        o = dict(c['oracle_res'])
    else:
        # ---- 2. re-run the constrained arm with recording ----------------
        cfg = load_config(None)
        dt = float(cfg.get('dt', 0.1))

        rec = _Recorder(dt)
        try:
            res = run_reach_trial(sc, ARMS['reach'], B=sel['B'],
                                  seed=sel['seed'], config=cfg,
                                  replan_every_s=2.5, a_q=a_q)
        finally:
            rec.restore()
        problems = []
        _check('arm.outcome', res.outcome, row['outcome'], problems)
        _check('arm.energy_spent', res.energy_spent, row['energy_spent'], problems)
        _check('arm.reach_m', res.reach_m, row['reach_m'], problems)
        _check('arm.reach_J', res.reach_J, row['reach_J'], problems)
        _check('arm.min_margin_J', res.min_margin_J, row['min_margin_J'], problems)
        _check('arm.turnback_time', res.turnback_time, row['turnback_time'], problems)
        _check('arm.turnback_reach_m', res.turnback_reach_m, row['turnback_reach_m'], problems)
        _check('arm.margin_at_turnback_J', res.margin_at_turnback_J,
               row['margin_at_turnback_J'], problems)
        _check('arm.replans', res.replans, row['replans'], problems)
        _check('arm.mission_time', res.mission_time, row['mission_time'], problems)
        if problems:
            print("[FAIL] reach-arm re-run is NOT bit-identical:")
            for p in problems:
                print("   ", p)
            return 1
        print(f"[verify] reach arm bit-identical to the campaign row "
              f"({len(problems)} deviations)")
        arm_out, arm_ret = rec.split(res.turnback_time)

        # ---- 3. re-run the oracle with recording --------------------------
        omod = _load_oracle_module()
        routefield = build_field(sc, terrain=sc.terrain,
                                 r_body=0.25 + omod.ROUTE_BUFFER)
        mfield = build_field(sc)
        rec2 = _Recorder(dt)
        try:
            o = omod.oracle_rollout(sc, jfield, routefield, mfield, sel['B'],
                                    a_q, sel['seed'], cfg)
        finally:
            rec2.restore()
        oproblems = []
        _check('oracle.outcome', o['outcome'], sel['outcome'], oproblems)
        _check('oracle.reach_m', o['reach_m'], sel['reach_m'], oproblems)
        _check('oracle.energy_spent', o['energy_spent'], sel['energy_spent'], oproblems)
        _check('oracle.turnback_time', o['turnback_time'], sel['turnback_time'], oproblems)
        _check('oracle.turnback_reach_m', o['turnback_reach_m'],
               sel['turnback_reach_m'], oproblems)
        _check('oracle.certified_ceiling_m', o['certified_ceiling_m'],
               sel['certified_ceiling_m'], oproblems)
        if oproblems:
            print("[FAIL] oracle re-run is NOT bit-identical:")
            for p in oproblems:
                print("   ", p)
            return 1
        print(f"[verify] oracle bit-identical to the recorded trial "
              f"({len(oproblems)} deviations)")
        ora_out, ora_ret = rec2.split(o['turnback_time'])

        # cache the verified trajectories so figure-iteration rounds do not
        # re-pay the solver wall time (the cache stores the summary fields
        # the verification compared, for a cheap re-check on load)
        json.dump({'map': sel['map'], 'seed': sel['seed'], 'B': sel['B'],
                   'a_q': a_q, 'arm_row': row, 'oracle_row': sel,
                   'arm_out': arm_out, 'arm_ret': arm_ret,
                   'ora_out': ora_out, 'ora_ret': ora_ret,
                   'arm_res': {'outcome': res.outcome,
                               'energy_spent': res.energy_spent,
                               'reach_m': res.reach_m,
                               'margin_at_turnback_J': res.margin_at_turnback_J,
                               'turnback_time': res.turnback_time},
                   'oracle_res': {'outcome': o['outcome'],
                                  'energy_spent': o['energy_spent'],
                                  'reach_m': o['reach_m'],
                                  'turnback_time': o['turnback_time'],
                                  'certified_ceiling_m': o['certified_ceiling_m']}},
                  open(cache_path, 'w'))
        print(f"[cache] trajectories + verification record -> {cache_path}")


    # ---- 4. plot -------------------------------------------------------
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({
        'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.5,
        'lines.linewidth': 1.3, 'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'figure.dpi': 200, 'savefig.dpi': 200,
        'pdf.fonttype': 42,
    })

    w_m = sc.bounds[2] - sc.bounds[0]
    h_m = sc.bounds[3] - sc.bounds[1]
    over = max(max(0.0, oy + orad - sc.bounds[3])
               for ox, oy, orad in sc.obstacles)
    pad = max(max(0.0, max(0.0, ox + orad) - sc.bounds[2])
              for ox, oy, orad in sc.obstacles)
    pad = max(over, pad,
              max(0.0, -min(0.0, min(oy - orad for _, oy, orad in sc.obstacles) - sc.bounds[1]))) + 0.25
    # explicit geometry: axes exactly match the (padded) map aspect, so
    # there is no aspect slack anywhere; the legend band sits above and
    # bbox_inches='tight' trims the rest
    left_in, bottom_in = 0.42, 0.34          # y-label + x-label bands
    legend_in = 0.52
    map_w_in = COL_SINGLE - left_in - 0.06
    # CROP to the mission region: the pair's trajectories, its certified
    # boundary loop, and every obstacle they interact with all live in
    # x < 12.5 m; the arena rectangle deliberately runs off the right
    # edge (standard crop -- the arena extends to 30 m, the caption
    # says so). Assert the crop contains the whole story:
    all_pts = [(p[1], p[2]) for p in arm_out + arm_ret + ora_out + ora_ret]
    crop_x1 = 12.5
    assert max(px for px, _ in all_pts) < crop_x1, \
        'mission exceeds the crop window'
    data_w = crop_x1 - (sc.bounds[0] - 0.6)
    data_h = h_m + 2 * pad
    map_h_in = map_w_in * data_h / data_w
    fig_h = bottom_in + map_h_in + 0.05 + legend_in
    fig = plt.figure(figsize=(COL_SINGLE, fig_h))
    ax = fig.add_axes([left_in / COL_SINGLE,
                       bottom_in / fig_h,
                       map_w_in / COL_SINGLE,
                       map_h_in / fig_h])
    halo = [pe.withStroke(linewidth=2.2, foreground='white')]
    halo_strong = [pe.withStroke(linewidth=2.6, foreground='white')]

    # terrain regions (light fill + dashed edge, the map convention)
    if sc.terrain is not None:
        for reg in sc.terrain.regions:
            ax.add_patch(plt.Rectangle((reg.x0, reg.y0), reg.x1 - reg.x0,
                                       reg.y1 - reg.y0,
                                       facecolor='#edae49', alpha=0.10,
                                       edgecolor='#d8a94e', linewidth=0.8,
                                       linestyle='-',
                                       zorder=1))
    # obstacles: dark filled circles (the geometry both field and plant use)
    for ox, oy, orad in sc.obstacles:
        ax.add_patch(plt.Circle((ox, oy), orad, facecolor='0.15',
                                edgecolor='white', linewidth=0.7,
                                zorder=5))
    # certified boundary 2 D_J + a_q = B  ->  D_J = (B - a_q)/2
    X, Y, D, free = jfield.X, jfield.Y, jfield.D, jfield.free
    in_bounds = (((X[:, None] >= sc.bounds[0])
                  & (X[:, None] <= sc.bounds[2])
                  & (Y[None, :] >= sc.bounds[1])
                  & (Y[None, :] <= sc.bounds[3])))
    level = (sel['B'] - a_q) / 2.0
    has_boundary = False
    try:
        ax.contour(X, Y, np.where(free & in_bounds, D, np.nan).T,
                   levels=[level],
                   colors=['#444444'], linewidths=[1.3], linestyles=['--'],
                   zorder=4,
                   path_effects=[pe.withStroke(linewidth=2.4,
                                               foreground='white')])
        has_boundary = True
    except Exception:
        pass

    # trajectories: outbound solid, return dashed; white halo for
    # visibility over terrain fill and near obstacles
    ax.plot([p[1] for p in arm_out], [p[2] for p in arm_out], '-',
            color=C_ARM, zorder=6, linewidth=1.5, path_effects=halo)
    ax.plot([p[1] for p in arm_ret], [p[2] for p in arm_ret], '--',
            color=C_ARM, zorder=7, linewidth=1.3, path_effects=halo)
    ax.plot([p[1] for p in ora_out], [p[2] for p in ora_out], '-',
            color=C_ORA, zorder=6, linewidth=1.0, alpha=0.9,
            path_effects=halo_strong)
    ax.plot([p[1] for p in ora_ret], [p[2] for p in ora_ret], '--',
            color=C_ORA, zorder=8, dashes=(4, 3), linewidth=0.9,
            alpha=0.9, path_effects=halo_strong)

    # direction arrowheads mid-outbound (the two arms share the corridor
    # near the dock -- the arrows carry the direction where lines coincide)
    for leg, colr in ((arm_out, C_ARM), (ora_out, C_ORA),
                      (arm_ret, C_ARM), (ora_ret, C_ORA)):
        m = len(leg) // 5
        dx, dy = leg[m + 1][1] - leg[m - 1][1], leg[m + 1][2] - leg[m - 1][2]
        n = math.hypot(dx, dy)
        if n > 1e-9:
            ax.annotate('', xy=(leg[m][1] + dx / n * 0.55,
                                leg[m][2] + dy / n * 0.55),
                        xytext=(leg[m][1] - dx / n * 0.55,
                                leg[m][2] - dy / n * 0.55),
                        arrowprops=dict(arrowstyle='-|>', color=colr,
                                        lw=1.1,
                                        path_effects=[pe.withStroke(
                                            linewidth=2.6,
                                            foreground='white')]),
                        zorder=9)

    # turn-back markers (secondary encoding: circle = arm, square = oracle)
    tb_arm = (arm_out[-1] if res.turnback_time is not None else arm_out[-1])
    ax.plot(tb_arm[1], tb_arm[2], 'o', markersize=5.8,
            markerfacecolor=C_ARM, markeredgecolor='white',
            markeredgewidth=1.2, zorder=11, linestyle='',
            path_effects=[pe.withStroke(linewidth=2.8,
                                        foreground='white')])
    tb_ora = (ora_out[-1] if o['turnback_time'] is not None else ora_out[-1])
    ax.plot(tb_ora[1], tb_ora[2], 's', markersize=5.6,
            markerfacecolor=C_ORA, markeredgecolor='white',
            markeredgewidth=1.2, zorder=11, linestyle='',
            path_effects=[pe.withStroke(linewidth=2.8,
                                        foreground='white')])

    # dock: white star, black edge (identical in plot and legend)
    ax.plot(sc.dock[0], sc.dock[1], '*', markersize=12, linestyle='',
            markerfacecolor='white', markeredgecolor='black',
            markeredgewidth=1.0, zorder=10,
            path_effects=[pe.withStroke(linewidth=2.6,
                                         foreground='white')])

    # limits padded so no obstacle is clipped (some maps place circles
    # that overhang the arena edge); `pad` computed with the geometry
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_xlim(sc.bounds[0] - 0.6, crop_x1)
    ax.set_ylim(sc.bounds[1] - pad, sc.bounds[3] + pad)
    ax.set_aspect('equal', adjustable='box')
    ax.add_patch(plt.Rectangle((sc.bounds[0], sc.bounds[1]),
                               sc.bounds[2] - sc.bounds[0],
                               sc.bounds[3] - sc.bounds[1],
                               fill=False, edgecolor='0.35', linewidth=0.9,
                               zorder=3))
    for sp in ax.spines.values():
        sp.set_visible(False)         # the arena rectangle is the frame

    handles = [
        Line2D([], [], color=C_ARM, linestyle='-',
               label='REACH arm: outbound'),
        Line2D([], [], color=C_ARM, linestyle='--',
               label='REACH arm: return'),
        Line2D([], [], color=C_ORA, linestyle='-',
               label='Oracle: outbound'),
        Line2D([], [], color=C_ORA, linestyle='--',
               label='Oracle: return'),
        Line2D([], [], color=C_ARM, marker='o', linestyle='',
               markerfacecolor=C_ARM, markeredgecolor='white',
               markersize=5,
               label=f"REACH turn-back (+{res.margin_at_turnback_J:.1f} J in hand)"),
        Line2D([], [], color=C_ORA, marker='s', linestyle='',
               markerfacecolor=C_ORA, markeredgecolor='white',
               markersize=4.5,
               label=f"Oracle turn-back (returns with {sel['B'] - o['energy_spent']:.1f} J spare)"),
    ]
    if has_boundary:
        handles += [Line2D([], [], color='#1b1b1b', linewidth=1.1,
                           linestyle='--',
                           label='certified turn-back boundary')]
    handles += [
        Line2D([], [], marker='*', linestyle='', markerfacecolor='white',
               markeredgecolor='black', markeredgewidth=0.8,
               markersize=8, label='dock (start = goal)'),
        Line2D([], [], marker='o', color='0.15', markeredgecolor='white',
               markeredgewidth=0.7, linestyle='', markersize=5,
               label='obstacles'),
        Patch(facecolor='#edae49', alpha=0.4, edgecolor='#d8a94e',
              linestyle='-', label='terrain $m{=}3$'),
    ]
    legend_anchor_frac = (bottom_in + map_h_in + 0.06) / fig_h
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, legend_anchor_frac), ncol=2,
               frameon=False,
               borderaxespad=0.1, columnspacing=0.7, handlelength=1.4)

    import numpy as _np
    ax.set_xticks(_np.arange(0, crop_x1 + 0.01, 2.5))
    ax.set_yticks(_np.arange(0, h_m + 0.01, 2.0))
    out_pdf = os.path.join(outdir, 'fig_oracle_traj.pdf')
    out_png = os.path.join(outdir, 'fig_oracle_traj.png')
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)

    # ---- 5. provenance sidecar ----------------------------------------
    write_sidecar(out_pdf, run_id=args.run_id, inputs=[
        oracle_json, main_jsonl, scenario_path,
        os.path.join(maps_root, 'train', 'certificates.json'),
    ], extra={
        'figure': 'fig_oracle_traj',
        'selection_rule': ('training-map pairs of the deployed oracle '
                           'campaign where the oracle docked and the arm '
                           'did not collide; the pair whose arm/ceiling '
                           'fraction is closest to the pooled 0.596'),
        'selected': {k: sel[k] for k in ('map', 'seed', 'B', 'a_q',
                                         'certified_ceiling_m')},
        'pair_fractions': {
            'arm_over_ceiling': res.reach_m / o['certified_ceiling_m'],
            'oracle_over_ceiling': o['reach_m'] / o['certified_ceiling_m'],
        },
        'verification': 'both re-runs bit-identical to the recorded rows '
                         '(outcome, energies, reach, turn-back fields); '
                         'trajectories logged by a recording PioneerEnv '
                         'subclass, no logic touched',
        'arm_margin_at_turnback_J': res.margin_at_turnback_J,
        'oracle_spare_at_dock_J': sel['B'] - o['energy_spent'],
    })
    print(f"[done] {out_pdf}")
    print(f"[done] {out_png}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
