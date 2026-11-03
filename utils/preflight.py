"""Machine-quietness preflight. The timing stages are worthless on a busy box.

Per-step solve time is a headline measurement, and cross-process solve times
in the existing corpus differ 4-6x for
identical genomes. A measurement taken while something else is running is not a
noisy measurement, it is a wrong one -- and it looks publishable.

Two thresholds, deliberately different:

* Bulk statistical stages are throughput work. They tolerate load; they only
  need enough free cores to be worth starting.
* Timing stages need an exclusive machine. They refuse unless the box is quiet.

Used by scripts/run_campaign.py and as the campaign's preflight step.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Dict, List, Optional, Tuple

#: 1-minute load average above which a TIMING stage refuses to start.
TIMING_MAX_LOAD = 2.0

#: Foreign python processes above which a TIMING stage refuses to start.
TIMING_MAX_FOREIGN_PYTHON = 0

#: Fraction of physical cores that must be idle for a BULK stage to start.
BULK_MIN_IDLE_FRACTION = 0.25

#: Busy fraction above which a RESERVED core is considered contended.
RESERVED_CORE_MAX_BUSY = 5.0


def per_core_busy(interval: float = 2.0) -> Dict[int, float]:
    """Busy percentage per logical CPU, sampled over ``interval`` seconds."""
    import time

    def snap() -> Dict[int, tuple]:
        out: Dict[int, tuple] = {}
        with open('/proc/stat') as fh:
            for line in fh:
                if line.startswith('cpu') and len(line) > 3 and line[3].isdigit():
                    parts = line.split()
                    n = int(parts[0][3:])
                    v = [int(x) for x in parts[1:]]
                    out[n] = (v[3] + v[4], sum(v))       # (idle+iowait, total)
        return out

    a = snap()
    time.sleep(max(0.2, interval))
    b = snap()
    busy: Dict[int, float] = {}
    for n, (idle_a, tot_a) in a.items():
        idle_b, tot_b = b.get(n, (idle_a, tot_a))
        d_tot = tot_b - tot_a
        busy[n] = 100.0 * (1.0 - (idle_b - idle_a) / d_tot) if d_tot > 0 else 0.0
    return busy


def quietest_cpus(n: int = 1, interval: float = 2.0,
                  exclude: Optional[List[int]] = None) -> List[int]:
    """The ``n`` least-busy logical CPUs, for reserving a timing core set."""
    busy = per_core_busy(interval)
    for c in (exclude or []):
        busy.pop(c, None)
    return [c for c, _ in sorted(busy.items(), key=lambda kv: kv[1])[:n]]


def physical_cores() -> int:
    """Physical (not logical) core count. Falls back to logical // 2."""
    try:
        out = subprocess.run(['lscpu', '-p=Core,Socket'], capture_output=True,
                             text=True, timeout=30).stdout
        pairs = {l for l in out.splitlines() if l and not l.startswith('#')}
        if pairs:
            return len(pairs)
    except (OSError, subprocess.SubprocessError):
        pass
    n = os.cpu_count() or 2
    return max(1, n // 2)


def load_average() -> Tuple[float, float, float]:
    return os.getloadavg()


def foreign_python(exclude_pids: Optional[List[int]] = None) -> List[Dict]:
    """Python processes that are not us and not our children.

    Deliberately counts processes belonging to *any* user: another account's
    campaign perturbs our timing exactly as much as our own does.
    """
    exclude = set(exclude_pids or []) | {os.getpid(), os.getppid()}
    found = []
    try:
        out = subprocess.run(
            ['ps', '-eo', 'pid,ppid,pcpu,etimes,user:20,args'],
            capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return found
    for line in out.splitlines()[1:]:
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        pid, ppid, pcpu, etimes, user, args = parts
        try:
            pid_i, ppid_i, cpu = int(pid), int(ppid), float(pcpu)
        except ValueError:
            continue
        if pid_i in exclude or ppid_i in exclude:
            continue
        if not re.search(r'(^|/)python[0-9.]*\b', args):
            continue
        if cpu < 5.0:                      # idle interpreters do not perturb timing
            continue
        found.append({'pid': pid_i, 'cpu': cpu, 'user': user,
                      'etimes': int(etimes), 'cmd': args[:120]})
    return sorted(found, key=lambda d: -d['cpu'])


def check(mode: str, *, max_workers: Optional[int] = None,
          exclude_pids: Optional[List[int]] = None,
          cpus: Optional[List[int]] = None,
          interval: float = 2.0) -> Dict:
    """Assess readiness. ``mode`` is 'timing' or 'bulk'.

    For ``mode='timing'`` there are two gates, and which one applies depends on
    whether you are reserving cores:

    * **No ``cpus``** -- gate on global 1-minute load average and on foreign
      python processes. Correct when the box is meant to be doing nothing else.
    * **``cpus`` given** -- gate on the busy fraction of *those* CPUs. This is
      what reserving a core set means ("a core set with nothing
      else running"), and it is the only workable gate on a machine that also
      runs an interactive desktop: gnome-shell, Xorg and a remote-desktop session
      hold the global load around 1.5-2.5 permanently, so a global-load gate can
      never pass and would simply get overridden.

    Returns a dict with ``ok``, ``reasons`` and the raw measurements. Never
    raises and never kills anything -- reporting only.
    """
    cores = physical_cores()
    l1, l5, l15 = load_average()
    procs = foreign_python(exclude_pids)
    busy_cpu = sum(p['cpu'] for p in procs) / 100.0
    reasons: List[str] = []
    core_busy: Dict[int, float] = {}

    if mode == 'timing':
        if cpus:
            # Reserved-core mode: the foreign-python and
            # global-load checks DO NOT APPLY -- the whole point of reserving
            # a core set is that load elsewhere on the box does not matter,
            # only contention on the reserved CPUs. Until 2026-09-14 the
            # foreign-python check ran unconditionally here, which made the
            # reserved-core gate impossible to pass on this machine (the
            # co-tenant lerobot job holds a handful of low-CPU processes
            # indefinitely) -- a gate that can never pass is a gate that gets
            # overridden, the exact failure the docstring warns about. The
            # reserved cores' own busy fraction is the criterion.
            all_busy = per_core_busy(interval)
            core_busy = {c: all_busy.get(c, 100.0) for c in cpus}
            hot = {c: b for c, b in core_busy.items() if b > RESERVED_CORE_MAX_BUSY}
            if hot:
                reasons.append(
                    f"reserved CPUs are contended: "
                    f"{ {c: round(b, 1) for c, b in hot.items()} } "
                    f"(limit {RESERVED_CORE_MAX_BUSY}% busy)")
        else:
            if len(procs) > TIMING_MAX_FOREIGN_PYTHON:
                top = ', '.join(f"pid {p['pid']} ({p['user']}, {p['cpu']:.0f}% CPU, "
                                f"{p['etimes']//3600}h)" for p in procs[:4])
                reasons.append(
                    f"{len(procs)} foreign python process(es) above 5% CPU: {top}")
            if l1 > TIMING_MAX_LOAD:
                reasons.append(
                    f"1-min load average {l1:.2f} exceeds {TIMING_MAX_LOAD} -- timing "
                    f"measurements on a loaded box are wrong, not merely noisy. "
                    f"If the residual load is an interactive "
                    f"desktop rather than another campaign, reserve cores instead: "
                    f"pass --cpu/--cpus and this gate switches to per-core contention.")
    elif mode == 'bulk':
        free = cores - busy_cpu
        if free < BULK_MIN_IDLE_FRACTION * cores:
            reasons.append(
                f"only {free:.0f} of {cores} physical cores appear free "
                f"(< {BULK_MIN_IDLE_FRACTION:.0%})")
        if max_workers is not None and max_workers > free:
            reasons.append(
                f"--max-workers {max_workers} exceeds the {free:.0f} apparently "
                f"free cores; it would oversubscribe and slow everything down")
    else:
        raise ValueError(f"mode must be 'timing' or 'bulk', got {mode!r}")

    return {'ok': not reasons, 'mode': mode, 'reasons': reasons,
            'physical_cores': cores, 'loadavg': [l1, l5, l15],
            'foreign_python': procs, 'foreign_cpu_cores': busy_cpu,
            'reserved_cpus': list(cpus or []), 'reserved_cpu_busy': core_busy,
            'recommended_max_workers': max(1, int(cores - busy_cpu) - 2)}


def format_report(res: Dict) -> str:
    lines = [
        f"preflight ({res['mode']}): {'PASS' if res['ok'] else 'REFUSE'}",
        f"  physical cores       : {res['physical_cores']}",
        f"  load average         : {res['loadavg'][0]:.2f} {res['loadavg'][1]:.2f} {res['loadavg'][2]:.2f}",
        f"  foreign python procs : {len(res['foreign_python'])}"
        f"  ({res['foreign_cpu_cores']:.1f} cores' worth)",
        f"  recommended workers  : {res['recommended_max_workers']}",
    ]
    if res.get('reserved_cpus'):
        busy = res.get('reserved_cpu_busy') or {}
        lines.append(f"  reserved CPUs        : {res['reserved_cpus']}")
        lines.append("  reserved CPU busy%   : "
                     + ', '.join(f"cpu{c}={busy.get(c, float('nan')):.1f}%"
                                 for c in res['reserved_cpus']))
    for p in res['foreign_python'][:6]:
        lines.append(f"      pid {p['pid']:>8}  {p['cpu']:5.0f}%  {p['user']:<14} "
                     f"{p['etimes']//3600:>3}h  {p['cmd'][:70]}")
    for r in res['reasons']:
        lines.append(f"  REFUSE: {r}")
    return '\n'.join(lines)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['timing', 'bulk'], required=True)
    ap.add_argument('--max-workers', type=int, default=None)
    ap.add_argument('--cpus', type=int, nargs='*', default=None,
                    help='Reserved logical CPUs for a timing stage. Switches the '
                         'timing gate from global load average to the busy '
                         'fraction of these CPUs.')
    ap.add_argument('--suggest-cpus', type=int, default=0, metavar='N',
                    help='Print the N quietest logical CPUs and exit.')
    ap.add_argument('--interval', type=float, default=2.0)
    args = ap.parse_args(argv)
    if args.suggest_cpus:
        cpus = quietest_cpus(args.suggest_cpus, interval=args.interval)
        print(' '.join(map(str, cpus)))
        return 0
    res = check(args.mode, max_workers=args.max_workers, cpus=args.cpus,
                interval=args.interval)
    print(format_report(res))
    return 0 if res['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
