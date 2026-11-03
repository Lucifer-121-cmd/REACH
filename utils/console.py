"""Rich console formatting for EP²-MPC evolution monitoring.

Provides box-framed, color-coded output for each generation and candidate.
Falls back to plain-text formatting when ``rich`` is not installed.
"""
from __future__ import annotations

import sys
import time
from typing import Dict, Any, List, Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.columns import Columns
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
    from rich.live import Live
    from rich.layout import Layout
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────
# Colour palette (colorblind-safe)
# ──────────────────────────────────────────────────────────────────────
_C = {
    'title':    'bold cyan',
    'success':  'bold green',
    'fail':     'bold red',
    'warn':     'bold yellow',
    'energy':   'blue',
    'time':     'magenta',
    'solver':   'cyan',
    'hv':       'bold white',
    'dim':      'dim',
    'cand_ok':  'green',
    'cand_bad': 'red',
}

_console: Optional[Any] = None


def _get_console():
    global _console
    if _console is None and RICH_AVAILABLE:
        _console = Console(width=120)
    return _console


# ──────────────────────────────────────────────────────────────────────
# Public API — all functions work with or without ``rich``
# ──────────────────────────────────────────────────────────────────────

def print_banner():
    """Print EP²-MPC banner at startup."""
    if RICH_AVAILABLE:
        c = _get_console()
        banner = Text()
        banner.append("  EP²-MPC  ", style="bold white on blue")
        banner.append("  Energy-Predictive Proximity-Adaptive MPC  ", style="bold cyan")
        banner.append("  LLM-Guided Evolution  ", style="bold white on dark_green")
        c.print()
        c.print(Panel(banner, box=box.DOUBLE, border_style="cyan",
                       title="[bold]EP²-MPC Evolution Engine[/bold]",
                       subtitle="Pioneer 3-DX • CasADi+IPOPT • GLM/Qwen"))
        c.print()
    else:
        print("=" * 80)
        print("  EP²-MPC · Energy-Predictive Proximity-Adaptive MPC")
        print("  LLM-Guided Multi-Objective Evolution")
        print("=" * 80)
        print()


def print_generation_header(gen: int, n_gens: int, scenario: str):
    """Print generation header."""
    if RICH_AVAILABLE:
        c = _get_console()
        c.rule(f"[bold cyan]Generation {gen}/{n_gens}[/bold cyan]  │  "
               f"[dim]{scenario}[/dim]", style="cyan")
    else:
        print(f"\n{'─' * 80}")
        print(f"  Generation {gen}/{n_gens}  │  {scenario}")
        print(f"{'─' * 80}")


def print_candidate_result(
    idx: int,
    total: int,
    energy: float,
    nav_time: float,
    solver_ms: float,
    success_rate: float,
    path_dist: float = 0.0,
    is_best: bool = False,
    server: str = "S1",
):
    """Print single candidate evaluation result on one line."""
    success_pct = success_rate * 100
    tag = "★" if is_best else " "

    if RICH_AVAILABLE:
        c = _get_console()
        status_style = _C['cand_ok'] if success_rate >= 0.8 else _C['cand_bad']
        line = (
            f"  {tag} [{_C['dim']}][{server}][/{_C['dim']}] "
            f"Cand {idx + 1:2d}/{total} │ "
            f"[{_C['energy']}]E={energy:7.1f}J[/{_C['energy']}] │ "
            f"[{_C['time']}]T={nav_time:5.1f}s[/{_C['time']}] │ "
            f"[{_C['solver']}]S={solver_ms:6.1f}ms[/{_C['solver']}] │ "
            f"D={path_dist:5.1f}m │ "
            f"[{status_style}]{success_pct:5.1f}%[/{status_style}]"
        )
        c.print(line)
    else:
        ok = "✓" if success_rate >= 0.8 else "✗"
        print(f"  {tag} [{server}] Cand {idx + 1:2d}/{total} │ "
              f"E={energy:7.1f}J │ T={nav_time:5.1f}s │ "
              f"S={solver_ms:6.1f}ms │ D={path_dist:5.1f}m │ "
              f"{success_pct:5.1f}% {ok}")


def print_generation_summary(
    gen: int,
    hv: float,
    hv_prev: float,
    archive_size: int,
    best_energy: float,
    best_time: float,
    best_success: float,
    trend: str,
    strategy: str = "",
):
    """Print summary box after all candidates in a generation."""
    hv_delta = hv - hv_prev if hv_prev > 0 else hv
    hv_pct = (hv_delta / hv_prev * 100) if hv_prev > 0 else 0.0

    if RICH_AVAILABLE:
        c = _get_console()
        trend_style = {'improving': _C['success'], 'stagnating': _C['warn'],
                       'declining': _C['fail']}.get(trend, _C['dim'])

        table = Table(box=box.ROUNDED, show_header=False, border_style="blue",
                      title=f"[bold]Gen {gen} Summary[/bold]",
                      title_style="bold white", padding=(0, 1))
        table.add_column(width=22)
        table.add_column(width=28)
        table.add_column(width=22)
        table.add_column(width=28)

        table.add_row(
            f"[{_C['hv']}]HV[/{_C['hv']}]",
            f"{hv:.4f}  ({hv_pct:+.2f}%)",
            f"[{_C['energy']}]Best Energy[/{_C['energy']}]",
            f"{best_energy:.1f} J",
        )
        table.add_row(
            "Archive",
            f"{archive_size} solutions",
            f"[{_C['time']}]Best Time[/{_C['time']}]",
            f"{best_time:.1f} s",
        )
        table.add_row(
            "Trend",
            f"[{trend_style}]{trend}[/{trend_style}]",
            "Best Success",
            f"{best_success:.0%}",
        )
        c.print(table)

        if strategy and len(strategy) > 5:
            c.print(Panel(strategy[:300], title="[dim]LLM Strategy[/dim]",
                          border_style="dim", width=100))
    else:
        arrow = "↑" if hv_delta > 0 else ("→" if abs(hv_delta) < 1e-6 else "↓")
        print(f"\n  ┌{'─' * 76}┐")
        print(f"  │  Gen {gen} │ HV={hv:.4f} ({hv_pct:+.2f}%) {arrow} │ "
              f"Archive={archive_size} │ Trend={trend:12s}  │")
        print(f"  │  Best: E={best_energy:.1f}J  T={best_time:.1f}s  "
              f"Success={best_success:.0%}" + " " * 30 + "│")
        print(f"  └{'─' * 76}┘")
        if strategy and len(strategy) > 5:
            print(f"  LLM: {strategy[:120]}")


def print_stagnation_alert(n_inject: int):
    """Print stagnation detection alert."""
    if RICH_AVAILABLE:
        c = _get_console()
        c.print(f"  [bold yellow]⚠ Stagnation detected[/bold yellow] — "
                f"injecting {n_inject} diverse candidates")
    else:
        print(f"  ⚠ STAGNATION — injecting {n_inject} diverse candidates")


def print_phase_complete(scenario: str, total_time_s: float, final_hv: float):
    """Print phase completion banner."""
    mins = total_time_s / 60
    if RICH_AVAILABLE:
        c = _get_console()
        c.print()
        c.print(Panel(
            f"[bold green]✓ {scenario} evolution complete[/bold green]\n"
            f"  Time: {mins:.1f} min  │  Final HV: {final_hv:.4f}",
            box=box.DOUBLE, border_style="green",
        ))
    else:
        print(f"\n{'═' * 60}")
        print(f"  ✓ {scenario} complete  │  {mins:.1f} min  │  HV={final_hv:.4f}")
        print(f"{'═' * 60}")


def print_gate_check(gate_name: str, passed: bool, details: str = ""):
    """Print verification gate result."""
    if RICH_AVAILABLE:
        c = _get_console()
        if passed:
            c.print(f"  [{_C['success']}]✓ {gate_name} PASSED[/{_C['success']}] {details}")
        else:
            c.print(f"  [{_C['fail']}]✗ {gate_name} FAILED[/{_C['fail']}] {details}")
    else:
        tag = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {tag}: {gate_name} {details}")


def print_baseline_table(results: Dict[str, Dict[str, Any]], scenario: str):
    """Print formatted baseline comparison table."""
    if RICH_AVAILABLE:
        c = _get_console()
        table = Table(title=f"Baselines — {scenario}", box=box.ROUNDED,
                      border_style="blue")
        table.add_column("Method", style="bold", width=20)
        table.add_column("Energy (J)", justify="right", width=12)
        table.add_column("Time (s)", justify="right", width=10)
        table.add_column("Solve (ms)", justify="right", width=11)
        table.add_column("Dist (m)", justify="right", width=10)
        table.add_column("Success", justify="right", width=9)

        for name, r in results.items():
            sr = r.get('success_rate', 0)
            sr_style = "green" if sr >= 0.95 else ("yellow" if sr >= 0.5 else "red")
            table.add_row(
                name,
                f"{r['energy_mean']:.1f} ± {r.get('energy_std', 0):.1f}",
                f"{r['time_mean']:.1f}",
                f"{r.get('solver_ms_mean', 0):.2f}",
                f"{r.get('path_distance_mean', 0):.1f}",
                f"[{sr_style}]{sr:.0%}[/{sr_style}]",
            )
        c.print(table)
    else:
        print(f"\n  Baselines — {scenario}")
        print(f"  {'Method':20s} {'Energy':>10s} {'Time':>8s} {'Solve':>10s} "
              f"{'Dist':>8s} {'Success':>8s}")
        print(f"  {'─' * 70}")
        for name, r in results.items():
            print(f"  {name:20s} {r['energy_mean']:10.1f} {r['time_mean']:8.1f} "
                  f"{r.get('solver_ms_mean', 0):10.2f} "
                  f"{r.get('path_distance_mean', 0):8.1f} "
                  f"{r.get('success_rate', 0):8.0%}")


def create_progress(total: int, description: str = "Evaluating"):
    """Create a progress bar (returns context manager or None)."""
    if RICH_AVAILABLE:
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=_get_console(),
        )
        task = progress.add_task(description, total=total)
        return progress, task
    return None, None
