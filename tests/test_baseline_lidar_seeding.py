"""Guard: the baseline runners must seed the LiDAR exactly like EP2Evaluator.

Found 2026-09-11 on run_20260910T174433Z: F4/B5 wired per-trial LiDAR
seeding into ``EP2Evaluator`` but not into ``run_baseline`` /
``run_standard_mpc_casadi_baseline`` -- both called ``env.reset(start)``
with no ``lidar_seed``. Consequences, measured: baseline energies differed
between otherwise identical runs (per-trial +-0.5-3 J while the EP2 arm was
bit-identical), and "paired" trials did not see the same scan noise on the
two arms, so the pairing was nominal rather than exact.
"""
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _src():
    with open(os.path.join(REPO, 'evaluation/baselines.py'), encoding='utf-8') as f:
        return f.read()


def test_both_baseline_reset_calls_seed_lidar():
    """Every env.reset in the baseline runners passes lidar_seed with the
    same derivation the evaluator uses (seed + LIDAR_SEED_OFFSET)."""
    src = _src()
    resets = []
    for i, line in enumerate(src.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue          # comments may quote the defective call verbatim
        if 'env.reset(' in line:
            resets.append((i, line.strip()))
    assert resets, "no env.reset calls found in baselines.py"
    for lineno, call in resets:
        assert 'lidar_seed' in call, (
            f"line {lineno}: env.reset call without lidar_seed: {call!r}. An "
            "unseeded baseline trial is not bit-reproducible and breaks exact "
            "pairing with the EP2 arm.")


def test_offset_matches_evaluator():
    """The offset constant must equal EP2Evaluator's, or the two arms see
    different scan streams on the same trial."""
    from evaluation.baselines import LIDAR_SEED_OFFSET
    from evaluation.ep2_evaluator import EP2Evaluator
    ev = EP2Evaluator(config={'robot': {}, 'mpc': {}, 'evaluation': {}})
    assert LIDAR_SEED_OFFSET == ev.lidar_seed_offset == 1_000_003


def test_seed_derivation_is_per_trial():
    """The seed passed must vary per trial (master_seed + trial + offset),
    not be a fixed constant."""
    src = _src()
    assert src.count('lidar_seed=master_seed + trial + LIDAR_SEED_OFFSET') == 2, (
        "both baseline runners must derive the LiDAR seed per trial; a "
        "constant seed would make every trial's scan identical")
