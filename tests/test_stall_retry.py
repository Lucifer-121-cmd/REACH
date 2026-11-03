"""Guard: the stall-retry logic in ReachSupervisor._solve (2026-09-14).

The stall diagnostic (RUN_RECORD, 2026-09-14) measured a warm solve
returning gain = -1.05 J with slack = +87.1 J -- a local optimum leaving
87 J of certified margin unused -- while the COLD solve from the
identical state found gain = +65.0 J. The supervisor read the warm
result as a stall and turned back early. The fix: one cold retry when
gain < 0.5 J AND slack > STALL_RETRY_SLACK on a constraint arm, outbound
only. TRUE boundary stalls (slack ~ 0, reproduced by cold and 5000-iter
solves) must NOT be retried -- that is the honest price of compliance.

These tests pin the gate structure and the branch logic by source
inspection (the full solve path needs IPOPT and minutes; the gate is
what must not silently regress).
"""
import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import reach_evaluator as ev


def test_stall_retry_slack_constant_pinned():
    """The gate constant: high enough to exclude boundary stalls (diagnostic
    cases 0915/0916 stalled at slack 0.00), low enough to catch the
    measured warm-basin stall (slack +87)."""
    assert ev.STALL_RETRY_SLACK == 10.0


def test_stall_retry_gate_structure():
    """The retry must fire ONLY when: warm solve, outbound, constraint arm,
    NOT violating, stalled gain, slack above the gate. Each condition
    exists because its absence was a measured failure."""
    src = inspect.getsource(ev.ReachSupervisor._solve)
    assert 'STALL_RETRY_SLACK' in src
    # the gate conjunction, verbatim pieces
    for fragment in (
            'warm is not None and outbound and self.arm.use_constraint',
            "not res['_violates']",
            "< 0.5",
            "> STALL_RETRY_SLACK"):
        assert fragment in src, fragment
    # the retry must be cold (warm=None) and adopted only on improvement
    assert 'outbound=True,' in src
    assert 'warm=None' in src
    assert "(retry.get('reach_gain') or 0.0) > \\\n                    (res.get('reach_gain') or 0.0)" in src.replace(
        '                    ', '                    ')


def test_stall_retry_counted():
    """Every stall retry is counted so a campaign can see how often the
    warm basin cost a solve."""
    src = inspect.getsource(ev.ReachSupervisor._solve)
    assert 'stall_retries' in src


def test_violation_retry_untouched():
    """The pre-existing violation cold-retry must remain (its own measured
    defect: warm slack -5.0 vs cold +0.0, the 2.0 m premature turn-back)."""
    src = inspect.getsource(ev.ReachSupervisor._solve)
    assert "if res['_violates'] and warm is not None:" in src
