"""Regression tests for post-analysis issue-alignment fixes."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpc.mpc_params import PARAM_BOUNDS
from evolution.llm_evolution import LLMEvolution
from evaluation.baselines import collect_all_baselines


def test_d_safe_bound_excludes_sub_collision_dead_zone():
    """The d_safe box must sit entirely above the collision radius.

    Rewritten 2026-09-01. The previous assertion was ``lo == 0.05 and hi ==
    0.26``, which permitted the whole sub-collision dead zone: with
    d_safe < r_body = 0.25 the obstacle penalty ``fmax(0, d_safe - d_surface)²``
    is identically zero at every collision-free state, so the term contributes
    nothing until the robot is already in contact. The current box is
    [0.26, 0.38] and the whole point is that its lower end clears 0.25.
    """
    lo, hi = PARAM_BOUNDS['d_safe']
    assert (lo, hi) == (0.26, 0.38)
    assert lo > 0.25, "the dead zone below the collision radius must stay excluded"


def test_every_declared_d_safe_range_agrees():
    """The box is declared in four places; they must not drift apart."""
    import re
    lo, hi = PARAM_BOUNDS['d_safe']
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel, pattern in [
        ('llm/prompts_params.py',       r'd_safe\s*\|\s*0\.26\s*\|\s*0\.38'),
        ('llm/prompts_strategy.py',     r'd_safe\s*∈\s*\[0\.26,\s*0\.38\]'),
        ('evolution/llm_evolution.py',  r'd_safe MUST be >= 0\.26'),
    ]:
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            continue
        text = open(path, encoding='utf-8').read()
        assert re.search(pattern, text), (
            f"{rel} no longer declares the d_safe box as [{lo}, {hi}]; the LLM "
            f"prompt and the validator have drifted apart"
        )


def test_nsga2_no_llm_strategy_does_not_require_llm():
    evo = LLMEvolution(
        config={
            'evolution': {
                'llm_params_mode': 'random',
                'use_llm_params': False,
            },
            'llm': {},
        },
        scenario_name='corridor'
    )
    strategy = evo._llm_generate_strategy({'generation': 0})
    assert 'LLM strategy disabled' in strategy


def test_collect_all_baselines_includes_real_casadi_baseline(monkeypatch):
    """Rewritten for F8/C2: one reactive arm, and config['robot'] is mandatory.

    The previous version asserted ``'reactive_standard' in out`` and passed
    ``config={}``. Both are now wrong on purpose: reactive_standard and
    energy_aware_apf are deleted arms, and defaulting the robot limits is exactly
    how the APF arms ended up running at v_max=1.0 against the MPC's 1.2.
    """
    from evaluation import baselines as b

    class DummyResult:
        energy_mean = 0.0
        energy_std = 0.0
        time_mean = 0.0
        time_std = 0.0
        solver_ms_mean = 0.0
        solver_ms_std = 0.0
        success_rate = 1.0
        path_distance_mean = 0.0
        path_distance_std = 0.0
        n_trials = 1
        trial_results = []

    monkeypatch.setattr(b, 'run_baseline', lambda *args, **kwargs: DummyResult())
    monkeypatch.setattr(b, 'run_standard_mpc_casadi_baseline', lambda *args, **kwargs: DummyResult())

    robot = {'max_v': 1.2, 'max_omega': 3.14159, 'collision_radius': 0.25}
    out = collect_all_baselines('corridor', n_trials=1,
                                config={'robot': robot, 'mpc': {'dt': 0.1}})
    assert 'standard_mpc_casadi' in out
    assert 'apf' in out
    assert 'reactive_standard' not in out, "deleted arm resurrected"
    assert 'energy_aware_apf' not in out, "deleted arm resurrected"


def test_collect_all_baselines_requires_robot_config():
    """A missing config['robot'] must raise, not silently reinstate a handicap."""
    import pytest
    with pytest.raises(ValueError, match='robot'):
        collect_all_baselines('corridor', n_trials=1, config={})


def test_no_baseline_arm_is_speed_handicapped():
    """Every arm must run at the robot's real limits, not a hard-coded literal."""
    from evaluation.baselines import APFBaseline
    robot = {'max_v': 1.2, 'max_omega': 3.14159, 'collision_radius': 0.25}
    c = APFBaseline(robot=robot)
    assert (c.v_max, c.omega_max) == (1.2, 3.14159)


def test_standard_mpc_d_safe_clears_the_body_radius():
    """The comparator's d_safe was 0.23-0.245 against a 0.25 m radius."""
    from evaluation.baselines import SCENARIO_STD_MPC_WEIGHTS
    from mpc.standard_mpc_casadi import StandardMPCWeights
    for scen, w in SCENARIO_STD_MPC_WEIGHTS.items():
        assert w['d_safe'] > 0.25, f"{scen}: d_safe={w['d_safe']} is inside the body"
        assert 0.26 <= w['d_safe'] <= 0.38, f"{scen}: outside EP2's own box"
        assert 'w_v' not in w, f"{scen}: the velocity-forcing term is back"
    assert not hasattr(StandardMPCWeights(), 'w_v')
    assert StandardMPCWeights().d_safe > 0.25


def test_apf_gains_are_the_fitted_ones():
    """The APF gains must be the FITTED values, and every succession recorded.

    Guard, not decoration. Reverting to any earlier value silently re-cripples the
    comparator while every test that only checks shapes and speed limits keeps
    passing -- which is exactly how the original submission acquired a crippled
    comparator. Three successive fits, each forced by a geometry change:

      (2.2, 2.2)        pre-C2, fitted centre-to-surface, no wall term
      (1.0, 2.2)        after C1/C2 added r_body; s_curve 0.00 -> 1.00, chicane
                        still 0.00 because the force law had NO wall term
      (0.02, 2.2, 1.20) after the wall term was added; chicane 0.00 -> 1.000, all
                        three maps at 1.000, validated 250 trials/map

    Provenance: scripts/17_tune_baselines.py --arm apf, declared 350-point box,
    42,000 trials, recorded in <RUN_ID>/tuning/apf_gain_sweep.json. The superseded
    boundary winner is kept as apf_gain_sweep.BOX1_EDGE_WINNER.json.
    """
    import inspect
    from evaluation.baselines import (APFBaseline, APF_K_REP, APF_D_INFLUENCE,
                                      APF_D_WALL_INFLUENCE, APF_GAINS_PRE_C2)
    assert (APF_K_REP, APF_D_INFLUENCE, APF_D_WALL_INFLUENCE) == (0.02, 2.2, 1.20)
    assert APF_GAINS_PRE_C2 == (2.2, 2.2), 'the historical reference moved'
    assert (APF_K_REP, APF_D_INFLUENCE) != APF_GAINS_PRE_C2

    robot = {'max_v': 1.2, 'max_omega': 3.14159, 'collision_radius': 0.25}
    c = APFBaseline(robot=robot)
    assert (c.k_rep, c.d_influence) == (APF_K_REP, APF_D_INFLUENCE)
    assert c.d_wall_influence == APF_D_WALL_INFLUENCE

    # One source of truth: the constructor and the force law must not carry
    # independent literals that can drift apart.
    from evaluation.baselines import _potential_field_command
    sig = inspect.signature(_potential_field_command)
    assert sig.parameters['k_rep'].default == APF_K_REP
    assert sig.parameters['d_influence'].default == APF_D_INFLUENCE


def test_std_mpc_margin_matches_ep2s_own_search_bound():
    """The comparator's d_safe must be inside EP2's OWN bound, not stricter.

    All 18 fully-successful grid points have d_safe = 0.26, which is the floor of
    PARAM_BOUNDS['d_safe'] = (0.26, 0.38) -- the range EP2's evolution could itself
    explore. That boundary is principled and must not be widened downward: giving
    the baseline a margin EP2 was never allowed to consider would bias the
    comparison in the baseline's favour and is indefensible to a reviewer.
    """
    from evaluation.baselines import (SCENARIO_STD_MPC_WEIGHTS,
                                      SCENARIO_STD_MPC_WEIGHTS_PRE_WALLS)
    from mpc.mpc_params import PARAM_BOUNDS
    lo, hi = PARAM_BOUNDS['d_safe']
    for scen, w in SCENARIO_STD_MPC_WEIGHTS.items():
        assert lo <= w['d_safe'] <= hi, (
            f"{scen}: d_safe={w['d_safe']} outside EP2's own search bound "
            f"[{lo}, {hi}]")
        assert w['d_safe'] > 0.25, f"{scen}: d_safe is inside the body radius"
        assert 'w_v' not in w, f"{scen}: the velocity-forcing term is back"
    # The pre-wall values are kept for the tuning-parity table, and must not be
    # what the code actually uses.
    assert SCENARIO_STD_MPC_WEIGHTS != SCENARIO_STD_MPC_WEIGHTS_PRE_WALLS


def test_apf_clearance_is_body_to_surface():
    """The re-tune is only correct if the convention it was fitted to is in force."""
    from evaluation.baselines import _potential_field_command
    state = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
    obstacles = [(1.6, 0.6, 0.3)]
    # Obstacle offset laterally so the repulsion has a y-component and the
    # command is not saturated: a head-on obstacle drives omega to +/-omega_max
    # under both conventions and hides the difference. Both clearances exceed the
    # 0.8 m near-obstacle speed cap, so d_obs plays no part and the whole
    # difference comes from the repulsion magnitude, which is the point.
    d_center = 1.6 ** 2 + 0.6 ** 2
    d_center **= 0.5
    with_body = _potential_field_command(state, (5.0, 0.0), obstacles,
                                         d_center - 0.3 - 0.25,
                                         1.2, 3.14159, r_body=0.25)
    without = _potential_field_command(state, (5.0, 0.0), obstacles,
                                       d_center - 0.3,
                                       1.2, 3.14159, r_body=0.0)
    assert abs(with_body[1]) < 3.14159 and abs(without[1]) < 3.14159, 'saturated'
    # Body-to-surface clearance is 0.25 m smaller, so the repulsion is stronger
    # and the robot must turn harder away from the same obstacle.
    assert abs(with_body[1]) > abs(without[1]) * 1.5, (
        f"r_body is being ignored; C1/C2 geometry is not in force: "
        f"omega {with_body[1]:.4f} vs {without[1]:.4f}")
