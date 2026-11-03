"""Unit tests for env/pioneer_env.py."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest
from env.pioneer_env import PioneerEnv

def make_env():
    return PioneerEnv()

def test_reset_returns_valid_state():
    env = make_env()
    state = env.reset(np.array([1.0, 2.0, 0.5]))
    assert abs(state['x'] - 1.0) < 1e-9
    assert abs(state['y'] - 2.0) < 1e-9
    assert state['cumulative_energy'] == 0.0

def test_step_advances_position():
    env = make_env()
    env.reset(np.array([0.0, 0.0, 0.0]))
    state, _, _, _ = env.step(0.5, 0.0)
    assert state['x'] > 0.0

def test_collision_detection():
    env = make_env()
    env.set_scenario([(1.0, 0.0, 0.3)])
    env.reset(np.array([0.0, 0.0, 0.0]))
    # Move robot into obstacle
    for _ in range(15):
        env.step(0.5, 0.0)
    assert env.check_collision()

def test_goal_reached():
    env = make_env()
    env.reset(np.array([0.0, 0.0, 0.0]))
    goal = np.array([0.1, 0.0])
    assert env.is_goal_reached(goal, tolerance=0.3)

def test_energy_accumulation():
    env = make_env()
    env.reset()
    e0 = env.get_state()['cumulative_energy']
    env.step(0.5, 0.0)
    e1 = env.get_state()['cumulative_energy']
    assert e1 > e0
