"""Unit tests for evolution/pareto.py."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest
from evolution.pareto import dominates, fast_non_dominated_sort, crowding_distance, hypervolume, ParetoArchive

def test_dominance():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 3.0, 4.0, 5.0])
    assert dominates(a, b)
    assert not dominates(b, a)
    c = np.array([1.0, 3.0, 2.0, 4.0])
    assert not dominates(a, c)  # neither dominates

def test_non_dominated_sort():
    objs = np.array([[1, 2], [2, 1], [3, 3], [1.5, 1.5]])
    fronts = fast_non_dominated_sort(objs)
    assert len(fronts) >= 1
    assert 0 in fronts[0] and 1 in fronts[0]

def test_hypervolume_single_point():
    front = np.array([[1.0, 1.0, 1.0, 0.5]])
    ref = np.array([5.0, 5.0, 5.0, 1.0])
    hv = hypervolume(front, ref)
    assert hv > 0

def test_archive_update():
    archive = ParetoArchive()
    from mpc.mpc_params import MPCParams, DEFAULT_PARAMS
    p = MPCParams(**DEFAULT_PARAMS)
    archive.update(p, np.array([400.0, 12.0, 50.0, 0.1]))
    assert archive.size() == 1
    # Dominated point should not be added
    archive.update(p, np.array([500.0, 15.0, 60.0, 0.2]))
    assert archive.size() == 1

def test_crowding_distance():
    objs = np.array([[1, 4], [2, 3], [3, 2], [4, 1]])
    front_indices = [0, 1, 2, 3]
    distances = crowding_distance(objs, front_indices)
    assert distances[0] == np.inf
    assert distances[-1] == np.inf
    assert len(distances) == 4
