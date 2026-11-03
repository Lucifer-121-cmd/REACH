"""Genetic operators for EP²-MPC parameter evolution."""
import numpy as np
from typing import List, Optional, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpc.mpc_params import (
    MPCParams, PARAM_NAMES, PARAM_BOUNDS, params_to_array, array_to_params, clamp_params, get_bounds_array
)


def sbx_crossover(
    parent1: MPCParams,
    parent2: MPCParams,
    eta: float = 15.0,
    rng: Optional[np.random.Generator] = None
) -> Tuple[MPCParams, MPCParams]:
    """Simulated Binary Crossover (SBX) for real-valued parameter vectors.
    
    Produces offspring that mimic the distribution of single-point crossover
    in binary-encoded GAs. Higher eta → offspring closer to parents.
    
    Args:
        parent1, parent2: MPCParams
        eta: distribution index (larger = less exploratory)
        rng: random number generator
    
    Returns:
        (child1, child2) MPCParams
    """
    if rng is None:
        rng = np.random.default_rng()
    
    p1 = params_to_array(parent1)
    p2 = params_to_array(parent2)
    lbs, ubs = get_bounds_array()
    
    child1 = p1.copy()
    child2 = p2.copy()
    
    for i in range(len(p1)):
        if rng.random() > 0.5:
            continue  # Skip this gene with 50% probability
        
        y1, y2 = min(p1[i], p2[i]), max(p1[i], p2[i])
        
        if abs(y2 - y1) < 1e-10:
            child1[i] = y1
            child2[i] = y2
            continue
        
        lo, hi = lbs[i], ubs[i]
        
        # Compute beta
        u = rng.random()
        beta_left = 1.0 + 2.0 * (y1 - lo) / (y2 - y1)
        beta_right = 1.0 + 2.0 * (hi - y2) / (y2 - y1)
        
        alpha_left = 2.0 - beta_left ** (-(eta + 1.0))
        alpha_right = 2.0 - beta_right ** (-(eta + 1.0))
        
        if u <= 1.0 / alpha_left:
            beta_q_left = (u * alpha_left) ** (1.0 / (eta + 1.0))
        else:
            beta_q_left = (1.0 / (2.0 - u * alpha_left)) ** (1.0 / (eta + 1.0))
        
        if u <= 1.0 / alpha_right:
            beta_q_right = (u * alpha_right) ** (1.0 / (eta + 1.0))
        else:
            beta_q_right = (1.0 / (2.0 - u * alpha_right)) ** (1.0 / (eta + 1.0))
        
        c1 = 0.5 * ((y1 + y2) - beta_q_left * (y2 - y1))
        c2 = 0.5 * ((y1 + y2) + beta_q_right * (y2 - y1))
        
        child1[i] = np.clip(c1, lo, hi)
        child2[i] = np.clip(c2, lo, hi)
    
    return clamp_params(array_to_params(child1)), clamp_params(array_to_params(child2))


def polynomial_mutation(
    individual: MPCParams,
    eta: float = 20.0,
    mutation_prob: float = 0.1,
    rng: Optional[np.random.Generator] = None
) -> MPCParams:
    """Polynomial mutation for real-valued parameters.
    
    Each parameter is mutated independently with probability mutation_prob.
    Higher eta → smaller mutations.
    
    Args:
        individual: MPCParams to mutate
        eta: distribution index
        mutation_prob: probability of mutating each gene
        rng: random number generator
    
    Returns:
        mutated MPCParams
    """
    if rng is None:
        rng = np.random.default_rng()
    
    arr = params_to_array(individual)
    lbs, ubs = get_bounds_array()
    result = arr.copy()
    
    for i in range(len(arr)):
        if rng.random() > mutation_prob:
            continue
        
        y = arr[i]
        lo, hi = lbs[i], ubs[i]
        delta = hi - lo
        if delta < 1e-10:
            continue
        
        u = rng.random()
        delta1 = (y - lo) / delta
        delta2 = (hi - y) / delta
        
        if u <= 0.5:
            xy = 1.0 - delta1
            val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (eta + 1.0))
            delta_q = val ** (1.0 / (eta + 1.0)) - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (eta + 1.0))
            delta_q = 1.0 - val ** (1.0 / (eta + 1.0))
        
        result[i] = np.clip(y + delta_q * delta, lo, hi)
    
    return clamp_params(array_to_params(result))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.
    
    Returns value in [-1, 1], where 1 = identical direction.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def diversity_filter(
    candidate: MPCParams,
    population: List[MPCParams],
    threshold: float = 0.95
) -> bool:
    """Reject candidate if too similar to any existing population member.
    
    Uses cosine similarity on normalized parameter vectors.
    
    Args:
        candidate: candidate MPCParams
        population: existing population
        threshold: similarity threshold (reject if sim > threshold)
    
    Returns:
        True if candidate is sufficiently diverse (keep)
        False if too similar to existing member (reject)
    """
    if not population:
        return True
    
    lbs, ubs = get_bounds_array()
    delta = ubs - lbs
    
    def normalize(params: MPCParams) -> np.ndarray:
        """Normalize parameter vector to [0, 1] using parameter bounds."""
        arr = params_to_array(params)
        return (arr - lbs) / np.where(delta > 0, delta, 1.0)
    
    cand_norm = normalize(candidate)
    
    for member in population:
        mem_norm = normalize(member)
        sim = cosine_similarity(cand_norm, mem_norm)
        if sim > threshold:
            return False
    
    return True


def tournament_selection(
    population: List[MPCParams],
    objectives: np.ndarray,
    k: int = 2,
    rng: Optional[np.random.Generator] = None
) -> MPCParams:
    """Tournament selection based on Pareto rank and crowding distance.
    
    Selects the best candidate from k randomly chosen individuals.
    Prefers lower Pareto rank; breaks ties with crowding distance.
    
    Args:
        population: list of MPCParams
        objectives: (n_solutions, n_obj) array
        k: tournament size
        rng: random number generator
    
    Returns:
        selected MPCParams
    """
    from evolution.pareto import fast_non_dominated_sort, crowding_distance
    
    if rng is None:
        rng = np.random.default_rng()
    
    n = len(population)
    if n == 0:
        raise ValueError("Empty population")
    if n == 1:
        return population[0]
    
    # Compute Pareto ranks
    fronts = fast_non_dominated_sort(objectives)
    ranks = np.zeros(n, dtype=int)
    for rank, front in enumerate(fronts):
        for idx in front:
            ranks[idx] = rank
    
    # Compute crowding distances
    all_distances = np.zeros(n)
    for rank_idx, front in enumerate(fronts):
        if len(front) > 0:
            dists = crowding_distance(objectives, front)
            for local_i, global_i in enumerate(front):
                all_distances[global_i] = dists[local_i]
    
    # Tournament
    candidates = rng.choice(n, size=min(k, n), replace=False)
    best_idx = candidates[0]
    for idx in candidates[1:]:
        if ranks[idx] < ranks[best_idx]:
            best_idx = idx
        elif ranks[idx] == ranks[best_idx] and all_distances[idx] > all_distances[best_idx]:
            best_idx = idx
    
    return population[best_idx]
