"""Evolution modules for EP²-MPC."""
from evolution.pareto import ParetoArchive, dominates, hypervolume
from evolution.population import StratifiedPopulation
from evolution.operators import sbx_crossover, polynomial_mutation
from evolution.stagnation import StagnationDetector, InjectionStrategy
