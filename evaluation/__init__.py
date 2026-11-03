"""Evaluation modules for EP²-MPC."""
from evaluation.ep2_evaluator import EP2Evaluator, EvalResult
from evaluation.baselines import collect_all_baselines
from evaluation.statistics import wilcoxon_test, cliffs_delta
