"""Main evolutionary loop for EP²-MPC with LLM guidance."""
import numpy as np
import json
import os
import time
import logging
import csv
from dataclasses import asdict
from typing import List, Optional, Dict, Any, Tuple
import sys

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:
    logger.info("tqdm not available; progress bars disabled.")
    tqdm = None

from mpc.mpc_params import MPCParams, params_to_array, array_to_params, clamp_params
from evolution.pareto import ParetoArchive, fast_non_dominated_sort
from evolution.population import StratifiedPopulation
from evolution.operators import sbx_crossover, polynomial_mutation, tournament_selection, diversity_filter
from evolution.stagnation import StagnationDetector, InjectionStrategy


class LLMEvolution:
    """EP²-MPC evolutionary optimization loop with LLM guidance.
    
    Each generation:
    1. Evaluate population on scenario (50 trials each)
    2. Update Pareto archive
    3. Query LLM strategy (GLM-4.7-Flash)
    4. Generate new candidates with LLM (Qwen3:14b)
    5. Apply SBX + mutation
    6. Check stagnation, inject if needed
    7. Save checkpoint
    """

    def __init__(self, config: Dict[str, Any], scenario_name: str):
        self.config = config
        self.scenario_name = scenario_name
        self.results_dir = config.get('results_dir', 'results')
        
        evo_cfg = config.get('evolution', {})
        self.evo_cfg = evo_cfg
        self.pop_size = evo_cfg.get('pop_size', 16)
        self.sbx_eta = evo_cfg.get('sbx_eta', 15)
        self.mutation_eta = evo_cfg.get('mutation_eta', 20)
        self.cos_threshold = evo_cfg.get('cosine_similarity_threshold', 0.95)
        self.injection_count = evo_cfg.get('injection_count', 6)
        self.stagnation_window = evo_cfg.get('stagnation_window', 3)
        self.stagnation_threshold = evo_cfg.get('stagnation_threshold', 0.001)
        self.use_tqdm = evo_cfg.get('use_tqdm', True)
        
        self.archive = ParetoArchive()
        self.stagnation_detector = StagnationDetector(
            window=self.stagnation_window,
            threshold=self.stagnation_threshold
        )
        self.injection_strategy = InjectionStrategy()
        self.hv_history: List[float] = []
        # Optional reproducibility seed (None keeps prior non-deterministic behavior).
        self.seed = evo_cfg.get('seed', None)
        self.rng = np.random.default_rng(seed=self.seed)
        self._evaluator = None
        self._llm_strategy_client = None
        self._llm_params_client = None
        self._last_llm_strategy: Optional[str] = None
        self._last_llm_params: Optional[List[Dict[str, Any]]] = None

    def _get_evaluator(self):
        """Lazily initialize evaluator."""
        if self._evaluator is None:
            from evaluation.ep2_evaluator import EP2Evaluator
            self._evaluator = EP2Evaluator(self.config)
        return self._evaluator

    def _get_llm_clients(self):
        """Lazily initialize LLM clients."""
        if self._llm_strategy_client is None:
            from llm.ollama_client import StrategyClient, ParamsClient
            llm_cfg = self.config.get('llm', {})
            # FIX M2: read separate base_url_strategy and base_url_params
            # (fallback to the legacy single base_url if not set)
            _default_url = llm_cfg.get('base_url', 'http://localhost:11434')
            _url_strategy = llm_cfg.get('base_url_strategy', _default_url)
            _url_params   = llm_cfg.get('base_url_params',   _default_url)
            _ttl = llm_cfg.get('availability_cache_ttl_sec', 30)

            self._llm_strategy_client = StrategyClient(
                model=llm_cfg.get('model_strategy', 'glm-4.7-flash'),
                timeout=None,
                base_url=_url_strategy,
                availability_cache_ttl_sec=_ttl,
            )
            self._llm_params_client = ParamsClient(
                model=llm_cfg.get('model_params', 'qwen3:14b'),
                timeout=None,
                base_url=_url_params,
                availability_cache_ttl_sec=_ttl,
            )
        return self._llm_strategy_client, self._llm_params_client

    def _mandatory_llm_unavailable_error(self, model_name: str, role: str) -> RuntimeError:
        """Build a consistent mandatory-LLM unavailability error."""
        ollama_bin = self.config.get('llm', {}).get('ollama_bin', 'ollama')
        return RuntimeError(
            f"Local {role} LLM ({model_name}) is unavailable. "
            f"First verify Ollama CLI ('{ollama_bin}') is installed and on PATH "
            f"(e.g., `which {ollama_bin}`), then ensure model availability with: "
            f"{ollama_bin} pull {model_name}"
        )

    def _llm_generate_strategy(self, archive_summary: Dict[str, Any]) -> str:
        """Generate a high-level exploration strategy using the strategy LLM."""
        if (self.evo_cfg.get('llm_params_mode', 'llm') == 'random'
                or self.evo_cfg.get('use_llm_params', True) is False):
            strategy = "Randomized NSGA-II search (LLM strategy disabled)"
            self._last_llm_strategy = strategy
            return strategy

        strategy_client, _ = self._get_llm_clients()
        model_name = getattr(strategy_client, 'model', 'unknown')
        if not strategy_client.is_available():
            raise self._mandatory_llm_unavailable_error(model_name, "strategy")

        from llm.prompts_strategy import build_strategy_prompt, parse_strategy_response

        generation = int(archive_summary.get('generation', 0))
        prompt = build_strategy_prompt(
            archive_summary=archive_summary,
            generation=generation,
            scenario_name=self.scenario_name,
            hv_history=self.hv_history,
        )

        logger.info("Querying strategy LLM (%s) for generation %d", model_name, generation)
        response = strategy_client.generate_with_retry(prompt)
        strategy = parse_strategy_response(response)
        self._last_llm_strategy = strategy
        logger.info("STRATEGY: %s", strategy)
        return strategy or "Balanced exploration of energy-time tradeoff."

    def _llm_generate_params(self, strategy: str, n_params: int) -> List[MPCParams]:
        """Generate MPC parameter candidates using the params LLM."""
        from mpc.mpc_params import random_params
        if (self.evo_cfg.get('llm_params_mode', 'llm') == 'random'
                or self.evo_cfg.get('use_llm_params', True) is False):
            return [random_params(self.rng) for _ in range(n_params)]

        _, params_client = self._get_llm_clients()
        model_name = getattr(params_client, 'model', 'unknown')
        if not params_client.is_available():
            raise self._mandatory_llm_unavailable_error(model_name, "params")

        from llm.prompts_params import build_params_prompt, parse_params_response, validate_and_clamp
        from mpc.mpc_params import PARAM_BOUNDS, D_SAFE_MAX

        hard_constraints = [
            f"d_safe MUST be >= 0.26 m AND < {D_SAFE_MAX:.3f} m (robot collision radius = 0.25m, gate clearance is 0.617m so 0.26m is physically safe)",

            "N_base must be integer in [8, 18]",

        ]
        prompt = build_params_prompt(
            strategy=strategy,
            n_params=n_params,
            bounds=PARAM_BOUNDS,
            hard_constraints=hard_constraints,
        )

        logger.info("Querying params LLM (%s) for %d candidates", model_name, n_params)
        response = params_client.generate_with_retry(prompt)
        parsed = parse_params_response(response)
        self._last_llm_params = parsed
        logger.info("PARAMS: %s", parsed)

        candidates: List[MPCParams] = []
        for idx, entry in enumerate(parsed):
            if not isinstance(entry, dict):
                continue
            try:
                candidates.append(validate_and_clamp(entry))
            except (ValueError, KeyError, TypeError) as e:
                logger.debug(
                    "Skipping invalid LLM params entry at index %d (type=%s): %s",
                    idx, type(entry).__name__, e
                )
                continue

        if len(candidates) < n_params:
            logger.warning(
                "Only %d/%d LLM params parsed; filling the rest with random valid params.",
                len(candidates), n_params
            )
        while len(candidates) < n_params:
            candidates.append(random_params(self.rng))

        return candidates[:n_params]

    def _evolve_generation(self, population: List[MPCParams], objectives: np.ndarray) -> List[MPCParams]:
        """Evolve population using selection, crossover, mutation, and diversity filtering."""
        if not population:
            return []

        new_population: List[MPCParams] = []
        max_attempts = max(self.pop_size * 20, 100)
        attempts = 0

        while len(new_population) < self.pop_size and attempts < max_attempts:
            parent1 = tournament_selection(population, objectives, rng=self.rng)
            parent2 = tournament_selection(population, objectives, rng=self.rng)

            child1, child2 = sbx_crossover(parent1, parent2, eta=self.sbx_eta, rng=self.rng)
            child1 = polynomial_mutation(child1, eta=self.mutation_eta, rng=self.rng)
            child2 = polynomial_mutation(child2, eta=self.mutation_eta, rng=self.rng)

            for child in (child1, child2):
                if len(new_population) >= self.pop_size:
                    break
                if diversity_filter(child, new_population, threshold=self.cos_threshold):
                    new_population.append(child)

            attempts += 1

        while len(new_population) < self.pop_size:
            parent = tournament_selection(population, objectives, rng=self.rng)
            child = polynomial_mutation(parent, eta=self.mutation_eta, rng=self.rng)
            new_population.append(child)

        return new_population

    def _generation_dir(self, generation: int) -> str:
        return os.path.join(self.results_dir, 'evolution', self.scenario_name, f'gen{generation}')

    def _summary_csv_path(self) -> str:
        return os.path.join(self.results_dir, 'evolution', self.scenario_name, 'summary.csv')

    def _write_json(self, path: str, payload: Dict[str, Any]) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, default=float)

    def _eval_result_to_dict(self, eval_result: Any) -> Dict[str, Any]:
        if eval_result is None:
            return {}
        if isinstance(eval_result, dict):
            return eval_result
        try:
            return asdict(eval_result)
        except TypeError as e:
            logger.debug("Failed to serialize eval_result with asdict(): %s", e)
            return {}

    def _append_summary_row(
        self,
        generation: int,
        index: int,
        params: MPCParams,
        objectives: np.ndarray,
        eval_result: Any,
        elapsed_sec: float,
    ) -> None:
        base_dir = os.path.join(self.results_dir, 'evolution', self.scenario_name)
        os.makedirs(base_dir, exist_ok=True)
        path = self._summary_csv_path()

        eval_dict = self._eval_result_to_dict(eval_result)
        params_dict = params.to_dict()
        trial_results = eval_dict.get('trial_results', [])

        def _extract_trial_field(field: str):
            return [
                tr.get(field, [])
                for tr in trial_results
                if isinstance(tr, dict)
            ]

        trajectories = _extract_trial_field('trajectory')
        horizon_histories = _extract_trial_field('adaptive_horizon_history')

        row = {
            'scenario': self.scenario_name,
            'generation': generation,
            'candidate_index': index,
            'energy': float(objectives[0]),
            'time': float(objectives[1]),
            'solver_ms': float(objectives[2]),
            'failure_rate': float(objectives[3]),
            'energy_mean': float(eval_dict.get('energy_mean', objectives[0])),
            'energy_std': float(eval_dict.get('energy_std', 0.0)),
            'time_mean': float(eval_dict.get('time_mean', objectives[1])),
            'time_std': float(eval_dict.get('time_std', 0.0)),
            'solver_ms_mean': float(eval_dict.get('solver_ms_mean', objectives[2])),
            'solver_ms_std': float(eval_dict.get('solver_ms_std', 0.0)),
            'success_rate': float(eval_dict.get('success_rate', 1.0 - objectives[3])),
            'path_distance_mean': float(eval_dict.get('path_distance_mean', 0.0)),
            'path_distance_std': float(eval_dict.get('path_distance_std', 0.0)),
            'n_trials': int(eval_dict.get('n_trials', 0)),
            'solver_status_counts': json.dumps(eval_dict.get('solver_status_counts', {})),
            'trajectories': json.dumps(trajectories),
            'adaptive_horizon_histories': json.dumps(horizon_histories),
            'elapsed_sec': float(elapsed_sec),
        }
        row.update(params_dict)

        fieldnames = list(row.keys())
        write_header = not os.path.exists(path)
        with open(path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _llm_csv_path(self) -> str:
        filename = f"{self.scenario_name.upper()}_LLM.csv"
        return os.path.join(self.results_dir, "evolution", self.scenario_name, filename)

    def _append_llm_row(self, generation: int, strategy: Optional[str], params: Optional[List[Dict[str, Any]]]) -> None:
        base_dir = os.path.join(self.results_dir, "evolution", self.scenario_name)
        os.makedirs(base_dir, exist_ok=True)

        path = self._llm_csv_path()
        write_header = not os.path.exists(path)

        # Convert params list to JSON string so it stays one cell
        params_str = json.dumps(params, ensure_ascii=False)

        with open(path, "a", encoding='utf-8', newline="") as f:
            writer = csv.writer(f)
            if write_header:
                # top title row + header row
                writer.writerow([self.scenario_name.upper()])
                writer.writerow(["gen", "Strategy", "Params"])
            writer.writerow([generation, strategy or "", params_str])
    

    def _evaluate_candidate(self, params: MPCParams, scenario_name: str) -> Tuple[Dict[str, Any], np.ndarray]:
        """Evaluate a candidate, returning full EvalResult + 4D objective vector."""
        evaluator = self._get_evaluator()
        try:
            n_trials = self.config.get('evaluation', {}).get('n_trials', 50)
            result = evaluator.evaluate(params, scenario_name, n_trials=n_trials)
            return result, result.objectives
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            logger.warning(f"Evaluation failed: {e}")
            fallback = {
                'energy_mean': 5000.0,
                'energy_std': 0.0,
                'time_mean': 30.0,
                'time_std': 0.0,
                'solver_ms_mean': 800.0,
                'solver_ms_std': 0.0,
                'success_rate': 0.0,
                'path_distance_mean': 0.0,
                'path_distance_std': 0.0,
                'n_trials': 0,
                'trial_results': [],
                'solver_status_counts': {},
            }
            return fallback, np.array([5000.0, 30.0, 800.0, 1.0])

    def _save_candidate_result(
        self,
        generation: int,
        index: int,
        params: MPCParams,
        objectives: np.ndarray,
        eval_result: Any,
        elapsed_sec: float,
    ) -> None:
        gen_dir = self._generation_dir(generation)
        candidates_dir = os.path.join(gen_dir, 'candidates')
        os.makedirs(candidates_dir, exist_ok=True)
        payload = {
            'scenario': self.scenario_name,
            'generation': generation,
            'candidate_index': index,
            'params': params.to_dict(),
            'objectives': objectives.tolist(),
            'eval_result': self._eval_result_to_dict(eval_result),
            'elapsed_sec': float(elapsed_sec),
        }
        path = os.path.join(candidates_dir, f'can{index}.json')
        self._write_json(path, payload)
        self._append_summary_row(generation, index, params, objectives, eval_result, elapsed_sec)

    def _save_checkpoint(self, generation: int, population: List[MPCParams], archive: ParetoArchive, objectives: Optional[np.ndarray] = None, out_dir: Optional[str] = None):
        """Save generation checkpoint to results/."""
        out_dir = out_dir or os.path.join(self.results_dir, 'evolution', self.scenario_name)
        os.makedirs(out_dir, exist_ok=True)
        
        checkpoint = {
            'generation': generation,
            'hv_history': self.hv_history,
            'archive_size': archive.size(),
            'archive_summary': archive.get_summary(),
            'archive': [
                {
                    'params': e.params.to_dict(),
                    'objectives': e.objectives.tolist(),
                    'generation': e.generation,
                }
                for e in archive.get_pareto_front()
            ],
            'population': [p.to_dict() for p in population],
            'objectives': objectives.tolist() if objectives is not None else [],
        }

        path = os.path.join(out_dir, f'checkpoint_gen{generation:03d}.json')
        self._write_json(path, checkpoint)
        
        # Save Pareto front
        front_entries = archive.get_pareto_front()
        front_data = [
            {
                'params': e.params.to_dict(),
                'objectives': e.objectives.tolist(),
                'generation': e.generation
            }
            for e in front_entries
        ]
        front_path = os.path.join(out_dir, f'pareto_gen{generation:03d}.json')
        self._write_json(front_path, front_data)
        return checkpoint

    def _log_generation(self, gen: int, hv: float, best_energy: float, best_time: float):
        """Log generation statistics to console."""
        trend = self.stagnation_detector.trend()
        logger.info("[Gen %3d] HV=%.4f | BestE=%.1fJ | BestT=%.1fs | Archive=%d | %s",
                    gen, hv, best_energy, best_time, self.archive.size(), trend)

    def load_checkpoint(self, checkpoint_path: str, fresh_history: bool = False):
        """Load checkpoint and restore population + hv history + archive."""
        with open(checkpoint_path, "r", encoding='utf-8') as f:
            ckpt = json.load(f)
        population = [MPCParams.from_dict(p) for p in ckpt.get("population", [])]
        if not fresh_history:
            self.hv_history = ckpt.get("hv_history", [])
        else:
            self.hv_history = []

        pareto_path = str(checkpoint_path).replace("checkpoint_gen", "pareto_gen")
        if os.path.exists(pareto_path):
            try:
                with open(pareto_path, "r", encoding='utf-8') as f:
                    entries = json.load(f)
                for e in entries:
                    params = MPCParams.from_dict(e["params"])
                    objectives = np.array(e["objectives"], dtype=float)
                    gen = int(e.get("generation", ckpt.get("generation", 0)))
                    self.archive.update(params, objectives, generation=gen)
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.warning(f"Failed to load pareto archive: {e}")

        start_gen = int(ckpt.get("generation", 0)) + 1
        saved_objectives = ckpt.get("objectives", [])
        
        objectives = None  # None signals "no valid objectives saved"
        if saved_objectives:
            arr = np.array(saved_objectives, dtype=float)
            if np.any(arr != 0):  # only accept if at least one non-zero value
                objectives = arr
        return start_gen, population, objectives

   
    def run(
        self,
        n_generations: int = 20,
        pop_size: Optional[int] = None,
        initial_population: Optional[List[MPCParams]] = None,
        start_generation: int = 0,
        initial_objectives: Optional[np.ndarray] = None,
        eval_hook=None,     #added for distribution setup

    ) -> ParetoArchive:
        """Main evolutionary loop.
        
        Args:
            n_generations: number of generations
            pop_size: population size (overrides config)
            initial_population: optional seed population
        
        Returns:
            Final ParetoArchive
        """
        if pop_size is not None:
            self.pop_size = pop_size
        
        # Initialize population
        if initial_population is not None:
            population = initial_population[:self.pop_size]
        else:
            strat_pop = StratifiedPopulation(seed=42)
            strat_pop.initialize(self.pop_size)
            population = strat_pop.get_population()
        

        # Keep objectives (if provided) for LLM evolution, but never skip evaluation
        if initial_objectives is not None and initial_objectives.shape == (len(population), 4):
            objectives = initial_objectives
        else:
            objectives = np.zeros((len(population), 4))


        gen_iter = range(start_generation, n_generations)
        if tqdm is not None and self.use_tqdm:
            gen_iter = tqdm(gen_iter, desc=f"Evolution ({self.scenario_name})")

                
        for gen in gen_iter:
            logger.info(f"Generation {gen}/{n_generations}")
            gen_dir = self._generation_dir(gen)
            os.makedirs(gen_dir, exist_ok=True)

            # ---- LLM guidance BEFORE evaluation (gen > 0 only) ----
            if gen > 0:
                archive_summary = self.archive.get_summary()
                archive_summary['generation'] = gen - 1

                strategy = self._llm_generate_strategy(archive_summary)
                n_llm = max(2, self.pop_size // 4)
                llm_params = self._llm_generate_params(strategy, n_llm)
                self._append_llm_row(gen - 1, self._last_llm_strategy, self._last_llm_params)

                success_mask = objectives[:, 3] < 0.5
                if np.any(success_mask):
                    success_pop = [population[i] for i in range(len(population)) if success_mask[i]]
                    success_obj = objectives[success_mask]
                    population = self._evolve_generation(success_pop, success_obj)
                else:
                    population = self._evolve_generation(population, objectives)

                for j, lp in enumerate(llm_params):
                    if j < len(population):
                        population[-(j + 1)] = lp

                objectives = np.zeros((len(population), 4))

            # ---- Evaluate population (always) ----
            if eval_hook:
                # eval_hook must return list of dicts with "objectives" field
                results = eval_hook(population, gen)
                for i, result in enumerate(results):
                    params = population[i]
                    eval_result = result
                    objectives[i] = np.array(result["objectives"])
                    self.archive.update(params, objectives[i], generation=gen)
                    elapsed = 0.0
                    self._save_candidate_result(gen, i, params, objectives[i], eval_result, elapsed)
            else:
                cand_iter = enumerate(population)
                if tqdm is not None and self.use_tqdm:
                    cand_iter = tqdm(cand_iter, total=len(population), leave=False,
                                    desc=f"Gen {gen} candidates")
                for i, params in cand_iter:
                    cand_start = time.time()
                    logger.info(f"Gen {gen} | Candidate {i+1}/{len(population)} | start")
                    eval_result, objectives[i] = self._evaluate_candidate(params, self.scenario_name)
                    self.archive.update(params, objectives[i], generation=gen)
                    elapsed = time.time() - cand_start
                    self._save_candidate_result(gen, i, params, objectives[i], eval_result, elapsed)
                    logger.info(f"Gen {gen} | Candidate {i+1}/{len(population)} | done in {elapsed:.1f}s")

                           
            # Compute HV
            hv = self.archive.get_hypervolume()
            self.hv_history.append(hv)
            self.stagnation_detector.update(hv)
            
            # Best objectives — only from successful candidates to avoid
            # misleading console output from crashed/failed evaluations
            success_mask = objectives[:, 3] < 0.1
            if np.any(success_mask):
                best_energy = float(np.min(objectives[success_mask, 0]))
                best_time   = float(np.min(objectives[success_mask, 1]))
            else:
                best_energy = float(np.min(objectives[:, 0]))
                best_time   = float(np.min(objectives[:, 1]))
            self._log_generation(gen, hv, best_energy, best_time)

            # Save per-generation artifacts (every generation)
            checkpoint = self._save_checkpoint(gen, population, self.archive, objectives=objectives, out_dir=gen_dir)

            # Only write latest_checkpoint.json on final generation to prevent
            # shorter child runs from overwriting the main evolution's checkpoint
            if gen == n_generations - 1:
                base_dir = os.path.join(self.results_dir, 'evolution', self.scenario_name)
                latest_path = os.path.join(base_dir, 'latest_checkpoint.json')
                self._write_json(latest_path, checkpoint)

            # Save root checkpoints + Pareto like before (every 5 gens or final)
            if gen % 5 == 0 or gen == n_generations - 1:
                self._save_checkpoint(gen, population, self.archive, objectives=objectives)
            
            # Stagnation check
            if self.stagnation_detector.is_stagnating():
                logger.info("Stagnation detected - injecting %d new candidates",
                            self.injection_count)
                injected = self.injection_strategy.inject(
                    self.archive, self.pop_size, n_inject=self.injection_count
                )
                # Replace the worst solutions ranked by summed objective values.
                n_replace = min(len(injected), len(population))
                worst_indices = np.argsort(np.sum(objectives, axis=1))[::-1][:n_replace]
                # Build (abs_index, new_params) pairs for clean iteration
                replacements = [
                    (int(worst_indices[k]), injected[k])
                    for k in range(n_replace)
                ]
                # Update population slots (same regardless of eval path)
                for worst_idx, new_p in replacements:
                    population[worst_idx] = new_p

                if eval_hook:
                    # Distributed: batch-evaluate all injected candidates at once
                    inj_candidates = [new_p for _, new_p in replacements]
                    inj_results = eval_hook(inj_candidates, gen)
                    for (worst_idx, new_p), result in zip(replacements, inj_results):
                        objectives[worst_idx] = np.array(result["objectives"])
                        # FIX C2: injected candidates must enter the archive so HV
                        # reflects their contribution and the LLM prompt is informed.
                        self.archive.update(
                            new_p, objectives[worst_idx], generation=gen,
                            metadata={'source': 'stagnation_injection'}
                        )
                        self._save_candidate_result(
                            gen, worst_idx, new_p, objectives[worst_idx], result, 0.0
                        )
                else:
                    # Local: evaluate one by one (original behaviour)
                    for worst_idx, new_p in replacements:
                        cand_start = time.time()
                        eval_result, objectives[worst_idx] = self._evaluate_candidate(
                            new_p, self.scenario_name
                        )
                        elapsed = time.time() - cand_start
                        # FIX C2: injected candidates must enter the archive so HV
                        # reflects their contribution and the LLM prompt is informed.
                        self.archive.update(
                            new_p, objectives[worst_idx], generation=gen,
                            metadata={'source': 'stagnation_injection'}
                        )
                        self._save_candidate_result(
                            gen, worst_idx, new_p, objectives[worst_idx], eval_result, elapsed
                        )

                logger.info("Injection complete. Archive size: %d", self.archive.size())
                self.stagnation_detector.reset()
        
        return self.archive