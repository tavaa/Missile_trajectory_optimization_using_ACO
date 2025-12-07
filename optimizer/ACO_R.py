import numpy as np
import yaml
import random
import copy

class ContinuousACO:
    """
    Implementation of ACO-R (Ant Colony Optimization for Continuous Domains).
    Based on the algorithm described by Socha & Dorigo, adapted by Kiyak et al.
    
    Key Features:
    - Solution Archive (Pheromone Model equivalent).
    - Gaussian Kernel Probability Density Function for sampling.
    - Ranking-based weight assignment (Eq. 1).
    - Modified Elitism (q_mod) parameter (Paper contribution).
    """

    def __init__(self, aco_config_path, constraints_config_path):
        """
        Initializes the ACO algorithm with configuration parameters and design constraints.

        Args:
            aco_config_path (str): Path to 'config/ACO_params.yaml'.
            constraints_config_path (str): Path to 'config/missile_constraints.yaml'.
        """
        # 1. Load Configuration
        with open(aco_config_path, 'r') as f:
            self.aco_params = yaml.safe_load(f)['aco']
            
        with open(constraints_config_path, 'r') as f:
            constraints = yaml.safe_load(f)['design_variables']

        # 2. Parse Algorithm Parameters
        self.k = self.aco_params['archive_size']      # Archive Size
        self.m = self.aco_params['new_solutions']     # Ants per iteration
        self.q = self.aco_params['q_weight']          # Gaussian weight param
        self.zeta = self.aco_params['zeta']           # Convergence speed
        self.q_mod = self.aco_params['q_mod_elitism'] # Extra elitism prob

        # 3. Parse Design Variables (Bounds)
        self.var_names = list(constraints.keys())
        self.n_vars = len(self.var_names)
        
        # Numpy arrays for fast vectorized operations
        self.min_bounds = np.array([constraints[name]['min'] for name in self.var_names])
        self.max_bounds = np.array([constraints[name]['max'] for name in self.var_names])
        
        # 4. Internal State
        # Archive X: Solutions matrix (k, n_vars)
        # Archive F: Fitness vector (k,)
        self.archive_X = None
        self.archive_F = None
        
        # Pre-compute rank weights (Eq. 1)
        self._precompute_weights()

    def _precompute_weights(self):
        """
        Calculates weights w_j for each rank j in the archive.
        Formula: w_j = (1 / (q*k*sqrt(2pi))) * exp( ... )
        """
        ranks = np.arange(1, self.k + 1) # Rank 1 to k
        term1 = 1.0 / (self.q * self.k * np.sqrt(2 * np.pi))
        term2 = -((ranks - 1) ** 2) / (2 * (self.q ** 2) * (self.k ** 2))
        
        w = term1 * np.exp(term2)
        
        # Normalize to create a probability distribution
        self.probabilities = w / np.sum(w)

    def initialize(self, evaluator_callback):
        """
        Fills the initial archive with random uniform solutions.
        
        Args:
            evaluator_callback (callable): Function mapping params_dict -> fitness.
        """
        # Random generation within bounds
        random_pop = np.random.uniform(0, 1, (self.k, self.n_vars))
        self.archive_X = self.min_bounds + random_pop * (self.max_bounds - self.min_bounds)
        self.archive_F = np.zeros(self.k)

        # Initial Evaluation
        for i in range(self.k):
            params = dict(zip(self.var_names, self.archive_X[i]))
            fitness = evaluator_callback(params)
            self.archive_F[i] = fitness
            
        self._sort_archive()

    def _sort_archive(self):
        """Sorts the archive based on fitness (Ascending: Lower is better)."""
        sorted_indices = np.argsort(self.archive_F)
        self.archive_X = self.archive_X[sorted_indices]
        self.archive_F = self.archive_F[sorted_indices]

    def _select_guide_index(self):
        """
        Selects a guide solution index from the archive.
        Implements the q_mod elitism modification from the paper.
        """
        # 1. Extra Elitism
        if random.random() < self.q_mod:
            return 0 # Force selection of the best solution
        
        # 2. Roulette Wheel Selection based on Gaussian weights
        return np.random.choice(self.k, p=self.probabilities)

    def _calculate_sigma(self, guide_index):
        """
        Calculates standard deviation for sampling.
        Eq. 2: sigma = zeta * sum(|x_r - x_l|) / (k-1)
        """
        guide_solution = self.archive_X[guide_index]
        
        # Manhattan distance sum between guide and all other solutions
        distances = np.abs(self.archive_X - guide_solution)
        sigma = self.zeta * np.sum(distances, axis=0) / (self.k - 1)
        
        # Prevent zero division / collapse
        sigma = np.maximum(sigma, 1e-9) 
        
        return sigma

    def optimize_step(self, evaluator_callback):
        """
        Executes ONE iteration step (generates 'm' ants, evaluates, updates archive).
        
        Returns:
            float: Current best fitness in the archive.
        """
        
        new_solutions_X = []
        new_solutions_F = []
        
        # --- Construction Phase ---
        for _ in range(self.m):
            # 1. Select Guide
            l = self._select_guide_index()
            
            # 2. Calculate PDF Parameters
            mu = self.archive_X[l]
            sigma = self._calculate_sigma(l)
            
            # 3. Sample New Solution
            new_sol = np.random.normal(mu, sigma, self.n_vars)
            
            # 4. Enforce Bounds (Clipping)
            new_sol = np.clip(new_sol, self.min_bounds, self.max_bounds)
            
            # 5. Evaluate
            params_dict = dict(zip(self.var_names, new_sol))
            fitness = evaluator_callback(params_dict)
            
            new_solutions_X.append(new_sol)
            new_solutions_F.append(fitness)

        # --- Pheromone Update Phase ---
        # Append new solutions
        self.archive_X = np.vstack([self.archive_X, np.array(new_solutions_X)])
        self.archive_F = np.concatenate([self.archive_F, np.array(new_solutions_F)])
        
        # Sort
        self._sort_archive()
        
        # Prune (Keep only best k)
        self.archive_X = self.archive_X[:self.k]
        self.archive_F = self.archive_F[:self.k]
        
        return self.archive_F[0]

    def get_best_solution(self):
        """Returns the parameters dict and fitness of the best solution."""
        best_params = dict(zip(self.var_names, self.archive_X[0]))
        best_fitness = self.archive_F[0]
        return best_params, best_fitness

    def reset_archive(self, keep_best=True):
        """
        Implements the 'Mini-Cycle' restart strategy.
        Keeps the best solution and re-initializes the rest of the archive randomly.
        """
        best_X = self.archive_X[0].copy()
        best_F = self.archive_F[0]
        
        # Generate new random population
        random_pop = np.random.uniform(0, 1, (self.k - 1, self.n_vars))
        new_archive_part = self.min_bounds + random_pop * (self.max_bounds - self.min_bounds)
        
        # Reconstruct archive
        self.archive_X[0] = best_X
        self.archive_X[1:] = new_archive_part
        
        # Set fitness to infinity to force re-evaluation or displacement
        self.archive_F[0] = best_F
        self.archive_F[1:] = np.inf