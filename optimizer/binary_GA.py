import numpy as np
import yaml
import random
import copy

class BinaryGA:
    """
    Implementation of a Binary Genetic Algorithm (Benchmark).
    Reference: IMPROVE algorithm mentioned in the paper.
    
    Features:
    - Fixed-length binary chromosome encoding.
    - 16-bit discretization per continuous variable.
    - Tournament Selection.
    - Single-point Crossover.
    - Bit-flip Mutation.
    - Elitism mechanism to preserve the best solution.
    """

    def __init__(self, ga_config_path, constraints_config_path):
        """
        Initializes the Genetic Algorithm with configuration parameters and design constraints.

        Args:
            ga_config_path (str): Path to 'config/ACO_params.yaml' (specifically the 'binary_ga' section).
            constraints_config_path (str): Path to 'config/missile_constraints.yaml'.
        """
        # 1. Load Configuration
        with open(ga_config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.ga_params = config['binary_ga']
            
        with open(constraints_config_path, 'r') as f:
            constraints = yaml.safe_load(f)['design_variables']

        # 2. GA Parameters
        self.pop_size = self.ga_params['population_size']
        self.mutation_rate = self.ga_params['mutation_rate']
        self.crossover_rate = self.ga_params['crossover_rate']
        self.bits_per_param = self.ga_params['bits_per_param']
        self.elitism = self.ga_params['elitism']

        # 3. Variable Parsing (Genotype -> Phenotype mapping)
        self.var_names = list(constraints.keys())
        self.n_vars = len(self.var_names)
        self.bounds = constraints
        
        # Total chromosome length (sum of bits for every variable)
        self.chromosome_length = self.n_vars * self.bits_per_param
        
        # 4. Internal State
        # Population: Matrix (pop_size, chromosome_length) of integers 0/1
        self.population = None
        self.fitnesses = None
        self.best_individual = None
        self.best_fitness = float('inf')

    def initialize(self, evaluator_callback):
        """
        Initializes the population with random bitstrings and evaluates them.
        
        Args:
            evaluator_callback (callable): A function that takes a dictionary of parameters
                                           and returns a scalar fitness value.
        
        Side Effects:
            - Populates self.population with random binary data.
            - Evaluates initial fitnesses.
            - Updates self.best_individual.
        """
        # Generate random bits (0 or 1)
        self.population = np.random.randint(2, size=(self.pop_size, self.chromosome_length))
        self.fitnesses = np.zeros(self.pop_size)
        
        # Initial Evaluation
        for i in range(self.pop_size):
            phenotype = self.decode(self.population[i])
            self.fitnesses[i] = evaluator_callback(phenotype)
            
        self._update_best()

    def decode(self, chromosome):
        """
        Decodes a binary chromosome (genotype) into a dictionary of physical parameters (phenotype).
        Maps the binary integer value to the continuous range [min, max].

        Args:
            chromosome (np.array): 1D array of 0s and 1s.

        Returns:
            dict: Dictionary where keys are parameter names and values are floats.
        """
        params = {}
        
        for i, var_name in enumerate(self.var_names):
            # Extract the bit slice corresponding to the variable
            start = i * self.bits_per_param
            end = start + self.bits_per_param
            bits = chromosome[start:end]
            
            # Binary to Integer conversion
            # Example: [1, 0, 1] -> 5
            # Using dot product with powers of two for efficiency
            powers_of_two = 1 << np.arange(self.bits_per_param)[::-1]
            int_val = bits.dot(powers_of_two)
            
            # Integer to Float Mapping (Normalization)
            # Value = Min + (Int / (2^N - 1)) * (Max - Min)
            var_min = self.bounds[var_name]['min']
            var_max = self.bounds[var_name]['max']
            max_int = (1 << self.bits_per_param) - 1
            
            normalized_val = int_val / max_int
            real_val = var_min + normalized_val * (var_max - var_min)
            
            params[var_name] = real_val
            
        return params

    def _update_best(self):
        """Updates the globally best individual found so far."""
        min_idx = np.argmin(self.fitnesses)
        current_best_fit = self.fitnesses[min_idx]
        
        if current_best_fit < self.best_fitness:
            self.best_fitness = current_best_fit
            self.best_individual = self.population[min_idx].copy()

    def _tournament_selection(self, k=3):
        """
        Selects a parent using Tournament Selection.
        
        Args:
            k (int): Tournament size (default 3).
        
        Returns:
            np.array: The chromosome of the winner.
        """
        indices = np.random.choice(self.pop_size, k, replace=False)
        competitors_fitness = self.fitnesses[indices]
        winner_idx = indices[np.argmin(competitors_fitness)]
        return self.population[winner_idx]

    def optimize_step(self, evaluator_callback):
        """
        Executes ONE generation of the Genetic Algorithm.
        Steps: Selection -> Crossover -> Mutation -> Evaluation.
        
        Args:
            evaluator_callback (callable): Function to evaluate phenotype fitness.

        Returns:
            float: The best fitness value of the new generation.
        """
        new_population = []
        start_idx = 0
        
        # 1. Elitism: Copy the best individual directly to the new generation
        if self.elitism and self.best_individual is not None:
            new_population.append(self.best_individual.copy())
            start_idx = 1
            
        # 2. Reproduction Loop
        while len(new_population) < self.pop_size:
            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.crossover_rate:
                # Single Point Crossover
                point = random.randint(1, self.chromosome_length - 1)
                child1 = np.concatenate([parent1[:point], parent2[point:]])
                child2 = np.concatenate([parent2[:point], parent1[point:]])
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            for child in [child1, child2]:
                if len(new_population) < self.pop_size:
                    # Vectorized Bit-flip mutation
                    mutation_mask = np.random.random(self.chromosome_length) < self.mutation_rate
                    # Logical XOR: if mask is True, 0->1 and 1->0
                    child = np.logical_xor(child, mutation_mask).astype(int)
                    new_population.append(child)

        self.population = np.array(new_population)
        
        # 3. Evaluation of New Generation
        # Note: We re-evaluate the elite individual to ensure consistency if the fitness function is noisy,
        # though logically it retains the previous fitness.
        for i in range(start_idx, self.pop_size):
            phenotype = self.decode(self.population[i])
            self.fitnesses[i] = evaluator_callback(phenotype)
            
        # If elitism is active, ensure the first slot retains the best known fitness
        if self.elitism:
            self.fitnesses[0] = self.best_fitness 

        self._update_best()
        
        return self.best_fitness

    def get_best_solution(self):
        """
        Retrieves the best solution found so far.
        
        Returns:
            tuple: (dictionary of best parameters, best fitness value)
        """
        best_params = self.decode(self.best_individual)
        return best_params, self.best_fitness