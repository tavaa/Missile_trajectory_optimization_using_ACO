import yaml
import numpy as np
import datetime
from rocketpy import Environment, Flight

class TrajectoryEvaluator:
    """
    Evaluator class responsible for running flight simulations and calculating
    the fitness score of a candidate design based on specific targets.

    It acts as a bridge between the optimization algorithms (abstract parameters)
    and the physics engine (RocketPy). It enforces environmental conditions
    and implements the cost functions described in the reference paper.
    """

    def __init__(self, config_path, rail_length=6.0):
        """
        Initializes the evaluator, loads optimization targets, and configures
        the simulation environment.

        Args:
            config_path (str): File path to 'missile_constraints.yaml'.
            rail_length (float): Launch rail length [m] (default 6.0m).
        """
        self.config = self._load_config(config_path)
        self.rail_length = rail_length
        
        # Load Ground Truth Targets from YAML configuration
        self.single_target = self.config['single_trajectory_target']
        self.triple_scenarios = self.config['multi_trajectory_scenarios']
        
        # --- ENVIRONMENT CONFIGURATION ---
        # Sets up the Spaceport America location and weather conditions.
        self.env = Environment(
            latitude=32.990254, 
            longitude=-106.974998, 
            elevation=1400 # ASL [m]
        )
        
        # Set date to tomorrow at noon for consistent solar position (if needed)
        tomorrow = datetime.date.today() + datetime.timedelta(days=1)
        self.env.set_date((tomorrow.year, tomorrow.month, tomorrow.day, 12))
        
        # Define Atmospheric Model with Crosswind
        # Note: A constant 5 m/s crosswind is applied to ensure robust optimization
        # against weathercocking effects, as per project requirements.
        self.env.set_atmospheric_model(
            type="standard_atmosphere", 
            wind_u=0.0, 
            wind_v=5.0 # 5 m/s Crosswind (assuming launch heading 90 deg)
        )

    def _load_config(self, path):
        """Helper to load the YAML configuration file."""
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _simulate_flight(self, rocket, launch_angle):
        """
        Executes the 6-DOF flight simulation using RocketPy.
        Includes safety checks for stability and launch safety.

        Args:
            rocket (rocketpy.Rocket): The rocket object to fly.
            launch_angle (float): Launch rail inclination [degrees].

        Returns:
            dict: Flight metrics (range, apogee, flight_time, burn_time).
            None: If the simulation crashes or safety checks fail.
        """
        try:
            flight = Flight(
                rocket=rocket, 
                environment=self.env, 
                rail_length=self.rail_length, 
                inclination=launch_angle, 
                heading=90, # Launch towards East
                terminate_on_apogee=False, # Simulate full flight until ground impact
                verbose=False
            )
            
            # --- Safety & Validity Checks ---
            
            # Check 1: Simulation convergence (valid apogee)
            if flight.apogee == 0 or np.isnan(flight.apogee): 
                return None
            
            # Check 2: Rail Departure Velocity
            # Minimum 15 m/s required for aerodynamic stability off the rail
            if flight.out_of_rail_velocity < 15.0: 
                return None 
            
            # Check 3: Static Stability Margin at Launch
            # Must be positive (usually > 1.0 cal) to avoid tumbling
            if flight.static_margin(0) < 0:
                return None

            # --- Data Extraction ---
            
            # Calculate Scalar Range (Euclidean distance from launchpad to impact)
            impact_x = flight.x_impact
            impact_y = flight.y_impact
            flight_range = np.sqrt(impact_x**2 + impact_y**2)
            
            # Extract Burn Time directly from the motor configuration
            # (flight.rocket.motor.burn_time is a tuple (start, end))
            tbo = rocket.motor.burn_time[1] 
            
            return {
                'range': flight_range,
                'apogee': flight.apogee - self.env.elevation, # Altitude AGL
                'flight_time': flight.t_final,
                'burn_time': tbo 
            }
            
        except Exception:
            # Catch integration errors (e.g., unstable rocket causing infinite loops)
            return None

    def _calculate_fitness_eq3(self, sim_data, target_data):
        """
        Calculates fitness using Equation 3 from the Kiyak et al. paper.
        Formula: Fitness = |dR|/10 + |dA|/10 + |dTOF|/1 + |dTBO|/1
        
        Args:
            sim_data (dict): Simulated metrics.
            target_data (dict): Target metrics (Ground Truth).

        Returns:
            float: Calculated weighted error score.
        """
        # Maximum penalty for failed simulations
        if sim_data is None:
            return 10000.0

        # Weights defined in the paper
        w_range = 10.0
        w_apogee = 10.0
        w_tof = 1.0
        w_tbo = 1.0

        # Absolute Errors
        err_range = abs(sim_data['range'] - target_data['range'])
        err_apogee = abs(sim_data['apogee'] - target_data['apogee'])
        err_tof = abs(sim_data['flight_time'] - target_data['flight_time'])
        err_tbo = abs(sim_data['burn_time'] - target_data['burn_time'])

        # Weighted Sum
        fitness = (err_range / w_range) + \
                  (err_apogee / w_apogee) + \
                  (err_tof / w_tof) + \
                  (err_tbo / w_tbo)
                  
        return fitness

    def evaluate(self, simulation_results, task_type='single'):
        """
        Main entry point to calculate the total fitness score.
        
        Args:
            simulation_results: 
                - dict: Single flight data (if task_type='single').
                - dict of dicts: Multiple flight data (if task_type='triple').
                - None: Indicates a crash/failure.
            task_type (str): 'single' or 'triple'.
            
        Returns:
            float: Total fitness score (Lower is better).
        """
        # Handle global simulation failure
        if simulation_results is None:
            return 10000.0

        if task_type == 'single':
            # --- TASK A: Single Trajectory (Eq. 3) ---
            return self._calculate_fitness_eq3(simulation_results, self.single_target)

        elif task_type == 'triple':
            # --- TASK B: Triple Trajectory (Eq. 4) ---
            # Sum of fitness scores from three distinct scenarios
            total_fitness = 0.0
            
            for scenario_key, scenario_config in self.triple_scenarios.items():
                # Extract specific simulation result for this scenario
                sim_data = simulation_results.get(scenario_key)
                # Extract specific target for this scenario
                target_data = scenario_config['target']
                
                total_fitness += self._calculate_fitness_eq3(sim_data, target_data)
                
            return total_fitness
        
        else:
            raise ValueError(f"Task type '{task_type}' not recognized.")