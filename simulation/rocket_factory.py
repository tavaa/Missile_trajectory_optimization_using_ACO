import numpy as np
import math
from rocketpy import Rocket, SolidMotor, NoseCone, TrapezoidalFins

class RocketFactory:
    """
    A factory class to create rocket models based on given parameters.
    """

    def __init__(self):
        """
        Initializes the RocketFactory with fixed physical constants and material properties.
        """
        # ==========================================
        # PHYSICAL CONSTANTS AND MATERIALS (FIXED)
        # ==========================================
        self.rail_length = 6.0            # Launch rail length [m]
        self.structural_density = 2000.0  # Structural density [kg/m^3] (e.g., Fiberglass/Aluminum)
        self.wall_thickness = 0.003       # Wall thickness [m]
        
        # Chemical/Propulsion Parameters
        self.propellant_isp = 210.0       # Specific Impulse [s] (Standard APCP)
        self.g0 = 9.81                    # Standard gravity [m/s^2]
        self.motor_dry_mass_fraction = 0.3 # Ratio of casing mass to propellant mass

    def generate_thrust_curve(self, total_impulse, burn_time):
        """
        Generates a trapezoidal thrust curve (ramp-up -> plateau -> tail-off).
        Ensures that the area under the curve exactly matches 'total_impulse'.

        Args:
            total_impulse (float): Total impulse of the motor [Ns].
            burn_time (float): Total burn time of the motor [s].

        Returns:
            list: A list of (time, thrust) tuples representing the thrust curve.
        """
        # Define time phases
        t_ignition = 0.05 * burn_time   # 5% time to reach pressure
        t_tail_start = 0.85 * burn_time # 85% start of thrust tail-off
        t_end = burn_time + 0.05        # 50ms extra for complete burnout
        
        # Time points
        times = [0, t_ignition, t_tail_start, burn_time, t_end]
        
        # Shape profile (Unitary)
        # 0 -> 1 (Ignition) -> 1 (Steady) -> 0 (Burnout)
        thrust_shape = [0, 1.0, 1.0, 0.0, 0.0]
        
        # Calculate integral (Area) of the base shape (manual trapezoidal method)
        area_1 = 0.5 * t_ignition
        area_2 = t_tail_start - t_ignition
        area_3 = 0.5 * (burn_time - t_tail_start)
        
        shape_total_area = area_1 + area_2 + area_3
        
        # Scale factor to achieve target impulse
        scale_factor = total_impulse / shape_total_area
        
        # Final scaled curve
        thrust_values = [v * scale_factor for v in thrust_shape]
        
        # Output in RocketPy format [(t, F), ...]
        return list(zip(times, thrust_values))

    def calculate_component_inertia(self, mass, radius, length, shape="hollow_cylinder"):
        """
        Calculates local moments of inertia (I_long, I_trans) relative to the
        component's center of mass.

        Args:
            mass (float): Mass of the component [kg].
            radius (float): Radius of the component [m].
            length (float): Length of the component [m].
            shape (str): Shape type ('hollow_cylinder', 'solid_cylinder', 'cone_shell').

        Returns:
            tuple: (I_long, I_trans) representing longitudinal and transverse inertia.
        """
        I_long = 0.0  # Z Axis (Roll)
        I_trans = 0.0 # X/Y Axis (Pitch/Yaw)

        if shape == "hollow_cylinder":
            # Fuselage or Empty Motor Casing
            I_long = mass * radius**2
            # I_x = m * (r^2/2 + h^2/12)
            I_trans = mass * (radius**2 / 2 + length**2 / 12)
            
        elif shape == "solid_cylinder":
            # Ballast or Filled Motor
            I_long = 0.5 * mass * radius**2
            I_trans = (mass / 12.0) * (3 * radius**2 + length**2)
            
        elif shape == "cone_shell":
            # Hollow Nose Cone
            I_long = 0.5 * mass * radius**2
            I_trans = mass * (radius**2 / 4 + length**2 / 6) # Standard approx
            
        return I_long, I_trans

    def create_rocket(self, params):
        """
        Assembles the complete rocket by calculating masses, inertia, drag, and motor
        based on optimization parameters.

        Args:
            params (dict): Dictionary containing rocket design parameters.

        Returns:
            rocketpy.Rocket: An instance of the Rocket class configured with the given parameters.
        """
        
        # 1. EXTRACT GEOMETRIC PARAMETERS
        r = params['radius']
        L_nose = params['nose_length']
        L_body = params['body_length']
        L_total = L_nose + L_body
        
        # Ballast (default 0 if not present in optimization)
        m_ballast = params.get('nose_ballast', 0.0)

        # ==========================================
        # 2. MOTOR GENERATION (SYNTHETIC)
        # ==========================================
        # Calculate motor masses
        propellant_mass = params['total_impulse'] / (self.propellant_isp * self.g0)
        motor_dry_mass = propellant_mass * self.motor_dry_mass_fraction
        
        # Generate Dynamic Thrust Curve
        thrust_source = self.generate_thrust_curve(params['total_impulse'], params['burn_time'])
        
        # Grain Geometry (Fictional for internal inertia simulation)
        grain_len = L_body * 0.4 
        
        # Inertia of empty motor casing (hollow cylinder)
        I_long_mot, I_trans_mot = self.calculate_component_inertia(
            motor_dry_mass, r*0.4, grain_len, "hollow_cylinder"
        )

        motor = SolidMotor(
            thrust_source=thrust_source,
            burn_time=params['burn_time'] + 0.05,
            grain_number=1,
            grain_density=1700, # Typical solid propellant density
            grain_outer_radius=r - self.wall_thickness - 0.001,
            grain_initial_inner_radius=r * 0.3,
            grain_initial_height=grain_len,
            grain_separation=0,                         # Required for RocketPy v1+
            grains_center_of_mass_position=grain_len/2, # Required for RocketPy v1+
            nozzle_radius=r * 0.35,
            throat_radius=r * 0.15,
            interpolation_method='linear',
            coordinate_system_orientation="nozzle_to_combustion_chamber",
            dry_mass=motor_dry_mass,
            dry_inertia=(I_trans_mot, I_trans_mot, I_long_mot),
            center_of_dry_mass_position=grain_len/2,
            nozzle_position=0
        )

        # ==========================================
        # 3. ANALYTICAL INERTIA CALCULATION (AIRFRAME)
        # ==========================================
        # Accumulator lists: (mass, cg_z_position, I_long_local, I_trans_local)
        components = []

        # A. Nose Cone
        area_nose = np.pi * r * np.sqrt(r**2 + L_nose**2)
        m_nose = area_nose * self.wall_thickness * self.structural_density
        cg_nose = L_nose * 0.666 # Approx CG conical surface (2/3 height)
        Il_n, It_n = self.calculate_component_inertia(m_nose, r, L_nose, "cone_shell")
        components.append((m_nose, cg_nose, Il_n, It_n))

        # B. Ballast - Positioned near the tip
        cg_bal = L_nose * 0.2
        Il_b, It_b = self.calculate_component_inertia(m_ballast, r*0.8, 0.05, "solid_cylinder")
        components.append((m_ballast, cg_bal, Il_b, It_b))

        # C. Body Tube
        area_body = 2 * np.pi * r * L_body
        m_body = area_body * self.wall_thickness * self.structural_density
        cg_body = L_nose + (L_body * 0.5)
        Il_bd, It_bd = self.calculate_component_inertia(m_body, r, L_body, "hollow_cylinder")
        components.append((m_body, cg_body, Il_bd, It_bd))

        # D. Fins
        fin_area = (params['fin_root_chord'] + params['fin_tip_chord']) / 2 * params['fin_span']
        m_fins = (fin_area * 0.005 * 1800) * 4 # 4 fins, 5mm thickness, composite material
        
        # Root attachment Z position
        fin_root_z = L_total - params['fin_distance_from_base'] - params['fin_root_chord']
        cg_fins_local_x = params['fin_root_chord'] * 0.5 # Simplified
        cg_fins_global = fin_root_z + cg_fins_local_x
        
        # Fin Inertia (Approx)
        r_gyr = r + params['fin_span']/3 # Average gyration radius
        Il_f = m_fins * r_gyr**2         # Rotating mass at distance r_gyr
        It_f = (m_fins / 12) * params['fin_root_chord']**2 # Approx flat plate
        components.append((m_fins, cg_fins_global, Il_f, It_f))

        # E. Payload/Avionics (Fixed)
        m_pay = 1.0 
        cg_pay = L_nose + 0.2
        Il_p, It_p = self.calculate_component_inertia(m_pay, r*0.8, 0.15, "solid_cylinder")
        components.append((m_pay, cg_pay, Il_p, It_p))

        # --- Component Aggregation (Parallel Axis Theorem) ---
        total_mass = 0.0
        moment_sum = 0.0
        
        # 1. Find Total Center of Mass
        for m, pos, _, _ in components:
            total_mass += m
            moment_sum += m * pos
        
        cg_rocket = moment_sum / total_mass
        
        I_long_tot = 0.0
        I_trans_tot = 0.0
        
        # 2. Shift Inertias
        for m, pos, Il, It in components:
            dist = pos - cg_rocket
            # Z Axis (Roll): Direct sum (coincident axes)
            I_long_tot += Il
            # X/Y Axis (Pitch/Yaw): Huygens-Steiner
            I_trans_tot += It + m * dist**2

        # ==========================================
        # 4. DYNAMIC DRAG CALCULATION (Skin Friction)
        # ==========================================
        # Cd_body = Cf * (S_wet / S_ref)
        S_wet_body = 2 * np.pi * r * L_body
        S_ref = np.pi * r**2
        Cf_approx = 0.0045  # Standard turbulent friction coeff
        
        Cd_body_friction = Cf_approx * (S_wet_body / S_ref)
        Cd_parasitic = 0.02 # Screws, rail buttons, imperfections
        
        total_base_drag = Cd_body_friction + Cd_parasitic

        # ==========================================
        # 5. ROCKET INSTANTIATION
        # ==========================================
        rocket = Rocket(
            radius=r,
            mass=total_mass,
            inertia=(I_trans_tot, I_trans_tot, I_long_tot),
            center_of_mass_without_motor=cg_rocket,
            coordinate_system_orientation="nose_to_tail",
            power_off_drag=total_base_drag, # Calculated dynamic drag
            power_on_drag=total_base_drag
        )

        # ==========================================
        # 6. ADD SURFACES
        # ==========================================
        rocket.add_motor(motor, position=L_total)

        nose_cone = NoseCone(
            length=L_nose,
            kind="von karman",
            base_radius=r,
            rocket_radius=r
        )
        rocket.add_surfaces(surfaces=nose_cone, positions=0)

        # Geometric Safety Check for Fins
        safe_tip = min(params['fin_tip_chord'], params['fin_root_chord'])
        # Calculate physical sweep length
        sweep_len = params['fin_span'] * math.tan(math.radians(params['fin_sweep_angle']))
        
        fins = TrapezoidalFins(
            n=4,
            root_chord=params['fin_root_chord'],
            tip_chord=safe_tip,
            span=params['fin_span'],
            sweep_length=sweep_len,
            cant_angle=0.0,
            rocket_radius=r
        )
        
        rocket.add_surfaces(surfaces=fins, positions=fin_root_z)

        # Rail Buttons
        rocket.set_rail_buttons(
            upper_button_position=cg_body - 0.2,
            lower_button_position=L_total - 0.1,
            angular_position=45
        )

        return rocket

    def get_launch_parameters(self, params):
        """
        Returns the launch parameters to be passed to the Flight object.

        Args:
            params (dict): Dictionary containing rocket design parameters.

        Returns:
            dict: Dictionary with 'launch_angle' and 'rail_length'.
        """
        return {
            "launch_angle": params['launch_angle'],
            "rail_length": self.rail_length
        }