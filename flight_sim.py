# flight_sim.py
import pygame
import math
import sys
import random
import time
import os
# Import necessary classes/functions
from parts import draw_part_shape, get_part_data
from rocket_data import RocketBlueprint, PlacedPart, AMBIENT_TEMPERATURE
from ui_elements import SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, BLACK, GRAY, RED, GREEN, BLUE, LIGHT_GRAY
# Define colors if not in ui_elements
try: from ui_elements import COLOR_SKY_BLUE, COLOR_SPACE_BLACK, COLOR_HORIZON, COLOR_GROUND
except ImportError: COLOR_SKY_BLUE, COLOR_SPACE_BLACK, COLOR_HORIZON, COLOR_GROUND = (135, 206, 250), (0,0,0), (170, 210, 230), (0, 150, 0)
try: from ui_elements import COLOR_FLAME, COLOR_UI_BAR, COLOR_UI_BAR_BG, COLOR_EXPLOSION
except ImportError: COLOR_FLAME, COLOR_UI_BAR, COLOR_UI_BAR_BG, COLOR_EXPLOSION = (255,100,0), (0,200,0), (50,50,50), [(255,255,0),(255,150,0),(200,50,0),(150,150,150)]
try: from ui_elements import COLOR_ENGINE_ENABLED, COLOR_ENGINE_DISABLED, COLOR_ACTIVATABLE_READY, COLOR_ACTIVATABLE_USED
except ImportError: COLOR_ENGINE_ENABLED, COLOR_ENGINE_DISABLED, COLOR_ACTIVATABLE_READY, COLOR_ACTIVATABLE_USED = GREEN, RED, BLUE, GRAY

# --- Flight Sim Constants ---
GRAVITY = 9.81 * 6 # Slightly stronger gravity for faster gameplay
ROTATION_SPEED = 200 # Degrees per second with max control input
REACTION_WHEEL_TORQUE = 10000 # Torque provided by reaction wheels (Nm?)
ANGULAR_DAMPING = 0.3 # Slows down rotation over time
COLLISION_DAMAGE_FACTOR = 0.7 # Multiplier for impact damage calculation
MIN_IMPACT_VEL_DAMAGE = 38 # Minimum velocity (m/s) for collision damage
THROTTLE_CHANGE_RATE = 0.5 # How fast throttle changes (0 to 1 per second)
GROUND_Y = 1000 # World Y coordinate of the ground surface
WORLD_WIDTH = 5000 # Width of the playable world (for background effects)
BLUE_SKY_Y_LIMIT = -2000 # World Y where sky starts transitioning to space
SPACE_Y_LIMIT = -15000 # World Y where sky is fully space black
STAR_COUNT = 20000 # Number of stars for parallax background
STAR_FIELD_DEPTH = 10000 # Max depth for star parallax effect

# --- Air Density & Atmosphere ---
AIR_DENSITY_SEA_LEVEL = 1.225 # kg/m^3 at ground level
AIR_DENSITY_VACUUM = 0.0 # kg/m^3 in vacuum
ATMOSPHERE_SCALE_HEIGHT = 8500.0 # For exponential decay, in meters
ATMOSPHERE_EXP_LIMIT = 35000.0 # Altitude where exponential decay transitions to linear (m)
ATMOSPHERE_LINEAR_LIMIT = 70000.0 # Altitude where atmosphere ends (linear region) (m)
ATMOSPHERE_TARGET_DENSITY_FACTOR = 0 # Target density at linear limit (1% of sea level)

# --- Thermal Physics ---
HEAT_DISSIPATION_FACTOR_VACUUM = 0.01 # Base cooling rate in vacuum
HEAT_DISSIPATION_FACTOR_CONVECTION = 0.1 # Cooling added by air density
AERO_HEATING_FACTOR = 2e-7 # Factor for calculating heat from drag/velocity
OVERHEAT_DAMAGE_THRESHOLD_K = 50 # Kelvin over max_temp before damage starts
OVERHEAT_DAMAGE_RATE = 100 # HP damage per second when fully overheated
REENTRY_EFFECT_THRESHOLD_TEMP = 700.0 # Kelvin when reentry glow starts
REENTRY_EFFECT_MAX_TEMP_SCALE = 1.1 # Visual glow scales up to max_temp * this factor
REENTRY_PARTICLE_COLOR_START = pygame.Color(150, 0, 0)
REENTRY_PARTICLE_COLOR_END = pygame.Color(255, 255, 150)
REENTRY_PARTICLE_LIFETIME = 0.6 # seconds
REENTRY_PARTICLE_SPEED = 80 # pixels/sec

COLOR_EXPLOSION = [pygame.Color(c) for c in [(255,255,0),(255,150,0),(200,50,0), GRAY]]

# --- Particle System Classes ---
class EffectParticle:
    def __init__(self, pos, vel, life, start_color, end_color=None, start_radius=3, end_radius=0):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.life = life
        self.max_life = life
        self.start_color = pygame.Color(start_color)
        self.end_color = pygame.Color(end_color) if end_color else self.start_color
        self.start_radius = start_radius
        self.end_radius = end_radius
        self.current_radius = start_radius
        self.current_color = self.start_color

    def update(self, dt):
        self.life -= dt
        if self.life > 0:
            self.pos += self.vel * dt
            self.vel *= 0.97 # Simple drag
            # Interpolate color and radius
            t = max(0.0, min(1.0, 1.0 - (self.life / self.max_life))) # Lerp factor (0 -> 1)
            try:
                self.current_color = self.start_color.lerp(self.end_color, t)
            except ValueError: # Handle potential issues with Color lerp if colors are identical?
                self.current_color = self.start_color
            self.current_radius = self.start_radius + (self.end_radius - self.start_radius) * t
            return True # Still alive
        return False # Expired

    def draw(self, surface, camera):
        if self.life > 0 and self.current_radius >= 0.5:
            screen_pos = camera.apply(self.pos)
            # Basic culling
            if -self.current_radius < screen_pos.x < SCREEN_WIDTH + self.current_radius and \
               -self.current_radius < screen_pos.y < SCREEN_HEIGHT + self.current_radius:
                 try:
                     # Convert Pygame Color object to tuple for drawing funcs
                     draw_color = (self.current_color.r, self.current_color.g, self.current_color.b)
                     pygame.draw.circle(surface, draw_color, screen_pos, int(self.current_radius))
                 except ValueError: pass # Ignore potential errors from invalid color tuples during lerp extremes

class ParticleManager:
    def __init__(self):
        self.particles: list[EffectParticle] = []

    def add_explosion(self, pos, num_particles=15, max_life=0.5, max_speed=100, colors=COLOR_EXPLOSION):
        for _ in range(num_particles):
            angle = random.uniform(0, 360)
            speed = random.uniform(max_speed * 0.2, max_speed)
            vel = pygame.math.Vector2(speed, 0).rotate(angle)
            life = random.uniform(max_life * 0.3, max_life)
            color = random.choice(colors)
            radius = random.uniform(2, 5)
            self.particles.append(EffectParticle(pos, vel, life, color, start_radius=radius, end_radius=0))

    def add_reentry_spark(self, pos, base_vel, intensity_factor):
        # Sparks shoot roughly opposite to velocity, with some spread
        angle_offset = random.uniform(-25, 25)
        spark_dir = -base_vel.normalize() if base_vel.length_sq() > 0 else pygame.math.Vector2(0, 1) # Default down if no vel
        spark_vel = spark_dir.rotate(angle_offset) * REENTRY_PARTICLE_SPEED * (0.5 + intensity_factor) + base_vel * 0.1 # Inherit some base vel
        life = REENTRY_PARTICLE_LIFETIME * (0.7 + random.random() * 0.6) # Vary lifetime
        start_rad = 1 + 3 * intensity_factor
        end_rad = 0
        # Color depends on intensity (hotter = brighter yellow/white)
        start_col = REENTRY_PARTICLE_COLOR_START.lerp(REENTRY_PARTICLE_COLOR_END, intensity_factor * 0.8)
        end_col = REENTRY_PARTICLE_COLOR_START # Fade back to red
        self.particles.append(EffectParticle(pos, spark_vel, life, start_col, end_col, start_rad, end_rad))

    def update(self, dt):
        self.particles = [p for p in self.particles if p.update(dt)] # Efficiently remove dead particles

    def draw(self, surface, camera):
        for p in self.particles:
            p.draw(surface, camera)

# --- Camera Class ---
class Camera:
    def __init__(self, width, height):
        self.camera_rect = pygame.Rect(0, 0, width, height)
        self.width = width
        self.height = height
        self.offset = pygame.math.Vector2(0, 0)

    def apply(self, target_pos):
        """ Apply camera offset to a world position to get screen position """
        return target_pos - self.offset

    def apply_rect(self, target_rect):
        """ Apply camera offset to a world Rect """
        return target_rect.move(-self.offset.x, -self.offset.y)

    def update(self, target_pos):
        """ Center the camera on the target world position """
        x = target_pos.x - self.width // 2
        y = target_pos.y - self.height // 2
        self.offset = pygame.math.Vector2(x, y)

# --- FlyingRocket Class ---
class FlyingRocket:
    def __init__(self, parts_list: list[PlacedPart], initial_pos_offset: pygame.math.Vector2, initial_angle=0, initial_vel=pygame.math.Vector2(0,0), sim_instance_id=0, is_primary_control=False, original_root_ref=None):
        self.sim_instance_id = sim_instance_id
        self.parts = parts_list
        if not self.parts:
            raise ValueError("Cannot initialize FlyingRocket with an empty parts list.")
        self.blueprint_name = f"Rocket_{sim_instance_id}" # For debugging
        self.original_root_part_ref = original_root_ref # Keep reference to the original root part instance
        self.has_active_control = is_primary_control # Does this assembly have the controllable part?

        # Physics State
        self.pos = pygame.math.Vector2(0, 0) # World position of the assembly's blueprint origin (0,0) point
        self.vel = pygame.math.Vector2(initial_vel)
        self.acc = pygame.math.Vector2(0, 0)
        self.angle = initial_angle # Degrees, 0 = pointing up
        self.angular_velocity = 0.0 # Degrees per second

        # Control State
        self.throttle_level = 0.0 # 0.0 to 1.0
        self.master_thrust_enabled = False # Master switch for engines

        # Component References & State
        self.engines = []
        self.fuel_tanks = []
        self.parachutes = []
        self.separators = []
        total_fuel_cap_this_assembly = 0
        for i, part in enumerate(self.parts):
            # Initialize runtime part state
            part.current_hp = part.part_data.get("max_hp", 100)
            part.is_broken = False
            part.engine_enabled = True # Default to enabled
            part.deployed = False # For parachutes etc.
            part.separated = False # For separators
            part.current_temp = AMBIENT_TEMPERATURE # Start at ambient
            part.is_overheating = False # For visuals
            part.part_index = i # Useful for debugging?

            pt = part.part_data.get("type")
            if pt == "Engine": self.engines.append(part)
            elif pt == "FuelTank":
                self.fuel_tanks.append(part)
                total_fuel_cap_this_assembly += part.part_data.get("fuel_capacity", 0)
            elif pt == "Parachute": self.parachutes.append(part)
            elif pt == "Separator": self.separators.append(part)

        # Resources
        self.current_fuel = total_fuel_cap_this_assembly # Start full
        self.fuel_mass_per_unit = 0.1 # kg per unit of fuel (adjust as needed)

        # Calculated Physics Properties (recalculated as needed)
        self.total_mass = 0.01 # kg, avoid division by zero
        self.dry_mass = 0.0 # kg, mass without fuel
        self.moment_of_inertia = 10000 # kg*m^2, placeholder
        self.center_of_mass_offset = pygame.math.Vector2(0, 0) # Offset from blueprint origin (0,0) IN LOCAL ROCKET COORDINATES
        self.local_bounds = pygame.Rect(0,0,1,1) # Local AABB around all parts relative to blueprint origin

        # Calculate initial physics properties and bounds
        self.calculate_physics_properties()
        self.calculate_bounds()

        # Set initial world position based on desired CoM start and calculated offset
        # Rotate the initial CoM offset to world space and subtract from target start position
        initial_com_offset_rotated = self.center_of_mass_offset.rotate(-self.angle)
        self.pos = initial_pos_offset - initial_com_offset_rotated # Place blueprint origin so CoM is at initial_pos_offset

        # Status Flags
        self.landed = False # Is the rocket resting on the ground?
        self.thrusting = False # Are any engines firing? (for visuals)
        self.is_active = True # Set to False if destroyed or empty
        self.pending_separation = [] # List of separators activated this frame
        self.needs_connectivity_check = False # Flag if parts were destroyed
        self.was_landed_last_frame = False # For landing impact detection
        self.max_temp_reading = AMBIENT_TEMPERATURE # Hottest temp on any part this frame (for UI)

    def calculate_physics_properties(self):
        """ Recalculates mass, center of mass (CoM), and moment of inertia (MoI). """
        total_m = 0.0
        com_numerator = pygame.math.Vector2(0, 0)
        moi_sum = 0.0 # Moment of Inertia calculation using Parallel Axis Theorem

        if not self.parts: # Handle empty rocket case
            self.total_mass = 0.01
            self.center_of_mass_offset = pygame.math.Vector2(0,0)
            self.moment_of_inertia = 1.0
            self.dry_mass = 0.0
            return

        # Calculate total fuel mass and how it's distributed
        fuel_mass_total = self.current_fuel * self.fuel_mass_per_unit
        total_tank_capacity = sum(p.part_data.get("fuel_capacity", 0) for p in self.fuel_tanks)
        total_tank_capacity = max(1.0, total_tank_capacity) # Avoid division by zero if no tanks

        self.dry_mass = sum(p.part_data.get("mass", 0) for p in self.parts)
        current_com_offset_local = self.center_of_mass_offset # Store old CoM for MoI calc

        # --- First Pass: Calculate Total Mass and CoM ---
        for part in self.parts:
            part_mass_static = part.part_data.get("mass", 0)
            part_fuel_mass = 0
            # Distribute fuel mass proportionally to tank capacity
            if part.part_data.get("type") == "FuelTank" and total_tank_capacity > 0:
                part_fuel_mass = fuel_mass_total * (part.part_data.get("fuel_capacity", 0) / total_tank_capacity)

            part_mass_current = part_mass_static + part_fuel_mass
            total_m += part_mass_current
            com_numerator += part.relative_pos * part_mass_current # Sum of (position * mass)

        self.total_mass = max(0.01, total_m) # Avoid zero mass
        # New Center of Mass (local coordinates relative to blueprint origin)
        self.center_of_mass_offset = com_numerator / self.total_mass if self.total_mass > 0.01 else pygame.math.Vector2(0, 0)

        # --- Second Pass: Calculate Moment of Inertia around the new CoM ---
        for part in self.parts:
             part_mass_static = part.part_data.get("mass", 0)
             part_fuel_mass = 0
             if part.part_data.get("type") == "FuelTank" and total_tank_capacity > 0:
                 part_fuel_mass = fuel_mass_total * (part.part_data.get("fuel_capacity", 0) / total_tank_capacity)
             part_mass_current = part_mass_static + part_fuel_mass

             # MoI of the part around its own center (approximated as rectangle)
             w = part.part_data.get("width", 1); h = part.part_data.get("height", 1)
             i_part = (1/12.0) * part_mass_current * (w**2 + h**2)

             # Parallel Axis Theorem: I = I_cm + m*d^2
             dist_vec = part.relative_pos - self.center_of_mass_offset # Vector from new CoM to part's center
             d_sq = dist_vec.length_squared() # Squared distance
             moi_sum += i_part + part_mass_current * d_sq

        self.moment_of_inertia = max(1.0, moi_sum) # Avoid zero MoI

    def calculate_bounds(self):
        """ Calculates the local bounding box containing all parts relative to blueprint origin. """
        if not self.parts:
            self.local_bounds = pygame.Rect(0, 0, 0, 0)
            return
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        for p in self.parts:
            half_w = p.part_data['width'] / 2.0
            half_h = p.part_data['height'] / 2.0
            center_x = p.relative_pos.x
            center_y = p.relative_pos.y
            # Simple AABB based on local position (doesn't account for rotation within assembly)
            min_x = min(min_x, center_x - half_w)
            max_x = max(max_x, center_x + half_w)
            min_y = min(min_y, center_y - half_h)
            max_y = max(max_y, center_y + half_h)
        if min_x == float('inf'): # Handle case where list might be empty after check
             self.local_bounds = pygame.Rect(0,0,0,0)
        else: self.local_bounds = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)

    def get_world_com(self):
        """ Calculates the current world position of the Center of Mass. """
        # Rotate the local CoM offset by the rocket's angle and add to the rocket's origin position
        com_offset_rotated = self.center_of_mass_offset.rotate(-self.angle)
        return self.pos + com_offset_rotated

    def get_world_part_center(self, part: PlacedPart):
        """ Calculates the world position of a specific part's center. """
        part_offset_rotated = part.relative_pos.rotate(-self.angle)
        return self.pos + part_offset_rotated

    def get_parts_near_world_pos(self, world_pos: pygame.math.Vector2, radius: float = 20.0):
        """ Returns a list of parts whose centers are within radius of world_pos. """
        nearby_parts = []
        radius_sq = radius * radius
        for part in self.parts:
             part_world_center = self.get_world_part_center(part)
             if (part_world_center - world_pos).length_squared() < radius_sq:
                 nearby_parts.append(part)
        return nearby_parts

    def get_lowest_point_world(self) -> pygame.math.Vector2:
        """ Finds the world coordinates of the point on the rocket with the largest Y value (lowest on screen). """
        if not self.parts: return self.pos # Return origin if no parts
        lowest_y = -float('inf')
        lowest_point_world = self.pos # Default to origin
        for part in self.parts:
             part_center_world = self.get_world_part_center(part)
             w = part.part_data['width']; h = part.part_data['height']
             part_world_angle = self.angle # Assembly's angle
             # Get corners relative to part center, rotate, then add world center
             corners_local = [pygame.math.Vector2(x,y) for x in [-w/2, w/2] for y in [-h/2, h/2]]
             corners_world = [part_center_world + corner.rotate(-part_world_angle) for corner in corners_local]
             for corner in corners_world:
                 if corner.y > lowest_y:
                     lowest_y = corner.y
                     lowest_point_world = corner
        # If somehow no valid corner found (e.g., all parts zero size?), return CoM
        return lowest_point_world if lowest_y > -float('inf') else self.get_world_com()

    def get_world_part_aabb(self, part: PlacedPart) -> pygame.Rect:
        """ Calculates an Axis-Aligned Bounding Box for a part in world coordinates (approximate). """
        # TODO: This doesn't account for rotation, making collision inaccurate.
        # A more accurate method would use Separating Axis Theorem (SAT) or OBBs.
        part_data = part.part_data; w = part_data.get('width', 1); h = part_data.get('height', 1)
        world_center = self.get_world_part_center(part)
        # Simple AABB approximation
        aabb = pygame.Rect(0, 0, w, h)
        aabb.center = world_center
        return aabb


    def consume_fuel(self, amount):
        """ Consumes fuel, updates physics properties, returns True if successful. """
        if amount <= 0: return True # No consumption needed
        consumed = min(self.current_fuel, amount)
        if consumed > 0:
            self.current_fuel -= consumed
            self.calculate_physics_properties() # Recalculate mass, CoM, MoI
            return True
        return False # Out of fuel

    def get_thrust_data(self) -> tuple[pygame.math.Vector2, pygame.math.Vector2 | None, float]:
        """ Calculates total thrust force vector, average application point offset (local rotated), and total fuel consumption rate at 100% throttle. """
        total_thrust_potential = 0.0
        thrust_torque_numerator = pygame.math.Vector2(0, 0) # Sum of (local_pos * thrust_magnitude) for weighted average
        total_consumption_at_100_throttle = 0.0
        active_engine_count = 0

        if not self.master_thrust_enabled or self.throttle_level <= 0 or self.current_fuel <= 0:
            return pygame.math.Vector2(0,0), None, 0.0

        for engine in self.engines:
            if engine.is_broken or not engine.engine_enabled:
                continue # Skip broken or disabled engines
            thrust_magnitude = engine.part_data.get("thrust", 0)
            consumption_rate = engine.part_data.get("fuel_consumption", 0)
            # Engine position relative to blueprint origin, rotated by current angle
            engine_pos_offset_local_rotated = engine.relative_pos.rotate(-self.angle)

            total_thrust_potential += thrust_magnitude
            thrust_torque_numerator += engine_pos_offset_local_rotated * thrust_magnitude
            total_consumption_at_100_throttle += consumption_rate
            active_engine_count += 1

        force_vector = pygame.math.Vector2(0, 0)
        avg_thrust_application_point_offset = None # Offset from the blueprint origin (self.pos)

        if active_engine_count > 0 and total_thrust_potential > 0:
            actual_thrust_magnitude = total_thrust_potential * self.throttle_level
            # Thrust direction is opposite to the rocket's 'up' direction (angle 0)
            thrust_direction = pygame.math.Vector2(0, -1).rotate(-self.angle)
            force_vector = thrust_direction * actual_thrust_magnitude
            # Calculate the average application point (weighted by thrust)
            avg_thrust_application_point_offset = thrust_torque_numerator / total_thrust_potential

        return force_vector, avg_thrust_application_point_offset, total_consumption_at_100_throttle

    def apply_collision_damage(self, impact_velocity_magnitude, particle_manager: ParticleManager, specific_part_to_damage: PlacedPart | None = None):
        """ Applies damage to parts based on impact velocity. Damages multiple parts based on proximity to impact unless a specific part is given."""
        if impact_velocity_magnitude < MIN_IMPACT_VEL_DAMAGE:
            return # Impact too soft

        base_damage = (impact_velocity_magnitude ** 1.8) * COLLISION_DAMAGE_FACTOR # Damage scales non-linearly with speed

        parts_to_damage = []
        damage_multipliers = {} # Store multiplier for each part

        if specific_part_to_damage and specific_part_to_damage in self.parts:
             # Damage only the specified part (e.g., rocket-rocket collision)
             parts_to_damage = [specific_part_to_damage]
             damage_multipliers[specific_part_to_damage] = 1.0
        elif not specific_part_to_damage and self.parts:
            # Ground collision: Damage parts near the lowest point
            lowest_world_y = self.get_lowest_point_world().y
            world_com_y = self.get_world_com().y
            # Estimate rocket height relevant to impact (from CoM to lowest point)
            rocket_impact_height = abs(lowest_world_y - world_com_y) * 1.5 # Factor to spread damage slightly higher
            rocket_impact_height = max(1.0, rocket_impact_height) # Avoid division by zero

            for part in self.parts:
                part_center_y = self.get_world_part_center(part).y
                # How close is the part to the bottom? (0 = at bottom, increases upwards)
                relative_y_from_bottom = lowest_world_y - part_center_y
                # Damage factor decreases quadratically with distance from bottom
                damage_factor = max(0.0, min(1.0, 1.0 - (relative_y_from_bottom / rocket_impact_height)))**2
                if damage_factor > 0.01: # Only apply if significant factor
                     parts_to_damage.append(part)
                     damage_multipliers[part] = damage_factor

            # If the above logic somehow selects no parts (e.g., rocket is flat), damage the lowest part
            if not parts_to_damage and self.parts:
                 lowest_part = min(self.parts, key=lambda p: self.get_world_part_center(p).y)
                 parts_to_damage = [lowest_part]
                 damage_multipliers[lowest_part] = 1.0

        parts_destroyed_this_impact = []
        for part in parts_to_damage:
            if part.is_broken: continue # Skip already broken parts
            multiplier = damage_multipliers.get(part, 0.0)
            scaled_damage = base_damage * multiplier
            if scaled_damage < 0.1: continue # Ignore negligible damage

            part.current_hp -= scaled_damage
            # print(f"  Part {part.part_id} took {scaled_damage:.1f} impact damage (HP: {part.current_hp:.1f})")
            if part.current_hp <= 0 and not part.is_broken:
                 print(f"  >> Part '{part.part_id}' BROKEN by impact! <<")
                 part.is_broken = True
                 part.current_hp = 0
                 parts_destroyed_this_impact.append(part)
                 # Create visual effect for broken part
                 particle_manager.add_explosion(self.get_world_part_center(part))

        # If parts were destroyed, handle the consequences
        if parts_destroyed_this_impact:
            self.handle_destroyed_parts(parts_destroyed_this_impact)

    def handle_destroyed_parts(self, destroyed_parts: list[PlacedPart]):
        """ Removes destroyed parts, checks for control loss, and flags for connectivity check. """
        if not destroyed_parts: return

        original_part_count = len(self.parts)
        self.parts = [p for p in self.parts if p not in destroyed_parts] # Remove destroyed parts

        # Check if the original controlling part was destroyed
        if self.original_root_part_ref and self.original_root_part_ref in destroyed_parts:
            self.has_active_control = False # Lost control if root part broke
            print(f"[{self.sim_instance_id}] Lost control: Root part destroyed.")

        if not self.parts:
            self.is_active = False # Rocket is gone if no parts left
            print(f"[{self.sim_instance_id}] Deactivated: All parts destroyed.")
            return

        # If parts were removed, update component lists and flag for split check
        if len(self.parts) < original_part_count:
            self.needs_connectivity_check = True # Possible structural break
            # Update component lists (could be done more efficiently)
            self.engines = [e for e in self.engines if e in self.parts]
            self.fuel_tanks = [t for t in self.fuel_tanks if t in self.parts]
            self.parachutes = [pc for pc in self.parachutes if pc in self.parts]
            self.separators = [s for s in self.separators if s in self.parts]
            # Recalculate physics properties for the remaining structure
            self.calculate_physics_properties()
            self.calculate_bounds()

    def activate_part_at_pos(self, click_world_pos):
        """ Toggles state of activatable parts (engines, parachutes, separators) near the click position. """
        clicked_part = None
        min_dist_sq = 20**2 # Max click distance (squared)
        # Check activatable parts only
        parts_to_check = self.engines + self.parachutes + self.separators
        for part in parts_to_check:
            if part.is_broken: continue
            dist_sq = (self.get_world_part_center(part) - click_world_pos).length_squared()
            if dist_sq < min_dist_sq:
                clicked_part = part
                min_dist_sq = dist_sq # Found a closer part

        if not clicked_part: return False # Didn't click near anything relevant
        if not clicked_part.part_data.get("activatable", False): return False # Clicked part isn't activatable

        part_type = clicked_part.part_data.get("type")
        action_taken = False

        if part_type == "Engine":
            clicked_part.engine_enabled = not clicked_part.engine_enabled
            print(f"Toggled Engine {clicked_part.part_id} {'ON' if clicked_part.engine_enabled else 'OFF'}")
            action_taken = True
        elif part_type == "Parachute":
            if not clicked_part.deployed: # Can only deploy once
                clicked_part.deployed = True
                print(f"Deployed Parachute {clicked_part.part_id}!")
                action_taken = True
        elif part_type == "Separator":
            if not clicked_part.separated: # Can only separate once
                 # Add to pending list to be processed at end of frame
                 if clicked_part not in self.pending_separation:
                     self.pending_separation.append(clicked_part)
                     clicked_part.separated = True # Mark as activated
                     print(f"Activated Separator {clicked_part.part_id}.")
                     action_taken = True

        return action_taken

    def update(self, dt, current_air_density, particle_manager: ParticleManager):
        """ Main physics and state update loop for the rocket. """
        if not self.is_active or not self.parts:
            return # Don't update inactive/empty rockets

        # --- Reset Forces/Torque ---
        self.acc = pygame.math.Vector2(0, 0)
        net_torque = 0.0
        current_world_com = self.get_world_com()
        velocity_sq = self.vel.length_squared()
        velocity_mag = math.sqrt(velocity_sq)

        # --- Gravity ---
        if self.total_mass > 0.01:
            gravity_force = pygame.math.Vector2(0, GRAVITY * self.total_mass)
            self.acc += gravity_force / self.total_mass

        # --- Thrust ---
        thrust_force, thrust_app_local_offset_rotated, cons_rate_100 = self.get_thrust_data()
        self.thrusting = False
        if thrust_force.length_squared() > 0:
            if self.consume_fuel(cons_rate_100 * self.throttle_level * dt):
                self.thrusting = True # Consumed fuel, thrust is active
                if self.total_mass > 0.01:
                    self.acc += thrust_force / self.total_mass
                # Apply torque from thrust if application point is offset from CoM
                if thrust_app_local_offset_rotated:
                    # Vector from CoM to thrust application point (in world frame)
                    thrust_app_offset_from_com = (self.pos + thrust_app_local_offset_rotated) - current_world_com
                    # Torque = r x F (cross product)
                    net_torque += thrust_app_offset_from_com.cross(thrust_force)

        # --- Aerodynamics (Drag) & Heating ---
        total_drag_force = pygame.math.Vector2(0, 0)
        parts_destroyed_by_heat = []
        current_max_temp = AMBIENT_TEMPERATURE # Track max temp this frame

        for part in self.parts:
            if part.is_broken: continue
            part_data = part.part_data
            part_world_center = self.get_world_part_center(part)
            drag_force_on_part = pygame.math.Vector2(0, 0)
            aero_force_applied = False
            effective_area = 0.0 # Placeholder for area calculation

            # --- Drag Calculation ---
            if current_air_density > AIR_DENSITY_VACUUM and velocity_sq > 0.1: # Only apply drag if moving in air
                # Simplified effective area (could be improved based on angle)
                effective_area = (part_data['width'] + part_data['height']) / 2.0 * 0.1 # Scale factor for size
                drag_coeff = part_data['base_drag_coeff']
                # Add parachute drag if deployed
                if part_data['type'] == 'Parachute' and part.deployed:
                    drag_coeff += part_data['deploy_drag']
                    effective_area += (part_data['width'] * part_data['deploy_area_factor']) * 0.1 # Use deploy area factor

                if effective_area > 0 and drag_coeff > 0:
                    drag_magnitude = 0.5 * current_air_density * velocity_sq * effective_area * drag_coeff
                    if velocity_mag > 0.01: # Avoid division by zero / normalize(0)
                         drag_force_on_part = -self.vel.normalize() * drag_magnitude
                         total_drag_force += drag_force_on_part
                         # Apply torque from drag
                         drag_app_offset_from_com = part_world_center - current_world_com
                         net_torque += drag_app_offset_from_com.cross(drag_force_on_part)
                         aero_force_applied = True

            # --- Aerodynamic Heating ---
            heat_generated = 0.0
            if effective_area > 0 and velocity_mag > 10: # Only heat if moving fast through air
                # Heating = Factor * Density * Velocity^3 * Area * dt
                heat_generated = AERO_HEATING_FACTOR * current_air_density * (velocity_mag**3) * effective_area * dt

            # --- Cooling ---
            temp_diff = part.current_temp - AMBIENT_TEMPERATURE # Difference from ambient
            # Cooling rate depends on air density (convection) + base radiation
            cooling_rate = HEAT_DISSIPATION_FACTOR_VACUUM + HEAT_DISSIPATION_FACTOR_CONVECTION * current_air_density
            # Heat lost depends on temp diff, cooling rate, time, and surface area (approximated)
            heat_lost = cooling_rate * temp_diff * dt * (part_data['width'] * part_data['height'] * 0.01) # Area scale factor

            # --- Temperature Change ---
            thermal_mass = part_data['thermal_mass'] # J/K
            if thermal_mass > 0:
                delta_temp = (heat_generated - heat_lost) / thermal_mass # K = J / (J/K)
                part.current_temp += delta_temp
                part.current_temp = max(AMBIENT_TEMPERATURE, part.current_temp) # Don't cool below ambient

            # --- Overheating Check & Damage ---
            max_temp = part_data['max_temp']
            part.is_overheating = part.current_temp > REENTRY_EFFECT_THRESHOLD_TEMP # For visual effect trigger
            if part.current_temp > max_temp:
                overheat_amount = part.current_temp - max_temp
                # Damage scales with how much over max temp (relative to threshold)
                damage_factor = max(0.0, overheat_amount / OVERHEAT_DAMAGE_THRESHOLD_K)
                damage = OVERHEAT_DAMAGE_RATE * damage_factor * dt
                part.current_hp -= damage
                # print(f"  Part {part.part_id} took {damage:.1f} heat damage (T={part.current_temp:.0f}K, HP={part.current_hp:.1f})")
                if part.current_hp <= 0 and not part.is_broken:
                    print(f"  >> Part '{part.part_id}' DESTROYED by overheating! (T={part.current_temp:.0f}K) <<")
                    part.is_broken = True
                    part.current_hp = 0
                    parts_destroyed_by_heat.append(part)
                    particle_manager.add_explosion(part_world_center)

            current_max_temp = max(current_max_temp, part.current_temp) # Update overall max temp

        self.max_temp_reading = current_max_temp # Store for UI

        # --- Apply Net Drag Force ---
        if self.total_mass > 0.01:
            self.acc += total_drag_force / self.total_mass

        # --- Handle Heat Destruction ---
        if parts_destroyed_by_heat:
            self.handle_destroyed_parts(parts_destroyed_by_heat)
            # Need to recalculate CoM if parts broke, as it affects torque application points
            if self.is_active:
                self.calculate_physics_properties() # Updates CoM
                self.calculate_bounds()
                current_world_com = self.get_world_com() # Get potentially new CoM

        # --- Control Input (Reaction Wheels) ---
        control_torque = 0.0
        # Check if the original root part still exists and is not broken
        root_ref = self.original_root_part_ref
        self.has_active_control = (root_ref is not None) and (root_ref in self.parts) and (not root_ref.is_broken)
        if self.has_active_control: # Only allow control if root part is ok
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                control_torque += REACTION_WHEEL_TORQUE
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                control_torque -= REACTION_WHEEL_TORQUE
        net_torque += control_torque

        # --- Integration ---
        # Update velocity based on acceleration
        self.vel += self.acc * dt
        # Update position based on velocity
        self.pos += self.vel * dt

        # Update angular velocity based on torque and MoI
        if self.moment_of_inertia > 0:
            angular_acceleration = math.degrees(net_torque / self.moment_of_inertia) # rad/s^2 -> deg/s^2
            self.angular_velocity += angular_acceleration * dt

        # Apply angular damping
        self.angular_velocity *= (1.0 - ANGULAR_DAMPING * dt)
        # Update angle based on angular velocity
        self.angle = (self.angle + self.angular_velocity * dt) % 360

        # --- Ground Collision / Landing ---
        currently_on_ground = self.get_lowest_point_world().y >= GROUND_Y
        just_landed = currently_on_ground and not self.was_landed_last_frame
        just_took_off = not currently_on_ground and self.was_landed_last_frame

        if just_landed:
            # Calculate impact velocity magnitude (vertical component is most important for ground)
            impact_vel_mag = self.vel.length()
            # Apply damage if impact is hard enough OR if scraping sideways fast
            # Allow slightly lower vertical velocity threshold if Y velocity is positive (moving down)
            should_apply_damage = (self.vel.y > 1.0 and impact_vel_mag >= MIN_IMPACT_VEL_DAMAGE * 0.5) or \
                                  (impact_vel_mag >= MIN_IMPACT_VEL_DAMAGE)
            if should_apply_damage:
                # Apply damage distributed near the bottom
                self.apply_collision_damage(impact_vel_mag, particle_manager, None)

        if currently_on_ground:
            self.landed = True
            if self.is_active and self.parts: # If still alive after potential impact damage
                # Correct position to prevent sinking below ground
                correction = self.get_lowest_point_world().y - GROUND_Y
                if correction > 0: self.pos.y -= correction
                # Stop vertical velocity, dampen horizontal velocity and rotation
                self.vel.y = 0
                self.vel.x *= 0.6 # Friction
                self.angular_velocity = 0
            else:
                # If destroyed on impact, just stop everything
                 self.vel = pygame.math.Vector2(0,0); self.angular_velocity = 0
        else:
            self.landed = False

        # Destroy deployed parachutes on takeoff (they get ripped off)
        if just_took_off:
            destroyed_chutes = []
            for chute in self.parachutes:
                if chute.deployed and not chute.is_broken:
                    print(f"  >> Parachute {chute.part_id} destroyed by takeoff! <<")
                    chute.is_broken = True
                    chute.deployed = False # No longer considered deployed
                    chute.current_hp = 0
                    destroyed_chutes.append(chute)
                    particle_manager.add_explosion(self.get_world_part_center(chute), num_particles=5, max_life=0.3, max_speed=50)
            # No need to call handle_destroyed_parts here unless you want immediate structural checks

        self.was_landed_last_frame = self.landed


    def draw(self, surface, camera, particle_manager: ParticleManager):
        """ Draws the rocket assembly and its effects. Returns number of visually broken parts. """
        num_broken_visually = 0
        if not self.is_active: return 0

        # --- Check for Reentry Effects ---
        spawn_reentry_particles = False
        hottest_part_temp = AMBIENT_TEMPERATURE
        for part in self.parts:
             if part.is_overheating and not part.is_broken:
                 spawn_reentry_particles = True
                 hottest_part_temp = max(hottest_part_temp, part.current_temp)

        # Spawn reentry sparks if needed
        if spawn_reentry_particles and self.vel.length() > 50**2: # Only if moving reasonably fast
             # Intensity based on how hot the hottest part is relative to threshold and max temp
             intensity = max(0.0, min(1.0, (hottest_part_temp - REENTRY_EFFECT_THRESHOLD_TEMP) /
                                            (max(1.0, self.max_temp_reading * REENTRY_EFFECT_MAX_TEMP_SCALE - REENTRY_EFFECT_THRESHOLD_TEMP))))
             # Spawn particles near the leading edge (lowest point)
             leading_point = self.get_lowest_point_world()
             num_sparks = int(1 + intensity * 5) # More sparks for higher intensity
             for _ in range(num_sparks):
                 # Add slight random offset to spawn position
                 spawn_offset = pygame.math.Vector2(random.uniform(-5, 5), random.uniform(-5, 5))
                 particle_manager.add_reentry_spark(leading_point + spawn_offset, self.vel, intensity)


        # --- Draw Parts ---
        for part in self.parts:
            part_center_world = self.get_world_part_center(part)
            part_screen_pos = camera.apply(part_center_world)
            part_world_angle = self.angle # All parts share the assembly's angle
            indicator_color = None # For activatable part status
            activatable = part.part_data.get("activatable", False)
            is_parachute = part.part_data.get("type") == "Parachute"
            show_deployed_visual = part.deployed and not part.is_broken
            heat_factor = 0.0 # Default no glow

            # Hide deployed chute visual if landed
            if is_parachute and self.landed:
                show_deployed_visual = False

            # Calculate heat glow factor (0 to 1)
            if part.is_overheating and not part.is_broken:
                 max_temp_visual = part.part_data['max_temp'] * REENTRY_EFFECT_MAX_TEMP_SCALE # Glow scales up slightly beyond max temp
                 heat_factor = max(0.0, min(1.0, (part.current_temp - REENTRY_EFFECT_THRESHOLD_TEMP) /
                                                max(1.0, max_temp_visual - REENTRY_EFFECT_THRESHOLD_TEMP)))

            # --- Draw Part Shape ---
            if part.is_broken: num_broken_visually += 1
            elif activatable: # Draw status indicator for activatable parts
                 if is_parachute: # Blue=Ready, Gray=Deployed/Used
                     indicator_color = COLOR_ACTIVATABLE_USED if part.deployed else COLOR_ACTIVATABLE_READY
                 elif part.part_data.get("type") == "Engine": # Green=Enabled, Red=Disabled
                     indicator_color = COLOR_ENGINE_ENABLED if part.engine_enabled else COLOR_ENGINE_DISABLED
                 elif part.part_data.get("type") == "Separator": # Blue=Ready, Gray=Separated/Used
                     indicator_color = COLOR_ACTIVATABLE_USED if part.separated else COLOR_ACTIVATABLE_READY

            try: # Use the drawing function from parts.py
                 draw_part_shape(surface, part.part_data, part_screen_pos, part_world_angle,
                                 broken=part.is_broken, deployed=show_deployed_visual, heat_factor=heat_factor)
            except NameError: # Fallback if draw_part_shape isn't available
                 pygame.draw.circle(surface, RED if part.is_broken else GREEN, part_screen_pos, 5)

            # Draw indicator dot if applicable
            if indicator_color:
                 pygame.draw.circle(surface, indicator_color, part_screen_pos, 5) # Draw on top

        # --- Draw Engine Flames ---
        if self.thrusting:
            flame_scale = 0.5 + 0.5 * self.throttle_level # Flame size based on throttle
            for engine in self.engines:
                # Only draw flame if engine is active and providing thrust
                if engine.engine_enabled and not engine.is_broken:
                    engine_center_world = self.get_world_part_center(engine)
                    engine_world_angle = self.angle
                    # Calculate flame base position (bottom center of engine)
                    flame_base_offset_local = pygame.math.Vector2(0, engine.part_data["height"] / 2.0)
                    flame_base_offset_rotated = flame_base_offset_local.rotate(-engine_world_angle)
                    flame_base_world = engine_center_world + flame_base_offset_rotated

                    # Calculate flame shape points
                    flame_length = (15 + random.uniform(-2, 2)) * flame_scale # Add flicker
                    flame_width = engine.part_data["width"] * 0.8 * flame_scale
                    flame_dir_world = pygame.math.Vector2(0, 1).rotate(-engine_world_angle) # Direction flame points
                    flame_side_world = pygame.math.Vector2(1, 0).rotate(-engine_world_angle) # Perpendicular to flame dir
                    flame_tip_world = flame_base_world + flame_dir_world * flame_length
                    flame_left_world = flame_base_world - flame_side_world * flame_width / 2.0
                    flame_right_world = flame_base_world + flame_side_world * flame_width / 2.0

                    # Apply camera transform to points
                    flame_points_screen = [camera.apply(p) for p in [flame_left_world, flame_right_world, flame_tip_world]]

                    # Draw flame polygon
                    try: pygame.draw.polygon(surface, COLOR_FLAME, flame_points_screen)
                    except NameError: pygame.draw.line(surface, RED, camera.apply(flame_base_world), camera.apply(flame_tip_world), 3) # Fallback line


        return num_broken_visually

    def calculate_subassembly_world_com(self, assembly_parts: list[PlacedPart]) -> pygame.math.Vector2:
        """ Calculates the approximate world CoM for a subset of parts (used for splitting). """
        if not assembly_parts: return self.pos # Return current origin if empty
        com_numerator = pygame.math.Vector2(0, 0)
        total_assembly_mass = 0
        # Estimate fuel distribution within the subassembly
        subassembly_tank_capacity = sum(p.part_data.get("fuel_capacity", 0) for p in self.fuel_tanks if p in assembly_parts)
        subassembly_tank_capacity = max(1.0, subassembly_tank_capacity)
        # Assume fuel is distributed proportionally based on original total fuel and capacity
        total_original_capacity = sum(p.part_data.get("fuel_capacity", 0) for p in self.fuel_tanks)
        total_original_capacity = max(1.0, total_original_capacity)
        total_fuel_mass = self.current_fuel * self.fuel_mass_per_unit

        for part in assembly_parts:
            part_mass_static = part.part_data.get("mass", 0)
            part_fuel_mass = 0
            # Estimate fuel in this part based on its capacity relative to *original* total
            if part.part_data.get("type") == "FuelTank" and total_original_capacity > 0:
                part_fuel_mass = total_fuel_mass * (part.part_data.get("fuel_capacity", 0) / total_original_capacity)
                # Clamp fuel mass if subassembly capacity is lower than estimated share
                part_fuel_mass = min(part_fuel_mass, (part.part_data.get("fuel_capacity",0) / subassembly_tank_capacity) * total_fuel_mass if subassembly_tank_capacity > 0 else 0)


            part_mass_current = part_mass_static + part_fuel_mass
            com_numerator += self.get_world_part_center(part) * part_mass_current # Use world pos * current mass
            total_assembly_mass += part_mass_current

        if total_assembly_mass <= 0:
            # Fallback: return world center of the first part in the assembly
            return self.get_world_part_center(assembly_parts[0]) if assembly_parts else self.pos

        return com_numerator / total_assembly_mass


# --- Background/Terrain Functions ---
def create_stars(count, bounds):
    """ Creates a list of stars (position, depth) within given world bounds. """
    stars = []
    depth_range = bounds.height # Use height for depth variation
    for _ in range(count):
        x = random.uniform(bounds.left, bounds.right)
        y = random.uniform(bounds.top, bounds.bottom)
        z = random.uniform(1, max(2, depth_range)) # Depth (higher z = further away)
        stars.append((pygame.math.Vector2(x, y), z))
    return stars

def get_air_density(altitude_agl):
    """ Calculates air density (kg/m^3) based on altitude above ground level (m). """
    # --- UPDATED ATMOSPHERE MODEL ---
    scale_height = ATMOSPHERE_SCALE_HEIGHT

    if altitude_agl < 0:
        # Below ground? Use sea level density.
        return AIR_DENSITY_SEA_LEVEL
    elif 0 <= altitude_agl <= ATMOSPHERE_EXP_LIMIT:
        # Exponential decay region (0m to 35km)
        density = AIR_DENSITY_SEA_LEVEL * math.exp(-altitude_agl / scale_height)
        return max(AIR_DENSITY_VACUUM, density)
    elif ATMOSPHERE_EXP_LIMIT < altitude_agl <= ATMOSPHERE_LINEAR_LIMIT:
        # Linear interpolation region (35km to 70km)
        # Calculate density at the start of this region (35km)
        density_at_35k = AIR_DENSITY_SEA_LEVEL * math.exp(-ATMOSPHERE_EXP_LIMIT / scale_height)
        density_at_35k = max(AIR_DENSITY_VACUUM, density_at_35k) # Ensure non-negative

        # Calculate the target density at the end of this region (70km)
        density_at_70k_target = AIR_DENSITY_SEA_LEVEL * ATMOSPHERE_TARGET_DENSITY_FACTOR

        # Calculate the interpolation factor (0 at 35km, 1 at 70km)
        interp_factor = max(0.0, min(1.0,
            (altitude_agl - ATMOSPHERE_EXP_LIMIT) / (ATMOSPHERE_LINEAR_LIMIT - ATMOSPHERE_EXP_LIMIT)
        ))

        # Linear interpolation: density = start_density * (1-t) + end_density * t
        density = density_at_35k * (1.0 - interp_factor) + density_at_70k_target * interp_factor
        return max(AIR_DENSITY_VACUUM, density)
    else:
        # Above linear limit (70km+): Vacuum
        return AIR_DENSITY_VACUUM

def draw_earth_background(surface, camera, stars):
    """ Draws the sky gradient, stars, and ground based on camera position. """
    screen_rect = surface.get_rect()
    # Use camera center Y to determine atmosphere level
    avg_world_y = camera.offset.y + camera.height / 2
    ground_screen_y = camera.apply(pygame.math.Vector2(0, GROUND_Y)).y

    # Determine background color based on altitude
    try: _ = BLUE_SKY_Y_LIMIT; _=SPACE_Y_LIMIT; _=COLOR_SKY_BLUE; _=COLOR_SPACE_BLACK
    except NameError: BLUE_SKY_Y_LIMIT=-2000; SPACE_Y_LIMIT=-15000; COLOR_SKY_BLUE=(135,206,250); COLOR_SPACE_BLACK=(0,0,0) # Fallbacks

    if avg_world_y > BLUE_SKY_Y_LIMIT: # Low altitude - Blue Sky
        # Fill blue above ground, or full screen if ground not visible
        if ground_screen_y < screen_rect.bottom:
            pygame.draw.rect(surface, COLOR_SKY_BLUE, (0, 0, screen_rect.width, ground_screen_y))
        else:
            surface.fill(COLOR_SKY_BLUE)
    elif avg_world_y < SPACE_Y_LIMIT: # High altitude - Space Black
        surface.fill(COLOR_SPACE_BLACK)
        draw_stars(surface, stars, camera, alpha=255) # Draw stars fully visible
    else: # Transition altitude - Gradient Sky
        # Interpolate between sky blue and space black
        interp = max(0.0, min(1.0, (avg_world_y - BLUE_SKY_Y_LIMIT) / (SPACE_Y_LIMIT - BLUE_SKY_Y_LIMIT)))
        bg_color = pygame.Color(COLOR_SKY_BLUE).lerp(COLOR_SPACE_BLACK, interp)
        surface.fill(bg_color)
        # Fade in stars
        star_alpha = int(255 * interp)
        if star_alpha > 10:
            draw_stars(surface, stars, camera, alpha=star_alpha)

def draw_stars(surface, stars, camera, alpha=255):
    """ Draws stars with parallax effect based on camera offset and star depth. """
    if alpha <= 0: return
    screen_rect = surface.get_rect()
    base_color = pygame.Color(200, 200, 200) # Base star color
    try: depth_scaling = STAR_FIELD_DEPTH
    except NameError: depth_scaling = 10000 # Fallback

    for world_pos, z in stars:
        # Parallax factor: closer stars (smaller z) move less than camera
        # Further stars (larger z) move more with camera? No, should be opposite.
        # Parallax = 1 / depth_factor. Depth factor increases with z.
        # Let's scale z relative to half the total depth for a multiplier effect.
        parallax_factor = 1.0 / ( (z / (depth_scaling / 20.0)) + 1.0) # Adjust divisor for desired effect strength

        # Effective camera offset for this star based on parallax
        effective_camera_offset = camera.offset * parallax_factor
        # Calculate screen position
        screen_pos = world_pos - effective_camera_offset
        sx, sy = int(screen_pos.x), int(screen_pos.y)

        # Cull stars outside the screen
        if 0 <= sx < screen_rect.width and 0 <= sy < screen_rect.height:
            # Star size based on depth (further = smaller)
            size = max(1, int(2.5 * (1.0 - z / max(1, depth_scaling))))
            # Apply alpha transparency
            alpha_factor = alpha / 255.0
            final_color_tuple = (int(base_color.r * alpha_factor),
                                 int(base_color.g * alpha_factor),
                                 int(base_color.b * alpha_factor))
            # Draw if color is not black
            if final_color_tuple != (0,0,0):
                 try:
                     pygame.draw.circle(surface, final_color_tuple, (sx, sy), size)
                 except ValueError: pass # Ignore potential color errors

def draw_terrain(surface, camera):
    """ Draws the ground surface. """
    world_width = WORLD_WIDTH; ground_y = GROUND_Y; ground_color = COLOR_GROUND
    # Draw a wide rectangle representing the ground, considering camera view
    # Extend width beyond screen to avoid visible edges when panning
    view_rect_world = pygame.Rect(camera.offset.x - world_width, ground_y,
                                  camera.width + world_width * 2, SCREEN_HEIGHT * 2) # Make height large enough
    rect_screen = camera.apply_rect(view_rect_world)
    pygame.draw.rect(surface, ground_color, rect_screen)


# --- Simulation Runner Function ---
def run_simulation(screen, clock, blueprint_file):
    print(f"--- Starting Simulation ---")
    print(f"Loading blueprint: {blueprint_file}")
    if not os.path.exists(blueprint_file):
        print(f"Error: Blueprint file not found: {blueprint_file}")
        return # Exit if blueprint doesn't exist

    initial_blueprint = RocketBlueprint.load_from_json(blueprint_file)
    if not initial_blueprint or not initial_blueprint.parts:
        print("Blueprint load failed or is empty.")
        return # Exit if load fails or empty

    all_rockets: list[FlyingRocket] = [] # List to hold all active rocket assemblies
    controlled_rocket: FlyingRocket | None = None # The assembly currently under player control
    next_sim_id = 0 # Counter for unique rocket instance IDs

    # Find the original root part (CommandPod preferably) from the blueprint
    original_root_part_instance = None
    if initial_blueprint.parts:
        for part in initial_blueprint.parts:
             if part.part_data and part.part_data.get("type") == "CommandPod":
                 original_root_part_instance = part
                 break
        if not original_root_part_instance and initial_blueprint.parts:
            # Fallback to the first part if no command pod found
            original_root_part_instance = initial_blueprint.parts[0]

    # Initial setup: Find connected components in the blueprint
    initial_subassemblies = initial_blueprint.find_connected_subassemblies()
    if not initial_subassemblies:
         print("Error: No parts found after initial connectivity check.")
         return

    initial_spawn_y_offset = 5 # Spawn slightly above ground

    # Create FlyingRocket instances for each initial subassembly
    for i, assembly_parts in enumerate(initial_subassemblies):
        if not assembly_parts: continue # Skip empty assemblies if any

        # Calculate initial position to place the assembly on the ground
        # Need CoM and lowest point relative to blueprint origin for the assembly
        temp_bp_for_calc = RocketBlueprint()
        temp_bp_for_calc.parts = assembly_parts
        initial_com_local = temp_bp_for_calc.calculate_subassembly_world_com(assembly_parts) # Relative to blueprint 0,0
        lowest_offset_y = temp_bp_for_calc.get_lowest_point_offset_y() # Y-offset of lowest point from blueprint 0,0
        start_x = i * 50 # Stagger initial positions horizontally
        target_lowest_y = GROUND_Y - initial_spawn_y_offset # Desired world Y for the lowest point
        # Calculate required world Y for the CoM
        start_y = target_lowest_y - (lowest_offset_y - initial_com_local.y)
        target_initial_com_pos = pygame.math.Vector2(start_x, start_y)

        # Determine if this assembly contains the original root part
        contains_original_root = original_root_part_instance and (original_root_part_instance in assembly_parts)
        # Assign primary control if it contains the root, or if it's the first assembly and no root assigned yet
        is_primary = (controlled_rocket is None and contains_original_root) or \
                     (controlled_rocket is None and i == 0) # Fallback to first assembly

        try:
            rocket_instance = FlyingRocket(list(assembly_parts), # Pass a copy of the list
                                           target_initial_com_pos,
                                           initial_angle=0,
                                           initial_vel=pygame.math.Vector2(0,0),
                                           sim_instance_id=next_sim_id,
                                           is_primary_control=is_primary,
                                           original_root_ref=original_root_part_instance) # Pass ref to original root
            all_rockets.append(rocket_instance)
            if is_primary:
                 controlled_rocket = rocket_instance
                 # Re-check control status based on root ref presence in THIS instance
                 root_ref_in_instance = controlled_rocket.original_root_part_ref
                 controlled_rocket.has_active_control = (root_ref_in_instance is not None) and \
                                                        (root_ref_in_instance in controlled_rocket.parts) and \
                                                        (not root_ref_in_instance.is_broken)
                 control_status_msg = 'CONTROLLED' if controlled_rocket.has_active_control else 'NO CONTROL (Root Missing/Broken)'
                 print(f"Created initial rocket {next_sim_id} ({control_status_msg}) with {len(assembly_parts)} parts.")
            else:
                 rocket_instance.has_active_control = False # Ensure non-primary starts without control flag
                 print(f"Created initial rocket {next_sim_id} (DEBRIS/UNCONTROLLED) with {len(assembly_parts)} parts.")
            next_sim_id += 1
        except Exception as e:
            print(f"Error initializing rocket instance {next_sim_id}: {e}")


    # Fallback control assignment if primary assignment failed but rockets exist
    if controlled_rocket is None and all_rockets:
         print("Warning: No primary control assigned initially. Assigning fallback control to first rocket.")
         controlled_rocket = all_rockets[0]
         # Check if this fallback rocket actually has the root part
         root_ref_in_fallback = controlled_rocket.original_root_part_ref
         controlled_rocket.has_active_control = (root_ref_in_fallback is not None) and \
                                               (root_ref_in_fallback in controlled_rocket.parts) and \
                                               (not root_ref_in_fallback.is_broken)


    # --- Simulation Setup ---
    camera = Camera(SCREEN_WIDTH, SCREEN_HEIGHT)
    if controlled_rocket: camera.update(controlled_rocket.get_world_com())
    elif all_rockets: camera.update(all_rockets[0].get_world_com())
    else: camera.update(pygame.math.Vector2(0, GROUND_Y)) # Center on ground if no rockets

    # Create star field
    try:
        # Define bounds for star generation (wider and taller than visible area)
        star_area_bounds = pygame.Rect(-WORLD_WIDTH*2, SPACE_Y_LIMIT - STAR_FIELD_DEPTH,
                                       WORLD_WIDTH*4, abs(SPACE_Y_LIMIT) + GROUND_Y + STAR_FIELD_DEPTH * 1.5)
        stars = create_stars(STAR_COUNT*2, star_area_bounds) # Create more stars over larger area
    except NameError: stars = [] # Fallback if constants missing

    # UI Fonts and Particle Manager
    ui_font = pygame.font.SysFont(None, 20)
    ui_font_large = pygame.font.SysFont(None, 36)
    particle_manager = ParticleManager()

    sim_running = True
    last_respawn_time = time.time() # Cooldown for respawn key

    # --- Main Simulation Loop ---
    while sim_running:
        dt = clock.tick(60) / 1000.0 # Delta time in seconds
        dt = min(dt, 0.05) # Clamp delta time to prevent physics instability

        # Lists for managing rocket creation/deletion during the frame
        newly_created_rockets_this_frame: list[FlyingRocket] = []
        rockets_to_remove_this_frame: list[FlyingRocket] = []

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    sim_running = False # Exit simulation loop

                # --- Controlled Rocket Actions ---
                if controlled_rocket and controlled_rocket.has_active_control:
                    # Toggle Master Thrust
                    if event.key == pygame.K_SPACE:
                        controlled_rocket.master_thrust_enabled = not controlled_rocket.master_thrust_enabled

                    # Deploy Parachutes (Deploy all ready chutes on controlled vessel)
                    if event.key == pygame.K_p:
                        print("Attempting parachute deployment...")
                        chutes_deployed_this_press = 0
                        # Iterate through parachutes on the controlled rocket
                        for chute in controlled_rocket.parachutes:
                            if not chute.deployed and not chute.is_broken:
                                chute.deployed = True
                                print(f"Deployed {chute.part_id} via key.")
                                chutes_deployed_this_press += 1
                        if chutes_deployed_this_press == 0: print("No ready parachutes found.")

                # --- Respawn ---
                current_time = time.time()
                if event.key == pygame.K_r and (current_time - last_respawn_time > 1.0): # 1 sec cooldown
                    print("--- RESPAWNING ROCKET ---")
                    last_respawn_time = current_time
                    # Clear existing state
                    all_rockets.clear()
                    controlled_rocket = None
                    newly_created_rockets_this_frame.clear()
                    rockets_to_remove_this_frame.clear()
                    particle_manager.particles.clear() # Clear particles

                    # Reload blueprint and re-initialize
                    reloaded_blueprint = RocketBlueprint.load_from_json(blueprint_file)
                    if reloaded_blueprint and reloaded_blueprint.parts:
                        # Repeat the initial setup logic
                        original_root_part_instance = None
                        for part in reloaded_blueprint.parts:
                             if part.part_data and part.part_data.get("type") == "CommandPod":
                                 original_root_part_instance = part; break
                        if not original_root_part_instance: original_root_part_instance = reloaded_blueprint.parts[0]

                        initial_subassemblies = reloaded_blueprint.find_connected_subassemblies()
                        for i, assembly_parts in enumerate(initial_subassemblies):
                             # Recalculate spawn position
                             temp_bp=RocketBlueprint(); temp_bp.parts=assembly_parts
                             initial_com_local=temp_bp.calculate_subassembly_world_com(assembly_parts)
                             lowest_offset_y=temp_bp.get_lowest_point_offset_y()
                             start_x=i*50; target_lowest_y=GROUND_Y-initial_spawn_y_offset
                             start_y=target_lowest_y-(lowest_offset_y-initial_com_local.y)
                             target_initial_com_pos=pygame.math.Vector2(start_x, start_y)

                             contains_original_root = original_root_part_instance and (original_root_part_instance in assembly_parts)
                             is_primary = (controlled_rocket is None and contains_original_root) or (controlled_rocket is None and i == 0)
                             try:
                                 rocket_instance = FlyingRocket(list(assembly_parts), target_initial_com_pos, 0, pygame.math.Vector2(0,0), next_sim_id, is_primary, original_root_part_instance)
                                 # Use the creation list directly, add to all_rockets later
                                 newly_created_rockets_this_frame.append(rocket_instance)
                                 if is_primary:
                                     controlled_rocket = rocket_instance
                                     root_ref=controlled_rocket.original_root_part_ref
                                     controlled_rocket.has_active_control=(root_ref is not None) and (root_ref in controlled_rocket.parts) and (not root_ref.is_broken)
                                 else: rocket_instance.has_active_control = False
                                 next_sim_id += 1
                             except Exception as e: print(f"Respawn Error creating instance: {e}")

                        # Fallback control assignment after creating all respawned rockets
                        if controlled_rocket is None and newly_created_rockets_this_frame:
                             controlled_rocket = newly_created_rockets_this_frame[0]
                             root_ref=controlled_rocket.original_root_part_ref
                             controlled_rocket.has_active_control=(root_ref is not None) and (root_ref in controlled_rocket.parts) and (not root_ref.is_broken)
                        print("Respawn Complete.")
                    else:
                        print("Respawn Failed: Cannot reload blueprint.")
                    # Add newly created rockets from respawn to the main list
                    all_rockets.extend(newly_created_rockets_this_frame)
                    newly_created_rockets_this_frame.clear() # Clear the temp list


            # --- Activate Part via Click ---
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: # Left Click
                 if controlled_rocket: # Can only interact with controlled rocket
                     click_screen_pos = pygame.math.Vector2(event.pos)
                     # Convert screen click to world coordinates
                     click_world_pos = click_screen_pos + camera.offset
                     controlled_rocket.activate_part_at_pos(click_world_pos)


        # --- Continuous Controls (Throttle) ---
        if controlled_rocket and controlled_rocket.has_active_control:
            keys = pygame.key.get_pressed()
            throttle_change = 0
            if keys[pygame.K_w] or keys[pygame.K_UP]: # Increase throttle
                throttle_change += THROTTLE_CHANGE_RATE * dt
            if keys[pygame.K_s] or keys[pygame.K_DOWN]: # Decrease throttle
                throttle_change -= THROTTLE_CHANGE_RATE * dt

            if throttle_change != 0:
                controlled_rocket.throttle_level = max(0.0, min(1.0, controlled_rocket.throttle_level + throttle_change))


        # --- Update Rockets ---
        for rocket in all_rockets:
            if not rocket.is_active: continue # Skip inactive rockets

            # Calculate air density at the rocket's CoM altitude
            current_air_density = 0.0
            try:
                altitude_agl = max(0, GROUND_Y - rocket.get_world_com().y) # Altitude above ground level
                current_air_density = get_air_density(altitude_agl)
            except NameError: current_air_density = 0.0 # Fallback if function missing

            rocket.update(dt, current_air_density, particle_manager)

            # Mark rocket for removal if it became inactive during update (e.g., all parts destroyed)
            if not rocket.is_active and rocket not in rockets_to_remove_this_frame:
                rockets_to_remove_this_frame.append(rocket)


        # --- Inter-Rocket Collision Detection & Resolution ---
        collision_pairs_processed = set() # Avoid processing same pair multiple times per frame
        for i, r1 in enumerate(all_rockets):
            if r1 in rockets_to_remove_this_frame or not r1.is_active or not r1.parts: continue

            for j in range(i + 1, len(all_rockets)):
                r2 = all_rockets[j]
                if r2 in rockets_to_remove_this_frame or not r2.is_active or not r2.parts: continue

                # Broad phase (optional - check distance between CoMs first?)

                # Narrow phase: Check collision between parts of r1 and r2
                collision_found_between_r1_r2 = False
                for p1_idx, p1 in enumerate(r1.parts):
                    if p1.is_broken: continue
                    rect1 = r1.get_world_part_aabb(p1) # Use approximate AABB for now

                    for p2_idx, p2 in enumerate(r2.parts):
                        if p2.is_broken: continue
                        rect2 = r2.get_world_part_aabb(p2)

                        if rect1.colliderect(rect2):
                            # Collision detected!
                            pair_key = tuple(sorted((r1.sim_instance_id, r2.sim_instance_id)))
                            if pair_key in collision_pairs_processed: continue # Already handled this pair

                            collision_found_between_r1_r2 = True
                            collision_pairs_processed.add(pair_key)

                            # --- Collision Response ---
                            # 1. Damage
                            relative_velocity = r1.vel - r2.vel
                            impact_speed = relative_velocity.length()
                            # Apply damage specifically to the colliding parts
                            r1.apply_collision_damage(impact_speed, particle_manager, specific_part_to_damage=p1)
                            r2.apply_collision_damage(impact_speed, particle_manager, specific_part_to_damage=p2)

                            # 2. Physics (Simple Separation) - Needs improvement
                            overlap_vec = pygame.math.Vector2(rect1.center) - pygame.math.Vector2(rect2.center)
                            if overlap_vec.length_squared() > 1e-4: # Avoid division by zero if centers are identical
                                dist_centers = overlap_vec.length()
                                # Estimate overlap based on AABB sizes
                                overlap_dist = (rect1.width + rect2.width + rect1.height + rect2.height) / 4.0 - dist_centers
                                if overlap_dist > 0:
                                    # Push rockets apart based on mass ratio
                                    push_factor = 0.5 # How strongly to push apart
                                    total_m = r1.total_mass + r2.total_mass
                                    if total_m > 0.01:
                                        separation_vector = overlap_vec.normalize() * overlap_dist
                                        r1.pos += separation_vector * (r2.total_mass / total_m) * push_factor
                                        r2.pos -= separation_vector * (r1.total_mass / total_m) * push_factor
                                        # Could also apply impulse based on relative velocity here

                            # Break inner loops once a collision is found for this pair
                            break
                    if collision_found_between_r1_r2: break


        # --- Process Connectivity Checks and Separations ---
        # Iterate over a copy of the list as new rockets might be added
        rockets_to_process = list(all_rockets)
        for rocket in rockets_to_process:
            # Skip rockets already marked for removal or inactive
            if rocket in rockets_to_remove_this_frame or not rocket.is_active: continue

            processed_split_this_frame = False # Flag to prevent handling both destruction and separation splits in one frame

            # 1. Check for splits due to part destruction
            if rocket.needs_connectivity_check:
                rocket.needs_connectivity_check = False # Reset flag
                # Find connected subassemblies within the *remaining* parts
                temp_bp = RocketBlueprint(); temp_bp.parts = rocket.parts
                subassemblies = temp_bp.find_connected_subassemblies()

                if len(subassemblies) > 1: # Structure broke into multiple pieces
                    processed_split_this_frame = True
                    print(f"[{rocket.sim_instance_id}] SPLIT (Destruction) into {len(subassemblies)} pieces!")
                    # Mark the original rocket for removal
                    if rocket not in rockets_to_remove_this_frame:
                        rockets_to_remove_this_frame.append(rocket)

                    # Create new FlyingRocket instances for each new piece
                    for assembly in subassemblies:
                         if not assembly: continue
                         try:
                             # Calculate CoM for the new piece
                             sub_com_world = rocket.calculate_subassembly_world_com(assembly)
                             # Check if this piece contains the original root part
                             contains_root = rocket.original_root_part_ref in assembly
                             # Assign control if original had control AND this piece has the root
                             is_primary = rocket.has_active_control and contains_root
                             # Create new rocket, inheriting velocity and angle
                             new_rocket = FlyingRocket(list(assembly), sub_com_world,
                                                       rocket.angle, rocket.vel,
                                                       next_sim_id, is_primary,
                                                       rocket.original_root_part_ref)
                             new_rocket.angular_velocity = rocket.angular_velocity # Inherit spin
                             newly_created_rockets_this_frame.append(new_rocket)
                             next_sim_id += 1
                         except Exception as e: print(f"Error creating rocket from destruction split: {e}")

            # 2. Check for splits due to activated separators (only if not already split by destruction)
            if rocket.pending_separation and not processed_split_this_frame:
                 separators_activated_this_frame = list(rocket.pending_separation)
                 rocket.pending_separation.clear() # Clear pending list
                 parts_before_separation = list(rocket.parts) # Copy current parts
                 split_occurred = False
                 current_parts_in_rocket = list(rocket.parts) # Parts remaining in this potential original assembly

                 for sep_part in separators_activated_this_frame:
                     # Ensure separator still exists in the current assembly being processed
                     if sep_part not in current_parts_in_rocket: continue

                     # Find world position of separator (for applying force)
                     separator_world_pos = rocket.get_world_part_center(sep_part)
                     separation_force = sep_part.part_data.get("separation_force", 1000) # N

                     # Temporarily remove the separator and check connectivity
                     parts_to_check_connectivity = [p for p in current_parts_in_rocket if p != sep_part]
                     temp_bp = RocketBlueprint(); temp_bp.parts = parts_to_check_connectivity
                     subassemblies = temp_bp.find_connected_subassemblies()

                     if len(subassemblies) > 1: # Separation caused a split!
                         split_occurred = True
                         print(f"  > SPLIT by Separator {sep_part.part_id} into {len(subassemblies)} pieces!")
                         # Mark the original combined rocket for removal
                         if rocket not in rockets_to_remove_this_frame:
                             rockets_to_remove_this_frame.append(rocket)

                         # Create new rockets for the separated pieces
                         for assembly in subassemblies:
                              if not assembly: continue
                              try:
                                  # Calculate CoM, check for root, determine control
                                  sub_com_world = rocket.calculate_subassembly_world_com(assembly)
                                  contains_root = rocket.original_root_part_ref in assembly
                                  is_primary = rocket.has_active_control and contains_root
                                  # Create new rocket instance
                                  new_rocket = FlyingRocket(list(assembly), sub_com_world,
                                                            rocket.angle, rocket.vel,
                                                            next_sim_id, is_primary,
                                                            rocket.original_root_part_ref)
                                  new_rocket.angular_velocity = rocket.angular_velocity # Inherit spin

                                  # Apply separation impulse (Force / Mass * dt_impulse)
                                  # Calculate direction away from separator
                                  separation_vector = new_rocket.get_world_com() - separator_world_pos
                                  separation_direction = separation_vector.normalize() if separation_vector.length_sq() > 1e-4 else pygame.math.Vector2(0,-1).rotate(-rocket.angle) # Default up if coincident
                                  # Impulse magnitude (approximate - adjust factor 0.05 as needed)
                                  impulse_magnitude = (separation_force / max(0.1, new_rocket.total_mass)) * 0.05
                                  delta_velocity = separation_direction * impulse_magnitude
                                  new_rocket.vel += delta_velocity # Apply impulse

                                  newly_created_rockets_this_frame.append(new_rocket)
                                  next_sim_id += 1
                              except Exception as e: print(f"Error creating rocket from separation split: {e}")
                         # Stop processing further separators on this original rocket as it's being replaced
                         break
                     else:
                         # Separation didn't split this piece, just remove the separator part from it
                         current_parts_in_rocket = parts_to_check_connectivity


                 # If a split occurred, the original rocket is removed later.
                 # If no split occurred BUT separators were processed, update the original rocket's part list
                 if not split_occurred and len(current_parts_in_rocket) < len(parts_before_separation):
                      rocket.parts = current_parts_in_rocket
                      # Update component lists and physics properties
                      rocket.engines = [e for e in rocket.engines if e in rocket.parts]
                      rocket.fuel_tanks = [t for t in rocket.fuel_tanks if t in rocket.parts]
                      rocket.parachutes = [pc for pc in rocket.parachutes if pc in rocket.parts]
                      rocket.separators = [s for s in rocket.separators if s in rocket.parts] # Remove used separator
                      if not rocket.parts:
                          rocket.is_active = False # Deactivate if empty
                          if rocket not in rockets_to_remove_this_frame: rockets_to_remove_this_frame.append(rocket)
                      else:
                          rocket.calculate_physics_properties(); rocket.calculate_bounds()


        # --- Update Rocket Lists ---
        # Add newly created rockets
        if newly_created_rockets_this_frame:
            new_controlled_rocket = None
            for new_rocket in newly_created_rockets_this_frame:
                all_rockets.append(new_rocket)
                # Check if this new rocket should take control
                if new_rocket.has_active_control:
                    # If there was an old controlled rocket, disable its control flag
                    if controlled_rocket and controlled_rocket not in rockets_to_remove_this_frame:
                        controlled_rocket.has_active_control = False
                    new_controlled_rocket = new_rocket # Assign control to this new one
            # Update the main controlled rocket reference if a new one was assigned
            if new_controlled_rocket:
                controlled_rocket = new_controlled_rocket

        # Remove rockets marked for deletion
        if rockets_to_remove_this_frame:
            was_controlled_rocket_removed = controlled_rocket in rockets_to_remove_this_frame
            all_rockets = [r for r in all_rockets if r not in rockets_to_remove_this_frame]

            # Handle loss of controlled rocket
            if was_controlled_rocket_removed:
                controlled_rocket = None
                # Try to find a new controlled rocket among the remaining ones
                # Prioritize rockets that explicitly have the has_active_control flag set
                for rkt in all_rockets:
                    if rkt.has_active_control:
                        controlled_rocket = rkt
                        break
                # If none found, fallback: find any rocket that still has the original root part intact
                if not controlled_rocket:
                    for rkt in all_rockets:
                        root_ref = rkt.original_root_part_ref
                        if root_ref and root_ref in rkt.parts and not root_ref.is_broken:
                            controlled_rocket = rkt
                            controlled_rocket.has_active_control = True # Explicitly grant control
                            print(f"Fallback control assigned to Rocket {controlled_rocket.sim_instance_id}.")
                            break

        # --- Camera Update ---
        if controlled_rocket:
            camera.update(controlled_rocket.get_world_com())
        elif all_rockets: # If no controlled rocket, follow the first one in the list
            camera.update(all_rockets[0].get_world_com())
        # Else: Camera stays where it was


        # --- Drawing ---
        screen.fill(BLACK) # Clear screen
        # Draw background (sky, stars)
        try: draw_earth_background(screen, camera, stars)
        except NameError: pass # Ignore if function missing
        # Draw terrain
        try: draw_terrain(screen, camera)
        except NameError: pass # Ignore if function missing

        # Draw rockets
        total_parts_drawn = 0
        total_broken_drawn = 0
        for rocket in all_rockets:
            if rocket.is_active:
                broken_count = rocket.draw(screen, camera, particle_manager)
                total_parts_drawn += len(rocket.parts)
                total_broken_drawn += broken_count

        # Update and draw particles
        particle_manager.update(dt)
        particle_manager.draw(screen, camera)

        # --- Draw UI ---
        if controlled_rocket:
            # Throttle Bar
            bar_w = 20; bar_h = 100; bar_x = 15; bar_y = SCREEN_HEIGHT - bar_h - 40
            pygame.draw.rect(screen, COLOR_UI_BAR_BG, (bar_x, bar_y, bar_w, bar_h))
            fill_h = bar_h * controlled_rocket.throttle_level
            pygame.draw.rect(screen, COLOR_UI_BAR, (bar_x, bar_y + bar_h - fill_h, bar_w, fill_h))
            pygame.draw.rect(screen, WHITE, (bar_x, bar_y, bar_w, bar_h), 1) # Border
            th_label = ui_font.render("Thr", True, WHITE)
            screen.blit(th_label, (bar_x, bar_y + bar_h + 5))
            th_value = ui_font.render(f"{controlled_rocket.throttle_level*100:.0f}%", True, WHITE)
            screen.blit(th_value, (bar_x, bar_y - 18))

            # Telemetry Data
            alt_agl = max(0, GROUND_Y - controlled_rocket.get_lowest_point_world().y) # Altitude above ground
            alt_msl = GROUND_Y - controlled_rocket.get_world_com().y # Altitude above sea level (0) - assuming ground is at positive Y
            ctrl_status = "OK" if controlled_rocket.has_active_control else "NO CTRL"
            thrust_status = "ON" if controlled_rocket.master_thrust_enabled else "OFF"
            landed_status = "LANDED" if controlled_rocket.landed else "FLYING"
            max_temp_k = controlled_rocket.max_temp_reading
            # Determine color for temperature readout
            temp_color = WHITE
            hottest_part_max_temp = AMBIENT_TEMPERATURE
            if controlled_rocket.parts: # Find max allowable temp on current parts
                 hottest_part_max_temp = max(p.part_data.get('max_temp', AMBIENT_TEMPERATURE) for p in controlled_rocket.parts)
            if max_temp_k > REENTRY_EFFECT_THRESHOLD_TEMP: temp_color = (255, 255, 0) # Yellow warning
            if max_temp_k > hottest_part_max_temp * 0.9: temp_color = (255, 100, 0) # Orange critical
            if max_temp_k > hottest_part_max_temp: temp_color = RED # Red danger

            status_texts = [
                f"Alt(AGL): {alt_agl:.1f}m", f"Alt(MSL): {alt_msl:.1f}m",
                f"Vvel: {controlled_rocket.vel.y:.1f}", f"Hvel: {controlled_rocket.vel.x:.1f}",
                f"Speed: {controlled_rocket.vel.length():.1f}",
                f"Angle: {controlled_rocket.angle:.1f}", f"AngVel: {controlled_rocket.angular_velocity:.1f}",
                f"Thr: {controlled_rocket.throttle_level*100:.0f}% [{thrust_status}]",
                f"Fuel: {controlled_rocket.current_fuel:.1f}",
                f"Mass: {controlled_rocket.total_mass:.1f}kg",
                f"Ctrl: {ctrl_status}", f"Status: {landed_status}",
                f"Inst: {controlled_rocket.sim_instance_id}",
                f"MaxTemp: {max_temp_k:.0f}K"
            ]
            text_y = 10
            control_color = WHITE if controlled_rocket.has_active_control else RED
            for i, text in enumerate(status_texts):
                line_color = temp_color if "MaxTemp" in text else (control_color if "Ctrl" in text else WHITE)
                text_surf = ui_font.render(text, True, line_color)
                screen.blit(text_surf, (bar_x + bar_w + 10, text_y + i * 18))

        elif not all_rockets: # All rockets are gone
            destroyed_text = ui_font_large.render("ALL ROCKETS DESTROYED", True, RED)
            text_rect = destroyed_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            screen.blit(destroyed_text, text_rect)
            respawn_text = ui_font.render("Press 'R' to Respawn", True, WHITE)
            respawn_rect = respawn_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 40))
            screen.blit(respawn_text, respawn_rect)

        # FPS Counter and Debug Info
        fps = clock.get_fps()
        screen.blit(ui_font.render(f"FPS: {fps:.1f}", True, WHITE), (SCREEN_WIDTH - 100, 10))
        screen.blit(ui_font.render(f"Objs: {len(all_rockets)}", True, WHITE), (SCREEN_WIDTH - 100, 30))
        screen.blit(ui_font.render(f"Parts: {total_parts_drawn}", True, WHITE), (SCREEN_WIDTH - 100, 50))
        screen.blit(ui_font.render(f"Particles: {len(particle_manager.particles)}", True, WHITE), (SCREEN_WIDTH - 100, 70))

        # --- Update Display ---
        pygame.display.flip()

    print("--- Exiting Simulation ---")


# --- Direct Run Logic (for testing) ---
if __name__ == '__main__':
     pygame.init()
     screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
     pygame.display.set_caption("PySpaceFlight - Direct Sim Test")
     clock = pygame.time.Clock()

     # Define test blueprint path
     assets_dir = "assets"
     test_blueprint = os.path.join(assets_dir, "grid_rocket_aero_test.json")

     # Create a default blueprint if the test one doesn't exist
     if not os.path.exists(test_blueprint):
         print("Test blueprint not found, creating basic default...")
         # Ensure assets directory exists
         os.makedirs(assets_dir, exist_ok=True)
         # Create a simple default rocket
         bp = RocketBlueprint("Default Test")
         bp.add_part("pod_mk1", (0, 0))
         bp.add_part("tank_small", (0, 30)) # Approximate placement
         bp.add_part("engine_basic", (0, 70)) # Approximate placement
         bp.save_to_json(test_blueprint)
         print(f"Saved default blueprint to {test_blueprint}")

     # Run the simulation
     run_simulation(screen, clock, test_blueprint)

     pygame.quit()
     sys.exit()