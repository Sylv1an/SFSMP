# flight_sim.py
import pygame
import math
import sys
import random
import time
import os
from collections import deque # Needed for BFS

# Import necessary classes/functions
from parts import draw_part_shape, get_part_data
from rocket_data import RocketBlueprint, PlacedPart, AMBIENT_TEMPERATURE, FUEL_MASS_PER_UNIT
from ui_elements import SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, BLACK, GRAY, RED, GREEN, BLUE, LIGHT_GRAY
# Define colors if not in ui_elements
try: from ui_elements import COLOR_SKY_BLUE, COLOR_SPACE_BLACK, COLOR_HORIZON, COLOR_GROUND
except ImportError: COLOR_SKY_BLUE, COLOR_SPACE_BLACK, COLOR_HORIZON, COLOR_GROUND = (135, 206, 250), (0,0,0), (170, 210, 230), (0, 150, 0)
try: from ui_elements import COLOR_FLAME, COLOR_UI_BAR, COLOR_UI_BAR_BG, COLOR_EXPLOSION
except ImportError: COLOR_FLAME, COLOR_UI_BAR, COLOR_UI_BAR_BG, COLOR_EXPLOSION = (255,100,0), (0,200,0), (50,50,50), [(255,255,0),(255,150,0),(200,50,0),(150,150,150)]
try: from ui_elements import COLOR_ENGINE_ENABLED, COLOR_ENGINE_DISABLED, COLOR_ACTIVATABLE_READY, COLOR_ACTIVATABLE_USED
except ImportError: COLOR_ENGINE_ENABLED, COLOR_ENGINE_DISABLED, COLOR_ACTIVATABLE_READY, COLOR_ACTIVATABLE_USED = GREEN, RED, BLUE, GRAY

# --- Flight Sim Constants ---
GRAVITY = 9.81 * 6
ROTATION_SPEED = 200
REACTION_WHEEL_TORQUE = 10000
ANGULAR_DAMPING = 0.3
COLLISION_DAMAGE_FACTOR = 0.7
MIN_IMPACT_VEL_DAMAGE = 10.0
THROTTLE_CHANGE_RATE = 0.5
GROUND_Y = 1000
WORLD_WIDTH = 5000
BLUE_SKY_Y_LIMIT = -2000
SPACE_Y_LIMIT = -15000
STAR_COUNT = 20000
STAR_FIELD_DEPTH = 10000
# *** NEW/ADJUSTED CONSTANTS for Separation Fix ***
COLLISION_GRACE_FRAMES = 150 # Number of frames new pieces ignore each other (~0.25 sec @ 60fps)
POSITIONAL_NUDGE_FACTOR = 1.0 # Pixels to initially push pieces apart (Keep, might not be needed if impulse works well)

# --- Air Density & Atmosphere ---
AIR_DENSITY_SEA_LEVEL = 0.9
AIR_DENSITY_VACUUM = 0.0
ATMOSPHERE_SCALE_HEIGHT = 8500.0
ATMOSPHERE_EXP_LIMIT = 35000.0
ATMOSPHERE_LINEAR_LIMIT = 70000.0
ATMOSPHERE_TARGET_DENSITY_FACTOR = 0.01

# --- Thermal Physics ---
HEAT_DISSIPATION_FACTOR_VACUUM = 1
HEAT_DISSIPATION_FACTOR_CONVECTION = 0.1
AERO_HEATING_FACTOR = 3e-4
OVERHEAT_DAMAGE_THRESHOLD_K = 50
OVERHEAT_DAMAGE_RATE = 100
REENTRY_EFFECT_THRESHOLD_TEMP = 700.0
REENTRY_EFFECT_MAX_TEMP_SCALE = 1.1
REENTRY_PARTICLE_COLOR_START = pygame.Color(150, 0, 0)
REENTRY_PARTICLE_COLOR_END = pygame.Color(255, 255, 150)
REENTRY_PARTICLE_LIFETIME = 0.6
REENTRY_PARTICLE_SPEED = 80

COLOR_EXPLOSION = [pygame.Color(c) for c in [(255,255,0),(255,150,0),(200,50,0), GRAY]]
DEFAULT_MAX_TEMP = 1600.0

# --- Particle System Classes (Unchanged) ---
class EffectParticle:
    def __init__(self, pos, vel, life, start_color, end_color=None, start_radius=3, end_radius=0):
        self.pos = pygame.math.Vector2(pos); self.vel = pygame.math.Vector2(vel)
        self.life = life; self.max_life = life; self.start_color = pygame.Color(start_color)
        self.end_color = pygame.Color(end_color) if end_color else self.start_color
        self.start_radius = start_radius; self.end_radius = end_radius
        self.current_radius = start_radius; self.current_color = self.start_color
    def update(self, dt):
        self.life -= dt
        if self.life > 0:
            self.pos += self.vel * dt
            self.vel *= 0.97 # Simple damping
            # Lerp color and radius
            time_ratio = max(0.0, min(1.0, 1.0 - (self.life / self.max_life)))
            try:
                self.current_color = self.start_color.lerp(self.end_color, time_ratio)
            except ValueError: # Handle potential issues with lerp if colors are identical?
                self.current_color = self.start_color
            self.current_radius = self.start_radius + (self.end_radius - self.start_radius) * time_ratio
            return True # Particle is still alive
        else:
            return False # Particle expired
    def draw(self, surface, camera):
        if self.life > 0 and self.current_radius >= 0.5: # Only draw if visible radius
            screen_pos = camera.apply(self.pos)
            # Basic culling
            if -self.current_radius < screen_pos.x < SCREEN_WIDTH + self.current_radius and \
               -self.current_radius < screen_pos.y < SCREEN_HEIGHT + self.current_radius:
                 try:
                     # Ensure color is a tuple of ints
                     draw_color = (int(self.current_color.r), int(self.current_color.g), int(self.current_color.b))
                     pygame.draw.circle(surface, draw_color, screen_pos, int(self.current_radius))
                 except ValueError: # Catch potential errors during drawing
                     pass # Optionally log an error here

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
        angle_offset = random.uniform(-25, 25)
        # Spark goes opposite to velocity direction
        if base_vel.length() > 0:
            spark_dir = -base_vel.normalize()
        else:
            spark_dir = pygame.math.Vector2(0, 1) # Default down if stationary

        spark_vel = spark_dir.rotate(angle_offset) * REENTRY_PARTICLE_SPEED * (0.5 + intensity_factor) + base_vel * 0.1 # Inherit some base velocity
        life = REENTRY_PARTICLE_LIFETIME * (0.7 + random.random() * 0.6) # Add randomness
        start_rad = 1 + 3 * intensity_factor
        end_rad = 0
        start_col = REENTRY_PARTICLE_COLOR_START.lerp(REENTRY_PARTICLE_COLOR_END, intensity_factor * 0.8)
        end_col = REENTRY_PARTICLE_COLOR_START # Fade back to red
        self.particles.append(EffectParticle(pos, spark_vel, life, start_col, end_col, start_rad, end_rad))

    def update(self, dt):
        # Update particles and remove expired ones
        self.particles = [p for p in self.particles if p.update(dt)]

    def draw(self, surface, camera):
        for p in self.particles:
            p.draw(surface, camera)

# --- Camera Class (Unchanged) ---
class Camera:
    def __init__(self, width, height):
        self.camera_rect = pygame.Rect(0, 0, width, height)
        self.width = width
        self.height = height
        self.offset = pygame.math.Vector2(0, 0) # World coords of top-left corner

    def apply(self, target_pos: pygame.math.Vector2) -> pygame.math.Vector2:
        # Convert world coordinates to screen coordinates
        return target_pos - self.offset

    def apply_rect(self, target_rect: pygame.Rect) -> pygame.Rect:
        # Convert a world Rect to screen coordinates
        return target_rect.move(-self.offset.x, -self.offset.y)

    def update(self, target_pos: pygame.math.Vector2):
        # Center the camera on the target world position
        x = target_pos.x - self.width // 2
        y = target_pos.y - self.height // 2
        # Update the offset
        self.offset = pygame.math.Vector2(x, y)

# --- FlyingRocket Class (Largely Unchanged internally, constructor takes frame) ---
class FlyingRocket:
    # *** MODIFIED __init__ to accept current_frame ***
    def __init__(self, parts_list: list[PlacedPart], initial_world_com_pos: pygame.math.Vector2, initial_angle=0, initial_vel=pygame.math.Vector2(0,0), sim_instance_id=0, is_primary_control=False, original_root_ref=None, current_frame=0):
        self.sim_instance_id = sim_instance_id
        self.parts = parts_list
        if not self.parts:
            raise ValueError("Cannot initialize FlyingRocket with an empty parts list.")
        self.blueprint_name = f"Rocket_{sim_instance_id}" # Mostly for debugging
        # Store a reference to the specific PlacedPart instance designated as the root
        self.original_root_part_ref = original_root_ref
        self.has_active_control = is_primary_control # Does this instance respond to player input?

        # *** NEW: Track creation frame (passed from run_simulation) ***
        self.creation_frame = current_frame

        # Physics State
        self.pos = pygame.math.Vector2(initial_world_com_pos) # World position of the origin (0,0) point used in blueprint
        self.vel = pygame.math.Vector2(initial_vel) # World velocity
        self.acc = pygame.math.Vector2(0, 0) # World acceleration
        self.angle = initial_angle # Degrees, positive = clockwise
        self.angular_velocity = 0.0 # Degrees per second

        # Control State
        self.throttle_level = 0.0 # 0.0 to 1.0
        self.master_thrust_enabled = False # Master switch for all engines

        # Component References & State (Populated below)
        self.engines: list[PlacedPart] = []
        self.fuel_tanks: list[PlacedPart] = []
        self.parachutes: list[PlacedPart] = []
        self.separators: list[PlacedPart] = []
        # Cache mapping engines to the tanks they can draw from
        self.engine_fuel_sources: dict[PlacedPart, list[PlacedPart]] = {}
        # Cache for direct part-to-part connections within this assembly
        self._part_connections_cache: dict[PlacedPart, list[PlacedPart]] | None = None

        # Populate component lists and assign indices
        for i, part in enumerate(self.parts):
            part.part_index = i # Store index within this rocket instance
            part_type = part.part_data.get("type")
            if part_type == "Engine":
                self.engines.append(part)
            elif part_type == "FuelTank":
                self.fuel_tanks.append(part)
                # Fuel state preservation: The PlacedPart instance passed in should
                # already have its current_fuel set correctly (either full from blueprint
                # or preserved from a split). We don't reset it here.
                # Clamp fuel just in case it's invalid.
                part.current_fuel = max(0.0, min(part.current_fuel, part.fuel_capacity))
            elif part_type == "Parachute":
                self.parachutes.append(part)
            elif part_type == "Separator":
                self.separators.append(part)

        # Calculated Physics Properties (Calculated below)
        self.total_mass = 0.01 # kg, includes fuel
        self.dry_mass = 0.0 # kg, excludes fuel
        self.moment_of_inertia = 10000.0 # kg*m^2 (approximation)
        self.center_of_mass_offset = pygame.math.Vector2(0, 0) # Offset from self.pos (blueprint origin) to CoM
        self.local_bounds = pygame.Rect(0,0,1,1) # AABB in local coordinates relative to blueprint origin

        # Initial calculations
        self.calculate_physics_properties() # Calculate mass, CoM offset, MoI
        self.calculate_bounds() # Calculate local AABB
        self._build_fuel_source_map() # Determine fuel flow paths

        # Correct initial world position: The initial_world_com_pos passed in *is* the target CoM.
        # We need to adjust self.pos (blueprint origin) so the calculated CoM ends up there.
        initial_com_offset_rotated = self.center_of_mass_offset.rotate(-self.angle)
        self.pos = initial_world_com_pos - initial_com_offset_rotated

        # Status Flags
        self.landed = False # Is the rocket currently stationary on the ground?
        self.thrusting = False # Are any engines currently firing?
        self.is_active = True # Should this rocket be simulated? (Set to False if destroyed/removed)
        self.pending_separation: list[PlacedPart] = [] # Separators activated this frame, processed in main loop
        self.needs_structural_update = False # Flag set when parts are destroyed/separated
        self.was_landed_last_frame = False # Used for detecting landing/takeoff transitions
        self.max_temp_reading = AMBIENT_TEMPERATURE # Max temp recorded across all parts this frame (for UI)
        # Track which engines actually fired (had fuel) this frame
        self.engine_firing_status: dict[PlacedPart, bool] = {e: False for e in self.engines}


    # --- Connectivity & Fuel Map Methods (Unchanged) ---
    def _build_connection_cache(self):
        """ Builds or rebuilds the cache mapping each part to its directly connected neighbors within this assembly. """
        self._part_connections_cache = {}
        all_parts_in_assembly = set(self.parts)
        CONNECTION_TOLERANCE_SQ = (3.0)**2 # Tolerance for points being "connected"

        # Pre-calculate world points (relative to blueprint origin) for all parts
        part_local_points = {} # Using local relative coords for calculation within assembly
        for part in self.parts:
            points = part.part_data.get("logical_points", {})
            local_points = {}
            for name, local_pos_rel_part in points.items():
                # Point's position relative to the *blueprint origin*
                local_points[name] = part.relative_pos + local_pos_rel_part.rotate(-part.relative_angle)
            part_local_points[part] = local_points

        for part in self.parts:
            self._part_connections_cache[part] = []
            current_part_data = part.part_data
            current_rules = current_part_data.get("attachment_rules", {})
            current_points = part_local_points.get(part, {})

            for other_part in all_parts_in_assembly:
                if other_part == part:
                    continue # Skip self

                other_part_data = other_part.part_data
                other_rules = other_part_data.get("attachment_rules", {})
                other_points = part_local_points.get(other_part, {})
                is_connected = False

                # Check connection between all points of current_part and other_part
                for cpn, clp in current_points.items(): # current point name, current local pos (rel to blueprint 0,0)
                    for opn, olp in other_points.items(): # other point name, other local pos (rel to blueprint 0,0)
                        # Check distance between the points (still in local blueprint space)
                        if (clp - olp).length_squared() < CONNECTION_TOLERANCE_SQ:
                            # Points are close, check rule compatibility
                            crn = self._get_rule_name_for_point(cpn, current_rules) # current rule name
                            orn = self._get_rule_name_for_point(opn, other_rules) # other rule name
                            if crn and orn:
                                if self._check_rule_compatibility(
                                    current_part_data.get("type"), crn, current_rules,
                                    other_part_data.get("type"), orn, other_rules):
                                    is_connected = True
                                    break # Found compatible connection between these points
                    if is_connected:
                        break # Found connection for this part pair

                if is_connected:
                    self._part_connections_cache[part].append(other_part)

    def _invalidate_connection_cache(self):
        """ Call this whenever the part list changes. """
        self._part_connections_cache = None

    def _get_connected_parts(self, part: PlacedPart) -> list[PlacedPart]:
        """ Returns a list of parts directly connected to the given part. Uses cache. """
        if self._part_connections_cache is None:
            self._build_connection_cache()
        # Ensure the part exists in the cache keys (might happen if called just after removal?)
        return self._part_connections_cache.get(part, [])

    def _get_rule_name_for_point(self, point_name: str, rules: dict) -> str | None:
        """ Maps a logical point name (e.g., 'bottom') to an attachment rule name (e.g., 'bottom_center'). """
        if "bottom" in point_name and "bottom_center" in rules: return "bottom_center"
        if "top" in point_name and "top_center" in rules: return "top_center"
        if "left" in point_name and "left_center" in rules: return "left_center"
        if "right" in point_name and "right_center" in rules: return "right_center"
        # Add more mappings if needed (e.g., 'back', 'front')
        return None

    def _check_rule_compatibility(self, type1, rule_name1, rules1, type2, rule_name2, rules2) -> bool:
        """ Checks if two attachment rules are compatible (type allowed, facing correct). """
        allowed_on_1 = rules1.get(rule_name1, {}).get("allowed_types", [])
        allowed_on_2 = rules2.get(rule_name2, {}).get("allowed_types", [])
        # Check if each part type is allowed by the other part's rule
        if type2 not in allowed_on_1 or type1 not in allowed_on_2:
            return False
        # Check if the attachment points are logically compatible (top-to-bottom, left-to-right)
        if ("top" in rule_name1 and "bottom" in rule_name2) or \
           ("bottom" in rule_name1 and "top" in rule_name2) or \
           ("left" in rule_name1 and "right" in rule_name2) or \
           ("right" in rule_name1 and "left" in rule_name2):
            return True
        # Add more compatibility checks if needed (e.g., front-to-back)
        return False

    def _find_accessible_tanks_for_engine(self, start_engine: PlacedPart) -> list[PlacedPart]:
        """ Performs a BFS from an engine to find connected fuel tanks, stopping at separators/other engines. """
        accessible_tanks: list[PlacedPart] = []
        queue = deque()
        visited_in_search = {start_engine} # Start search from engine

        # Start BFS with neighbors of the engine
        initial_neighbors = self._get_connected_parts(start_engine)
        for neighbor in initial_neighbors:
             if neighbor not in visited_in_search:
                 queue.append(neighbor)
                 visited_in_search.add(neighbor)

        while queue:
            current_part = queue.popleft()
            part_type = current_part.part_data.get("type")

            # Check if it's a fuel tank
            if part_type == "FuelTank":
                if current_part not in accessible_tanks: # Avoid duplicates
                    accessible_tanks.append(current_part)

            # Determine if fuel can flow *through* this part
            # Fuel stops at other engines and separators
            is_blocker = (part_type == "Separator") or (part_type == "Engine" and current_part != start_engine)

            if not is_blocker:
                # Continue BFS to neighbors
                neighbors = self._get_connected_parts(current_part)
                for neighbor in neighbors:
                    if neighbor not in visited_in_search:
                        visited_in_search.add(neighbor)
                        queue.append(neighbor)

        return accessible_tanks

    def _build_fuel_source_map(self):
        """ Builds the map connecting engines to their accessible fuel tanks. """
        # Ensure connections are up-to-date before searching
        self._invalidate_connection_cache()
        self.engine_fuel_sources = {}
        for engine in self.engines:
            self.engine_fuel_sources[engine] = self._find_accessible_tanks_for_engine(engine)

    # --- Physics Calculation Methods (Unchanged) ---
    def calculate_physics_properties(self):
        """ Recalculates total mass, dry mass, CoM offset (relative to self.pos), and MoI. """
        total_m = 0.0
        com_numerator = pygame.math.Vector2(0, 0)
        moi_sum = 0.0 # Moment of inertia sum relative to the calculated CoM
        current_dry_mass = 0.0

        # First pass: Calculate total mass and CoM offset
        for part in self.parts:
            part_mass_static = part.part_data.get("mass", 0)
            current_dry_mass += part_mass_static
            part_fuel_mass = 0.0
            if part.part_data.get("type") == "FuelTank":
                part_fuel_mass = part.current_fuel * FUEL_MASS_PER_UNIT

            part_mass_current = part_mass_static + part_fuel_mass
            total_m += part_mass_current
            # CoM calculation uses position relative to the blueprint origin (self.pos)
            com_numerator += part.relative_pos * part_mass_current

        self.dry_mass = current_dry_mass
        self.total_mass = max(0.01, total_m) # Avoid zero mass

        # Calculate CoM offset relative to blueprint origin (self.pos)
        if self.total_mass > 0.01:
            self.center_of_mass_offset = com_numerator / self.total_mass
        else:
            # Avoid division by zero, default to origin or first part pos?
            self.center_of_mass_offset = self.parts[0].relative_pos if self.parts else pygame.math.Vector2(0, 0)

        # Second pass: Calculate Moment of Inertia using Parallel Axis Theorem
        for part in self.parts:
             part_mass_static = part.part_data.get("mass", 0)
             part_fuel_mass = 0.0
             if part.part_data.get("type") == "FuelTank":
                 part_fuel_mass = part.current_fuel * FUEL_MASS_PER_UNIT
             part_mass_current = part_mass_static + part_fuel_mass

             # Approximate part's MoI around its own center (as rectangle)
             w = part.part_data.get("width", 1)
             h = part.part_data.get("height", 1)
             # MoI for a rectangle rotating around its center
             i_part_center = (1/12.0) * part_mass_current * (w**2 + h**2)

             # Distance squared from part's center (relative_pos) to the assembly's CoM (center_of_mass_offset)
             dist_vec = part.relative_pos - self.center_of_mass_offset
             d_sq = dist_vec.length_squared()

             # Parallel Axis Theorem: I = I_center + m*d^2
             moi_sum += i_part_center + part_mass_current * d_sq

        self.moment_of_inertia = max(1.0, moi_sum) # Avoid zero MoI

    def calculate_bounds(self):
        """ Calculates the AABB of the rocket in local coordinates relative to blueprint origin (self.pos). """
        if not self.parts:
            self.local_bounds = pygame.Rect(0, 0, 0, 0)
            return

        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')

        for p in self.parts:
            # Get part dimensions and center relative to blueprint origin
            half_w = p.part_data['width'] / 2.0
            half_h = p.part_data['height'] / 2.0
            center_x = p.relative_pos.x
            center_y = p.relative_pos.y
            # Find min/max extents based on part's AABB in local coords
            min_x = min(min_x, center_x - half_w)
            max_x = max(max_x, center_x + half_w)
            min_y = min(min_y, center_y - half_h)
            max_y = max(max_y, center_y + half_h)

        if min_x == float('inf'): # Handle case where there were parts but maybe no dimensions?
             self.local_bounds = pygame.Rect(0,0,0,0)
        else:
             self.local_bounds = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)

    # --- Coordinate and State Access Methods (Unchanged) ---
    def get_world_com(self) -> pygame.math.Vector2:
        """ Calculates the current world position of the center of mass. """
        # Rotate the CoM offset vector by the negative of the rocket's angle
        com_offset_rotated = self.center_of_mass_offset.rotate(-self.angle)
        # Add the rotated offset to the rocket's origin position
        return self.pos + com_offset_rotated

    def get_world_part_center(self, part: PlacedPart) -> pygame.math.Vector2:
        """ Calculates the current world position of a specific part's center. """
        # Rotate the part's relative position vector by the negative of the rocket's angle
        part_offset_rotated = part.relative_pos.rotate(-self.angle)
        # Add the rotated offset to the rocket's origin position
        return self.pos + part_offset_rotated

    def get_parts_near_world_pos(self, world_pos: pygame.math.Vector2, radius: float = 20.0) -> list[PlacedPart]:
        """ Finds parts whose centers are within a given radius of a world position. """
        nearby_parts = []
        radius_sq = radius * radius
        for part in self.parts:
             part_world_center = self.get_world_part_center(part)
             if (part_world_center - world_pos).length_squared() < radius_sq:
                 nearby_parts.append(part)
        return nearby_parts

    def get_lowest_point_world(self) -> pygame.math.Vector2:
        """ Finds the world coordinates of the point on the rocket with the largest Y value (lowest on screen). """
        if not self.parts:
            return self.pos # Return origin if no parts

        lowest_y = -float('inf')
        lowest_point_world = self.pos # Default fallback

        for part in self.parts:
             part_center_world = self.get_world_part_center(part)
             w = part.part_data.get('width', 0)
             h = part.part_data.get('height', 0)

             if w <= 0 or h <= 0: continue # Skip parts with no dimensions

             part_world_angle = self.angle # Assembly's angle

             # Define corners relative to part center
             corners_local = [
                 pygame.math.Vector2(-w/2, -h/2), pygame.math.Vector2( w/2, -h/2),
                 pygame.math.Vector2( w/2,  h/2), pygame.math.Vector2(-w/2,  h/2)
             ]
             # Rotate corners and translate to world space
             corners_world = [part_center_world + corner.rotate(-part_world_angle) for corner in corners_local]

             # Find the maximum Y value among the corners of this part
             for corner in corners_world:
                 if corner.y > lowest_y:
                     lowest_y = corner.y
                     lowest_point_world = corner # Track the point itself

        # If no valid corner found (e.g., all parts sizeless), return CoM
        if lowest_y == -float('inf'):
             return self.get_world_com()
        else:
             return lowest_point_world

    def get_world_part_aabb(self, part: PlacedPart) -> pygame.Rect:
         """ Calculates an approximate world-aligned AABB for a part (useful for broad-phase collision). """
         part_data = part.part_data
         w = part_data.get('width', 1)
         h = part_data.get('height', 1)
         world_center = self.get_world_part_center(part)
         # Calculate max dimension based on diagonal (conservative estimate for rotated part)
         max_dim = math.sqrt((w/2)**2 + (h/2)**2) * 2.1 # Add a small buffer
         # Create square AABB centered on the part
         aabb = pygame.Rect(0, 0, max_dim, max_dim)
         aabb.center = world_center
         return aabb

    def get_thrust_and_consumption(self, dt: float) -> tuple[pygame.math.Vector2, pygame.math.Vector2 | None, dict[PlacedPart, float]]:
        """ Calculates total thrust vector, average application point offset (local rotated), and requested fuel per engine. """
        total_force_vector = pygame.math.Vector2(0, 0)
        # Use thrust magnitude to weight the average application point calculation
        thrust_torque_numerator = pygame.math.Vector2(0, 0) # Sum of (position_offset * thrust_magnitude)
        total_thrust_magnitude_applied = 0.0
        fuel_consumption_request: dict[PlacedPart, float] = {} # Engine -> fuel needed this frame
        active_engine_count_this_frame = 0

        # Reset firing status for all engines
        for engine in self.engines:
            self.engine_firing_status[engine] = False

        # Check master switch and throttle
        if not self.master_thrust_enabled or self.throttle_level <= 0:
            self.thrusting = False
            return pygame.math.Vector2(0,0), None, {}

        # Iterate through engines
        for engine in self.engines:
            # Skip broken or disabled engines
            if engine.is_broken or not engine.engine_enabled:
                continue

            # Check available fuel
            accessible_tanks = self.engine_fuel_sources.get(engine, [])
            available_fuel_for_engine = sum(tank.current_fuel for tank in accessible_tanks if not tank.is_broken and tank.current_fuel > 1e-6)

            # Only calculate thrust if fuel *might* be available
            if available_fuel_for_engine > 1e-6:
                engine_thrust_potential = engine.part_data.get("thrust", 0)
                engine_consumption_rate = engine.part_data.get("fuel_consumption", 0)

                # Calculate thrust for this engine
                thrust_this_engine = engine_thrust_potential * self.throttle_level
                fuel_needed_this_engine = engine_consumption_rate * self.throttle_level * dt

                # Thrust direction is generally "up" relative to the rocket (-Y in Pygame)
                thrust_direction = pygame.math.Vector2(0, -1).rotate(-self.angle) # Rotate to world coords
                force_this_engine_vec = thrust_direction * thrust_this_engine

                # Accumulate total force
                total_force_vector += force_this_engine_vec

                # Calculate torque contribution: r x F
                # r = vector from CoM to engine center (in world frame)
                # F = force vector (already in world frame)
                engine_world_center = self.get_world_part_center(engine)
                world_com = self.get_world_com()
                engine_offset_from_com_world = engine_world_center - world_com

                # Accumulate for average application point calculation (using local offset for consistency)
                # engine_pos_offset_local = engine.relative_pos - self.center_of_mass_offset
                # Correct: engine pos relative to blueprint origin, rotated by world angle
                engine_pos_offset_local_rotated = engine.relative_pos.rotate(-self.angle) # Engine center offset from self.pos, in world frame

                thrust_torque_numerator += engine_pos_offset_local_rotated * thrust_this_engine # Weight by thrust
                total_thrust_magnitude_applied += thrust_this_engine

                active_engine_count_this_frame += 1
                # Request fuel (actual consumption happens in update loop)
                fuel_consumption_request[engine] = fuel_needed_this_engine
                # Tentatively mark as firing (will be set False in update if no fuel drawn)
                self.engine_firing_status[engine] = True

        # Calculate average thrust application point offset (local, rotated)
        avg_thrust_application_point_offset = None
        if active_engine_count_this_frame > 0 and total_thrust_magnitude_applied > 1e-6:
            avg_thrust_application_point_offset = thrust_torque_numerator / total_thrust_magnitude_applied

        self.thrusting = active_engine_count_this_frame > 0 # Update overall thrusting status

        return total_force_vector, avg_thrust_application_point_offset, fuel_consumption_request

    # --- Damage & Destruction Methods (Unchanged) ---
    def apply_collision_damage(self, impact_velocity_magnitude, particle_manager: ParticleManager, specific_part_to_damage: PlacedPart | None = None):
        """ Applies damage to parts based on impact velocity. Can target a specific part or distribute based on location. """
        # Ignore low-velocity impacts
        if impact_velocity_magnitude < MIN_IMPACT_VEL_DAMAGE:
            return

        # Calculate base damage (scales non-linearly with impact speed)
        base_damage = (impact_velocity_magnitude**1.8) * COLLISION_DAMAGE_FACTOR

        parts_to_damage = []
        damage_multipliers = {} # Part -> multiplier (0.0 to 1.0)

        if specific_part_to_damage and specific_part_to_damage in self.parts:
            # Damage only the specified part
            parts_to_damage = [specific_part_to_damage]
            damage_multipliers[specific_part_to_damage] = 1.0
        elif not specific_part_to_damage and self.parts:
            # Distribute damage based on which parts are "lowest" (highest Y)
            lowest_world_point = self.get_lowest_point_world()
            lowest_world_y = lowest_world_point.y
            world_com_y = self.get_world_com().y
            # Estimate the "impact height" range of the rocket
            rocket_impact_height = max(1.0, abs(lowest_world_y - world_com_y) * 1.5)

            for part in self.parts:
                part_center_y = self.get_world_part_center(part).y
                # How close is the part's center to the lowest point (0 = at lowest point)
                relative_y_from_bottom = lowest_world_y - part_center_y
                # Damage factor decreases quadratically with distance from the bottom
                # Factor = 1.0 at bottom, 0.0 far above
                damage_factor = max(0.0, min(1.0, 1.0 - (relative_y_from_bottom / rocket_impact_height)))**2
                if damage_factor > 0.01: # Apply damage if factor is significant
                    parts_to_damage.append(part)
                    damage_multipliers[part] = damage_factor

            # Fallback: If distribution calculation somehow finds no parts, damage the absolute lowest part
            if not parts_to_damage and self.parts:
                 # Find part whose center has max Y value
                 lowest_part = min(self.parts, key=lambda p: -self.get_world_part_center(p).y)
                 parts_to_damage = [lowest_part]
                 damage_multipliers[lowest_part] = 1.0

        parts_destroyed_this_impact = []
        for part in parts_to_damage:
            if part.is_broken:
                continue # Skip already broken parts

            multiplier = damage_multipliers.get(part, 0.0)
            scaled_damage = base_damage * multiplier
            if scaled_damage < 0.1: continue # Ignore trivial damage amounts

            part.current_hp -= scaled_damage
            # Check if part broke
            if part.current_hp <= 0 and not part.is_broken:
                 # print(f"  >> Part '{part.part_id}' BROKEN by impact! <<") # Debug
                 part.is_broken = True
                 part.current_hp = 0
                 parts_destroyed_this_impact.append(part)
                 # Add visual effect for destroyed part
                 particle_manager.add_explosion(self.get_world_part_center(part))

        # If parts were destroyed, trigger structural update check
        if parts_destroyed_this_impact:
            self.handle_destroyed_parts(parts_destroyed_this_impact)

    def handle_destroyed_parts(self, destroyed_parts: list[PlacedPart]):
        """ Removes destroyed parts and flags the rocket for a structural update check. """
        if not destroyed_parts:
            return # Nothing to do

        original_part_count = len(self.parts)

        # Filter out the destroyed parts
        self.parts = [p for p in self.parts if p not in destroyed_parts]

        # Check if the original root part was among the destroyed
        if self.original_root_part_ref and (self.original_root_part_ref in destroyed_parts):
            self.has_active_control = False # Lost control if root is gone
            # print(f"[{self.sim_instance_id}] Lost control: Root part destroyed.") # Debug

        # Check if the rocket is now empty
        if not self.parts:
            self.is_active = False # Deactivate the rocket if it has no parts left
            # print(f"[{self.sim_instance_id}] Deactivated: All parts destroyed.") # Debug
            return # Exit early

        # If the number of parts changed, we need to re-evaluate structure/physics
        if len(self.parts) < original_part_count:
            # --- Flag for update ---
            # This flag signals the *main simulation loop* (run_simulation)
            # that this rocket *might* have split into multiple pieces.
            # It also signals this rocket instance itself to recalculate its
            # own properties (mass, CoM, fuel map, etc.) during its next update() call.
            self.needs_structural_update = True

            # --- Internal Cleanup ---
            # Remove destroyed parts from component lists *within this instance*
            self.engines = [e for e in self.engines if e in self.parts]
            self.fuel_tanks = [t for t in self.fuel_tanks if t in self.parts]
            self.parachutes = [pc for pc in self.parachutes if pc in self.parts]
            self.separators = [s for s in self.separators if s in self.parts]
            # Reset engine firing status as structure might have changed fuel flow
            self.engine_firing_status = {e: False for e in self.engines}
            # Note: The actual *rebuild* (_build_fuel_source_map, calculate_physics_properties)
            # happens at the start of the *next* rocket.update() call if needs_structural_update is True.
            # The split check happens in run_simulation based on this flag.

    # --- Part Activation Method (Unchanged) ---
    def activate_part_at_pos(self, click_world_pos: pygame.math.Vector2):
        """ Handles activating parts (parachutes, separators, toggling engines) via clicking. """
        clicked_part: PlacedPart | None = None
        min_dist_sq = 20**2 # Click proximity tolerance (squared)

        # Find the closest non-broken part to the click
        for part in self.parts:
            if part.is_broken:
                continue
            dist_sq = (self.get_world_part_center(part) - click_world_pos).length_squared()
            if dist_sq < min_dist_sq:
                clicked_part = part
                min_dist_sq = dist_sq

        if not clicked_part:
            return False # No part clicked

        part_type = clicked_part.part_data.get("type")
        is_activatable = clicked_part.part_data.get("activatable", False)
        action_taken = False

        # Handle activatable parts (Parachutes, Separators)
        if is_activatable:
            if part_type == "Parachute":
                # Deploy parachute if not already deployed
                if not clicked_part.deployed:
                    clicked_part.deployed = True
                    # print(f"Deployed Parachute {clicked_part.part_id}!") # Debug
                    action_taken = True
            elif part_type == "Separator":
                # Activate separator if not already separated
                if not clicked_part.separated:
                    # Add to pending list for processing in the main loop
                    if clicked_part not in self.pending_separation:
                        self.pending_separation.append(clicked_part)
                        # Mark visually as activated immediately
                        clicked_part.separated = True # Visually marks as used/pending
                        # Flag for structural check in the main loop
                        self.needs_structural_update = True # Signal potential split
                        # print(f"Activated Separator {clicked_part.part_id}.") # Debug
                        action_taken = True

        # Handle non-activatable but toggleable parts (Engines)
        if not action_taken and part_type == "Engine":
            # Toggle engine enabled state
            clicked_part.engine_enabled = not clicked_part.engine_enabled
            # print(f"Toggled Engine {clicked_part.part_id} {'ON' if clicked_part.engine_enabled else 'OFF'}") # Debug
            action_taken = True

        return action_taken


    # --- Main Update Method ---
    def update(self, dt, current_air_density, particle_manager: ParticleManager):
        """ Main physics and state update loop for this rocket instance. """
        if not self.is_active or not self.parts:
            return # Don't update inactive or empty rockets

        # --- Structural Integrity Check & Recalculation ---
        # If parts were destroyed/separated in the *previous* frame,
        # the needs_structural_update flag is set. Recalculate physics
        # and fuel map *before* applying forces for *this* frame.
        if self.needs_structural_update:
             self._build_fuel_source_map() # Rebuild fuel map first
             self.calculate_physics_properties() # Then recalc mass, CoM, MoI
             self.calculate_bounds() # Update bounds
             # Reset the flag *after* internal recalculations are done
             self.needs_structural_update = False
             # Note: The check for splitting the *entire rocket* happens in run_simulation

        # --- Reset Forces/Torque for this frame ---
        self.acc = pygame.math.Vector2(0, 0)
        net_torque = 0.0
        current_world_com = self.get_world_com()
        velocity_sq = self.vel.length_squared()
        # Use safe_sqrt or check length before normalizing
        velocity_mag = math.sqrt(velocity_sq) if velocity_sq > 1e-9 else 0.0

        # --- Apply Gravity ---
        if self.total_mass > 0.01:
            gravity_force = pygame.math.Vector2(0, GRAVITY * self.total_mass)
            # Apply force directly to acceleration (F=ma => a=F/m)
            self.acc += gravity_force / self.total_mass

        # --- Calculate Potential Thrust & Fuel Needs ---
        # This gets the ideal thrust force and fuel needed, assuming fuel is available
        thrust_force_potential, thrust_app_local_offset_rotated, fuel_consumption_request = self.get_thrust_and_consumption(dt)

        # --- Consume Fuel ---
        total_fuel_drawn_this_frame = 0.0
        engines_actually_fired = set() # Track engines that successfully drew fuel
        mass_changed_by_fuel = False # Flag if mass changed due to fuel use

        # Only process if some engine requested fuel
        if fuel_consumption_request:
            for engine, fuel_needed in fuel_consumption_request.items():
                # Skip if this engine needs no fuel (e.g., throttle 0 but enabled)
                if fuel_needed <= 1e-9: # Use tolerance for float comparison
                    self.engine_firing_status[engine] = False # Ensure status is false
                    continue

                # Find valid tanks for this engine
                tanks = self.engine_fuel_sources.get(engine, [])
                valid_tanks = [t for t in tanks if not t.is_broken and t.current_fuel > 1e-9]
                available_fuel_for_engine = sum(tank.current_fuel for tank in valid_tanks)

                # Determine how much fuel can actually be drawn
                actual_fuel_to_draw = min(fuel_needed, available_fuel_for_engine)

                if actual_fuel_to_draw > 1e-9: # Check if any fuel can be drawn
                    fuel_drawn_this_engine = 0.0
                    # Draw fuel proportionally from available tanks
                    if available_fuel_for_engine > 1e-9: # Avoid division by zero
                        for tank in valid_tanks:
                            # Calculate proportion of available fuel this tank holds
                            proportion = tank.current_fuel / available_fuel_for_engine
                            # Amount to draw from this tank
                            draw_amount = actual_fuel_to_draw * proportion
                            # Clamp draw amount to what the tank actually has
                            draw_amount = min(draw_amount, tank.current_fuel)
                            # Subtract fuel
                            tank.current_fuel -= draw_amount
                            fuel_drawn_this_engine += draw_amount
                            # Ensure fuel doesn't go negative due to float errors
                            tank.current_fuel = max(0.0, tank.current_fuel)

                    total_fuel_drawn_this_frame += fuel_drawn_this_engine
                    mass_changed_by_fuel = True # Mass needs recalculation
                    engines_actually_fired.add(engine)
                    # Keep engine_firing_status as True (set during get_thrust...)
                else:
                    # Not enough fuel, engine didn't actually fire
                    self.engine_firing_status[engine] = False # Update status

            # If fuel was consumed, recalculate physics properties immediately
            # This ensures forces are applied using the correct mass for this frame
            if mass_changed_by_fuel:
                self.calculate_physics_properties()
                # CoM might have shifted, get the updated world CoM for torque calculations
                current_world_com = self.get_world_com()

        # --- Apply Actual Thrust Force & Torque ---
        # We apply the *potential* thrust calculated earlier, but only if the engine
        # actually fired (drew fuel). Torque depends on this actual force.
        actual_thrust_force_this_frame = pygame.math.Vector2(0, 0)
        actual_thrust_torque_numerator = pygame.math.Vector2(0, 0)
        actual_total_thrust_magnitude = 0.0

        for engine in engines_actually_fired: # Iterate only engines that got fuel
            engine_thrust_potential = engine.part_data.get("thrust", 0)
            thrust_this_engine = engine_thrust_potential * self.throttle_level
            thrust_direction = pygame.math.Vector2(0, -1).rotate(-self.angle)
            force_this_engine_vec = thrust_direction * thrust_this_engine
            actual_thrust_force_this_frame += force_this_engine_vec

            # Calculate torque contribution using the same method as get_thrust...
            engine_pos_offset_local_rotated = engine.relative_pos.rotate(-self.angle)
            actual_thrust_torque_numerator += engine_pos_offset_local_rotated * thrust_this_engine
            actual_total_thrust_magnitude += thrust_this_engine


        # Apply net thrust force to acceleration
        if self.total_mass > 0.01:
            self.acc += actual_thrust_force_this_frame / self.total_mass

        # Apply net thrust torque
        if actual_total_thrust_magnitude > 1e-6:
            # Calculate average application point offset (local rotated) for the *actual* thrust
            avg_thrust_app_local_offset_rotated = actual_thrust_torque_numerator / actual_total_thrust_magnitude
            # Calculate torque: r x F
            # r = vector from CoM to average thrust application point (world)
            # F = total actual thrust force vector (world)
            thrust_app_offset_from_com = (self.pos + avg_thrust_app_local_offset_rotated) - current_world_com
            net_torque += thrust_app_offset_from_com.cross(actual_thrust_force_this_frame)


        # --- Aerodynamics & Heating (Largely Unchanged) ---
        total_drag_force = pygame.math.Vector2(0, 0)
        parts_destroyed_by_heat = []
        current_max_temp = AMBIENT_TEMPERATURE # Track max temp across parts this frame

        for part in self.parts:
            if part.is_broken: continue # Skip broken parts

            part_data = part.part_data
            part_world_center = self.get_world_part_center(part)
            drag_force_on_part = pygame.math.Vector2(0, 0)
            # TODO: More sophisticated aero model needed here
            # Simple approximation based on avg dimension and base drag coeff
            effective_area = (part_data['width'] + part_data['height']) / 2.0 * 0.1 # Arbitrary scaling factor
            drag_coeff = part_data['base_drag_coeff']

            # Add parachute drag if deployed and active
            is_parachute = part_data['type'] == 'Parachute'
            parachute_active = is_parachute and part.deployed and not self.landed # Chutes don't work well on ground
            if parachute_active:
                drag_coeff += part_data['deploy_drag']
                effective_area += (part_data['width'] * part_data['deploy_area_factor']) * 0.1 # Add deployed area

            # Calculate drag force if moving through air
            if current_air_density > AIR_DENSITY_VACUUM and velocity_sq > 0.1 and effective_area > 0 and drag_coeff > 0:
                drag_magnitude = 0.5 * current_air_density * velocity_sq * effective_area * drag_coeff
                if velocity_mag > 0.01: # Avoid normalize(0)
                     drag_direction = -self.vel.normalize()
                     drag_force_on_part = drag_direction * drag_magnitude
                     total_drag_force += drag_force_on_part
                     # Apply torque from drag force acting on the part's center
                     drag_app_offset_from_com = part_world_center - current_world_com
                     net_torque += drag_app_offset_from_com.cross(drag_force_on_part)

            # --- Thermal Update ---
            heat_generated = 0.0
            # Aerodynamic Heating (simplified)
            if effective_area > 0 and velocity_mag > 50: # Only significant heating at higher speeds
                # Heating proportional to density * velocity^3 * area
                heat_generated = AERO_HEATING_FACTOR * current_air_density * (velocity_mag**3) * effective_area * dt

            # Cooling (Radiation + Convection)
            temp_diff = part.current_temp - AMBIENT_TEMPERATURE # Difference from ambient
            # Cooling rate depends on base radiation and air density convection
            cooling_rate = HEAT_DISSIPATION_FACTOR_VACUUM + HEAT_DISSIPATION_FACTOR_CONVECTION * current_air_density
            # Cooling proportional to temp diff, rate, and surface area (approximated)
            part_surface_area_approx = (part_data['width'] * part_data['height']) * 0.01 # Area scale factor
            heat_lost = cooling_rate * temp_diff * dt * part_surface_area_approx

            # Temperature Change
            thermal_mass = part_data['thermal_mass'] # J/K
            if thermal_mass > 1e-6: # Avoid division by zero
                # Delta Temp (K) = Heat Change (J) / Thermal Mass (J/K)
                delta_temp = (heat_generated - heat_lost) / thermal_mass
                part.current_temp += delta_temp
                # Clamp temperature to ambient minimum
                part.current_temp = max(AMBIENT_TEMPERATURE, part.current_temp)

            # Check for Overheating Effects and Damage
            max_temp = part_data['max_temp']
            # Flag for visual reentry/overheat effects
            part.is_overheating = part.current_temp > REENTRY_EFFECT_THRESHOLD_TEMP

            # Check if part exceeds its maximum temperature
            if part.current_temp > max_temp:
                overheat_amount = part.current_temp - max_temp
                # Damage scales with how much it's over max temp (relative to a threshold range)
                damage_factor = max(0.0, overheat_amount / OVERHEAT_DAMAGE_THRESHOLD_K)
                # Apply damage over time based on overheat factor
                damage = OVERHEAT_DAMAGE_RATE * damage_factor * dt
                part.current_hp -= damage

                # Check if part broke due to heat
                if part.current_hp <= 0 and not part.is_broken:
                    # print(f"  >> Part '{part.part_id}' DESTROYED by overheating! (T={part.current_temp:.0f}K) <<") # Debug
                    part.is_broken = True
                    part.current_hp = 0
                    parts_destroyed_by_heat.append(part)
                    # Add visual explosion effect
                    particle_manager.add_explosion(part_world_center)

            # Track the maximum temperature across all parts for this frame (UI display)
            current_max_temp = max(current_max_temp, part.current_temp)

        # Store max temp for UI
        self.max_temp_reading = current_max_temp

        # Apply Net Drag Force to acceleration
        if self.total_mass > 0.01:
            self.acc += total_drag_force / self.total_mass

        # Handle parts destroyed by heat *after* iterating through all parts
        if parts_destroyed_by_heat:
            self.handle_destroyed_parts(parts_destroyed_by_heat)
            # Structure might have changed, physics/fuel map will update next frame start
            # Split check will happen in run_simulation

        # --- Control Input (Reaction Wheels) ---
        control_torque = 0.0
        # Check if the designated root part still exists and isn't broken
        root_ref = self.original_root_part_ref
        self.has_active_control = (root_ref is not None) and \
                                  (root_ref in self.parts) and \
                                  (not root_ref.is_broken)

        # Only apply control torque if the rocket is controllable
        if self.has_active_control:
            keys = pygame.key.get_pressed()
            # Apply reaction wheel torque for rotation
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                control_torque += REACTION_WHEEL_TORQUE # Counter-clockwise torque
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                control_torque -= REACTION_WHEEL_TORQUE # Clockwise torque

        # Add control torque to the net torque
        net_torque += control_torque

        # --- Physics Integration (Update motion based on forces/torques) ---
        # Update velocity based on total acceleration
        self.vel += self.acc * dt
        # Update position based on new velocity
        self.pos += self.vel * dt

        # Update angular velocity based on net torque and moment of inertia
        if self.moment_of_inertia > 1e-6: # Avoid division by zero
            # Angular acceleration (rad/s^2) = Torque (Nm) / MoI (kg*m^2)
            angular_acceleration_rad = net_torque / self.moment_of_inertia
            # Convert to degrees/s^2 for internal angle representation
            angular_acceleration_deg = math.degrees(angular_acceleration_rad)
            self.angular_velocity += angular_acceleration_deg * dt

        # Apply angular damping (simulates friction, air resistance affecting rotation)
        self.angular_velocity *= (1.0 - ANGULAR_DAMPING * dt)
        # Update angle based on new angular velocity
        self.angle = (self.angle + self.angular_velocity * dt) % 360 # Keep angle between 0-360

        # --- Ground Collision / Landing Logic ---
        # Check if the lowest point is at or below ground level
        lowest_point = self.get_lowest_point_world()
        currently_on_ground = lowest_point.y >= GROUND_Y

        just_landed = currently_on_ground and not self.was_landed_last_frame
        just_took_off = not currently_on_ground and self.was_landed_last_frame

        # Handle landing impact damage
        if just_landed:
            impact_vel_mag = self.vel.length()
            # Apply damage if impact speed is high enough
            # Use slightly lower threshold if vertical velocity is significant part of impact
            vertical_impact_threshold = MIN_IMPACT_VEL_DAMAGE * 0.5
            should_apply_damage = (self.vel.y > 1.0 and impact_vel_mag >= vertical_impact_threshold) or \
                                  (impact_vel_mag >= MIN_IMPACT_VEL_DAMAGE)
            if should_apply_damage:
                # Apply damage, potentially distributed to lower parts
                self.apply_collision_damage(impact_vel_mag, particle_manager, specific_part_to_damage=None)

        # Handle physics when on the ground
        if currently_on_ground:
            self.landed = True
            # Only apply ground physics if the rocket wasn't destroyed by the impact
            if self.is_active and self.parts:
                # Correct position to prevent sinking below ground
                correction = lowest_point.y - GROUND_Y
                if correction > 0:
                    self.pos.y -= correction # Move rocket origin up
                # Stop vertical velocity completely
                self.vel.y = 0
                # Apply ground friction to horizontal velocity
                self.vel.x *= 0.6
                # Stop rotation completely when landed
                self.angular_velocity = 0
            else:
                # If destroyed on impact, just zero out velocity/rotation
                 self.vel = pygame.math.Vector2(0,0)
                 self.angular_velocity = 0
        else:
            # Not on the ground
            self.landed = False

        # Handle taking off with deployed parachutes (they get ripped off)
        if just_took_off:
            destroyed_chutes_on_takeoff = []
            for chute in self.parachutes:
                # If chute was deployed and isn't already broken
                if chute.deployed and not chute.is_broken:
                    # print(f"  >> Parachute {chute.part_id} destroyed by takeoff! <<") # Debug
                    chute.is_broken = True
                    chute.deployed = False # No longer considered deployed visually/aerodynamically
                    chute.current_hp = 0
                    destroyed_chutes_on_takeoff.append(chute)
                    # Add small visual effect for ripped chute
                    particle_manager.add_explosion(self.get_world_part_center(chute), num_particles=5, max_life=0.3, max_speed=50)
            # Handle destruction immediately if chutes broke
            if destroyed_chutes_on_takeoff:
                self.handle_destroyed_parts(destroyed_chutes_on_takeoff)
                # Sets needs_structural_update flag for next frame

        # Update landed state for next frame's transition check
        self.was_landed_last_frame = self.landed

    # --- Drawing Method (Unchanged) ---
    def draw(self, surface, camera, particle_manager: ParticleManager):
        """ Draws the rocket parts and effects (flames, reentry). Returns count of visually broken parts. """
        num_broken_visually = 0
        if not self.is_active: return 0 # Don't draw inactive rockets

        # --- Reentry/Overheating Effects ---
        spawn_reentry_particles = False
        hottest_part_temp = AMBIENT_TEMPERATURE
        hottest_part_max_temp = AMBIENT_TEMPERATURE # Max allowable temp on any part
        has_overheating_parts = False # Any part currently triggering visual overheat?

        # Find max temp and check for overheating parts
        for part in self.parts:
             if not part.is_broken:
                 # Find the highest allowable temperature on the current assembly
                 hottest_part_max_temp = max(hottest_part_max_temp, part.part_data.get('max_temp', DEFAULT_MAX_TEMP))
                 # Check if this part is visually overheating
                 if part.is_overheating:
                     spawn_reentry_particles = True
                     hottest_part_temp = max(hottest_part_temp, part.current_temp)
                     has_overheating_parts = True

        # Spawn reentry particle effects if needed
        if spawn_reentry_particles and self.vel.length_squared() > 50**2: # Only if moving fast
             # Calculate intensity based on how hot the hottest part is relative to visual threshold/max
             reentry_temp_range = max(1.0, (hottest_part_max_temp * REENTRY_EFFECT_MAX_TEMP_SCALE) - REENTRY_EFFECT_THRESHOLD_TEMP)
             intensity = max(0.0, min(1.0, (hottest_part_temp - REENTRY_EFFECT_THRESHOLD_TEMP) / reentry_temp_range ))
             # Spawn particles near the leading edge (lowest point)
             leading_point = self.get_lowest_point_world()
             num_sparks = int(1 + intensity * 5) # More sparks for higher intensity
             for _ in range(num_sparks):
                 # Add randomness to spawn position slightly
                 spark_pos = leading_point + pygame.math.Vector2(random.uniform(-5, 5), random.uniform(-5, 5))
                 particle_manager.add_reentry_spark(spark_pos, self.vel, intensity)

        # --- Draw Individual Parts ---
        for part in self.parts:
            part_center_world = self.get_world_part_center(part)
            part_screen_pos = camera.apply(part_center_world)
            part_world_angle = self.angle # Use the assembly's angle
            indicator_color = None # Color for status indicator circle
            is_activatable = part.part_data.get("activatable", False)
            part_type = part.part_data.get("type")
            is_parachute = part_type == "Parachute"
            is_engine = part_type == "Engine"
            is_separator = part_type == "Separator"
            # Determine visual deployed state (parachutes visually collapse on ground)
            show_deployed_visual = is_parachute and part.deployed and not part.is_broken and not self.landed
            heat_factor = 0.0 # Glow factor for overheating (0 to 1)

            # Calculate heat glow factor
            if part.is_overheating and not part.is_broken:
                # Scale glow based on current temp relative to visual threshold and max temp
                max_temp_visual = part.part_data.get('max_temp', DEFAULT_MAX_TEMP) * REENTRY_EFFECT_MAX_TEMP_SCALE
                heat_glow_range = max(1.0, max_temp_visual - REENTRY_EFFECT_THRESHOLD_TEMP)
                heat_factor = max(0.0, min(1.0, (part.current_temp - REENTRY_EFFECT_THRESHOLD_TEMP) / heat_glow_range))

            # Draw the part shape itself
            if part.is_broken:
                num_broken_visually += 1
            # Draw status indicators for certain part types
            elif is_parachute and is_activatable:
                indicator_color = COLOR_ACTIVATABLE_USED if part.deployed else COLOR_ACTIVATABLE_READY
            elif is_engine: # Engines are toggleable but not "activatable" in the same sense
                indicator_color = COLOR_ENGINE_ENABLED if part.engine_enabled else COLOR_ENGINE_DISABLED
            elif is_separator and is_activatable:
                indicator_color = COLOR_ACTIVATABLE_USED if part.separated else COLOR_ACTIVATABLE_READY # Used means pending/activated

            # Draw the main part shape (rectangle, potentially with effects)
            try:
                draw_part_shape(surface, part.part_data, part_screen_pos, part_world_angle,
                                broken=part.is_broken, deployed=show_deployed_visual, heat_factor=heat_factor)
            except NameError: # Fallback if draw_part_shape isn't imported correctly
                fallback_color = RED if part.is_broken else GREEN
                pygame.draw.circle(surface, fallback_color, part_screen_pos, 5)

            # Draw the status indicator circle on top
            if indicator_color:
                pygame.draw.circle(surface, indicator_color, part_screen_pos, 5)
                pygame.draw.circle(surface, BLACK, part_screen_pos, 5, 1) # Outline


        # --- Draw Engine Flames ---
        if self.thrusting: # Check the overall thrusting flag first
            flame_scale = 0.5 + 0.5 * self.throttle_level # Flame size based on throttle
            for engine, is_firing in self.engine_firing_status.items():
                # Draw flame only if this specific engine actually fired
                if is_firing and not engine.is_broken:
                    engine_center_world = self.get_world_part_center(engine)
                    engine_world_angle = self.angle
                    # Calculate flame base position (bottom center of engine)
                    flame_base_offset_local = pygame.math.Vector2(0, engine.part_data["height"] / 2.0) # Offset from engine center
                    flame_base_offset_rotated = flame_base_offset_local.rotate(-engine_world_angle)
                    flame_base_world = engine_center_world + flame_base_offset_rotated
                    # Calculate flame shape parameters
                    flame_length = (15 + random.uniform(-2, 2)) * flame_scale # Add flicker
                    flame_width = engine.part_data["width"] * 0.8 * flame_scale
                    # Flame points relative to base
                    flame_dir_world = pygame.math.Vector2(0, 1).rotate(-engine_world_angle) # Points "down" from rocket
                    flame_side_world = pygame.math.Vector2(1, 0).rotate(-engine_world_angle) # Points sideways
                    flame_tip_world = flame_base_world + flame_dir_world * flame_length
                    flame_left_world = flame_base_world - flame_side_world * flame_width / 2.0
                    flame_right_world = flame_base_world + flame_side_world * flame_width / 2.0
                    # Convert to screen coordinates
                    flame_points_screen = [camera.apply(p) for p in [flame_left_world, flame_right_world, flame_tip_world]]
                    # Draw the flame polygon
                    try:
                        pygame.draw.polygon(surface, COLOR_FLAME, flame_points_screen)
                    except NameError: # Fallback if COLOR_FLAME missing
                        pygame.draw.line(surface, RED, camera.apply(flame_base_world), camera.apply(flame_tip_world), 3)
                    except ValueError: # Handle potential errors if points are invalid
                        pass # Optionally log error


        return num_broken_visually

    # --- Helper for calculating CoM of a subassembly (used during splits) ---
    def calculate_subassembly_world_com(self, assembly_parts: list[PlacedPart]) -> pygame.math.Vector2:
        """ Calculates the center of mass (world coordinates) for a given list of parts that *belong to this rocket*. """
        if not assembly_parts:
            # Should not happen if called correctly, but provide a fallback
            print(f"Warning: calculate_subassembly_world_com called with empty list for rocket {self.sim_instance_id}. Returning self.pos.")
            return self.pos

        com_numerator = pygame.math.Vector2(0, 0)
        total_assembly_mass = 0.0 # Use float for mass

        for part in assembly_parts:
            # Ensure part has data and mass
            if not part.part_data:
                print(f"Warning: Part {part.part_id} in assembly calculation has no part_data.")
                continue # Skip this part

            # Use DRY mass for structural CoM calculation during splits
            # (Fuel distribution post-split is complex, simpler to use dry mass)
            part_mass_static = float(part.part_data.get("mass", 0.0))
            if part_mass_static <= 1e-6: # Skip effectively massless parts
                 continue

            # Get the current world center of the part *before* the split
            part_world_center = self.get_world_part_center(part)
            com_numerator += part_world_center * part_mass_static
            total_assembly_mass += part_mass_static

        # Handle cases where total mass is zero or negligible
        if total_assembly_mass <= 1e-6:
            # Return the world center of the first part as a reasonable estimate
            first_part_center = self.get_world_part_center(assembly_parts[0])
            print(f"Warning: Total mass for subassembly of rocket {self.sim_instance_id} is near zero. Returning first part center: {first_part_center}")
            return first_part_center
        else:
            # Calculate and return the world center of mass
            return com_numerator / total_assembly_mass

    def get_total_current_fuel(self) -> float:
        """ Calculates the total current fuel across all non-broken tanks in this assembly. """
        return sum(tank.current_fuel for tank in self.fuel_tanks if not tank.is_broken)


# --- Background/Terrain Functions (Unchanged) ---
def create_stars(count, bounds: pygame.Rect):
    """ Creates a list of stars with random positions and depths within bounds. """
    stars = []
    # Depth range affects parallax effect (larger z means further away)
    depth_range = bounds.height # Tie depth somewhat to the vertical extent
    for _ in range(count):
        x = random.uniform(bounds.left, bounds.right)
        y = random.uniform(bounds.top, bounds.bottom)
        # Ensure z is at least 1 to avoid division by zero in parallax
        z = random.uniform(1, max(2, depth_range))
        stars.append((pygame.math.Vector2(x, y), z))
    return stars

def get_air_density(altitude_agl):
    """ Calculates approximate air density based on altitude above ground level (AGL). """
    scale_height = ATMOSPHERE_SCALE_HEIGHT # Altitude where density drops by 1/e (~8.5km for Earth)
    if altitude_agl < 0:
        # Below ground? Use sea level density.
        return AIR_DENSITY_SEA_LEVEL
    elif 0 <= altitude_agl <= ATMOSPHERE_EXP_LIMIT:
        # Exponential decay up to 35km
        density = AIR_DENSITY_SEA_LEVEL * math.exp(-altitude_agl / scale_height)
        return max(AIR_DENSITY_VACUUM, density)
    elif ATMOSPHERE_EXP_LIMIT < altitude_agl <= ATMOSPHERE_LINEAR_LIMIT:
        # Linear interpolation from 35km density down to target density at 70km
        density_at_35k = AIR_DENSITY_SEA_LEVEL * math.exp(-ATMOSPHERE_EXP_LIMIT / scale_height)
        density_at_35k = max(AIR_DENSITY_VACUUM, density_at_35k) # Clamp lower bound
        # Target density very low at 70km limit
        density_at_70k_target = AIR_DENSITY_SEA_LEVEL * ATMOSPHERE_TARGET_DENSITY_FACTOR
        # Interpolation factor (0 at 35km, 1 at 70km)
        interp_factor = max(0.0, min(1.0, (altitude_agl - ATMOSPHERE_EXP_LIMIT) / (ATMOSPHERE_LINEAR_LIMIT - ATMOSPHERE_EXP_LIMIT)))
        density = density_at_35k * (1.0 - interp_factor) + density_at_70k_target * interp_factor
        return max(AIR_DENSITY_VACUUM, density)
    else:
        # Above 70km, assume vacuum
        return AIR_DENSITY_VACUUM

def draw_earth_background(surface, camera, stars):
    """ Draws the sky gradient (blue to black) and stars based on camera altitude. """
    screen_rect = surface.get_rect()
    # Use camera center Y to determine altitude for background color
    # camera.offset.y is the world Y coord at the top of the screen
    avg_world_y = camera.offset.y + camera.height / 2 # World Y at screen center
    ground_screen_y = camera.apply(pygame.math.Vector2(0, GROUND_Y)).y # Ground position on screen

    # Define transition altitudes and colors (ensure defined, use fallbacks if needed)
    try: _ = BLUE_SKY_Y_LIMIT; _ = SPACE_Y_LIMIT; _ = COLOR_SKY_BLUE; _ = COLOR_SPACE_BLACK
    except NameError: BLUE_SKY_Y_LIMIT=-2000; SPACE_Y_LIMIT=-15000; COLOR_SKY_BLUE=(135,206,250); COLOR_SPACE_BLACK=(0,0,0)

    # Determine background color based on average world Y
    if avg_world_y > BLUE_SKY_Y_LIMIT:
        # Mostly in atmosphere, draw blue sky above ground
        if ground_screen_y < screen_rect.bottom:
            # Draw sky rect only down to the ground line if visible
             pygame.draw.rect(surface, COLOR_SKY_BLUE, (0, 0, screen_rect.width, ground_screen_y))
        else:
             # Ground is off-screen below, fill whole screen with blue
             surface.fill(COLOR_SKY_BLUE)
    elif avg_world_y < SPACE_Y_LIMIT:
        # High in space, draw black background and stars
        surface.fill(COLOR_SPACE_BLACK)
        draw_stars(surface, stars, camera, alpha=255) # Full brightness stars
    else:
        # Transition zone: Interpolate color between blue sky and space black
        interp = max(0.0, min(1.0, (avg_world_y - BLUE_SKY_Y_LIMIT) / (SPACE_Y_LIMIT - BLUE_SKY_Y_LIMIT)))
        bg_color = pygame.Color(COLOR_SKY_BLUE).lerp(COLOR_SPACE_BLACK, interp)
        surface.fill(bg_color)
        # Fade stars in during transition
        star_alpha = int(255 * interp)
        if star_alpha > 10: # Only draw stars if they are somewhat visible
            draw_stars(surface, stars, camera, alpha=star_alpha)

def draw_stars(surface, stars, camera, alpha=255):
    """ Draws stars with parallax effect based on camera offset and star depth. """
    if alpha <= 0: return # Skip drawing if fully transparent
    screen_rect = surface.get_rect()
    base_color = pygame.Color(200, 200, 200) # Base star color

    try:
        depth_scaling = STAR_FIELD_DEPTH # Max depth for scaling parallax
    except NameError:
        depth_scaling = 10000 # Fallback value

    for world_pos, z in stars:
        # Calculate parallax factor (closer stars (smaller z) move less relative to camera)
        # Parallax = 1 / (scaled_depth + 1). Factor is 0.0 to 1.0
        # Smaller z => larger parallax factor => moves MORE with camera (appears closer) - This seems backwards?
        # Let's try: Further stars (larger z) should move LESS with camera (parallax factor closer to 0)
        # parallax_factor = 1.0 - (z / depth_scaling) # Incorrect?
        # Let's use the common 1 / (depth + constant) approach
        # Larger z => larger denominator => smaller parallax factor
        parallax_factor = 1.0 / ((z / (depth_scaling / 20.0)) + 1.0) # Adjust scaling as needed

        # Apply parallax to camera offset
        effective_camera_offset = camera.offset * parallax_factor
        # Calculate screen position
        screen_pos = world_pos - effective_camera_offset
        sx, sy = int(screen_pos.x), int(screen_pos.y)

        # Basic culling
        if 0 <= sx < screen_rect.width and 0 <= sy < screen_rect.height:
            # Determine star size based on depth (further stars are smaller)
            size = max(1, int(2.5 * (1.0 - z / max(1, depth_scaling)))) # Ensure size is at least 1
            # Apply alpha transparency
            alpha_factor = alpha / 255.0
            final_color_tuple = (
                int(base_color.r * alpha_factor),
                int(base_color.g * alpha_factor),
                int(base_color.b * alpha_factor)
            )
            # Draw star if color is not black
            if final_color_tuple != (0,0,0):
                 try:
                     pygame.draw.circle(surface, final_color_tuple, (sx, sy), size)
                 except ValueError: # Catch errors if color/pos/size are invalid
                     pass # Optionally log error

def draw_terrain(surface, camera):
    """ Draws a simple flat ground plane. """
    world_width = WORLD_WIDTH
    ground_y = GROUND_Y
    ground_color = COLOR_GROUND

    # Define a large rectangle representing the ground area visible
    # Extend horizontally beyond camera view to ensure it fills screen during rotation/movement
    view_rect_world = pygame.Rect(
        camera.offset.x - world_width,  # Start well left of camera
        ground_y,                       # Top edge is the ground level
        camera.width + world_width * 2, # Width covers camera view + extensions
        SCREEN_HEIGHT * 2               # Height ensures it covers bottom of screen
    )
    # Convert world rect to screen coordinates
    rect_screen = camera.apply_rect(view_rect_world)
    # Draw the ground rectangle
    pygame.draw.rect(surface, ground_color, rect_screen)


# --- Simulation Runner Function ---
# <<< MODIFIED to handle collision grace period >>>
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

    # --- Simulation State Variables ---
    all_rockets: list[FlyingRocket] = [] # List to hold all active rocket assemblies
    controlled_rocket: FlyingRocket | None = None # The assembly currently under player control
    next_sim_id = 0 # Counter for unique rocket instance IDs

    # *** NEW: Collision Grace Period Tracking ***
    # Stores pairs of rocket sim_instance_ids that should ignore collisions for a short time
    # Format: {(id1, id2): frames_remaining} where id1 < id2
    collision_grace_period_pairs: dict[tuple[int, int], int] = {}
    current_sim_frame = 0 # Frame counter for grace period

    # --- Initial Rocket Setup ---
    # Find the original root part (CommandPod preferably) from the blueprint
    original_root_part_instance = None
    if initial_blueprint.parts:
        # Search for CommandPod first
        for part in initial_blueprint.parts:
             if part.part_data and part.part_data.get("type") == "CommandPod":
                 original_root_part_instance = part
                 break
        # Fallback to the first part if no command pod found
        if not original_root_part_instance and initial_blueprint.parts:
            original_root_part_instance = initial_blueprint.parts[0]
            # print("Warning: No Command Pod found, using first part as root reference.") # Less verbose

    # Find connected components in the blueprint initially
    initial_subassemblies = initial_blueprint.find_connected_subassemblies()
    if not initial_subassemblies:
         print("Error: No connected parts found after initial connectivity check.")
         return

    initial_spawn_y_offset = 0 # Spawn directly ON the ground for simplicity

    # Create FlyingRocket instances for each initial subassembly
    for i, assembly_parts in enumerate(initial_subassemblies):
        if not assembly_parts: continue # Skip empty assemblies

        # Calculate initial position to place the assembly CoM on the ground plane
        # Create a temporary blueprint just for calculating properties of this subassembly
        temp_bp_for_calc = RocketBlueprint()
        temp_bp_for_calc.parts = assembly_parts
        # CoM relative to the blueprint's (0,0) origin
        initial_com_local = temp_bp_for_calc.calculate_subassembly_world_com(assembly_parts)
        # Find the lowest point's Y offset relative to the blueprint's (0,0) origin
        lowest_offset_y = temp_bp_for_calc.get_lowest_point_offset_y()

        # Determine spawn position
        start_x = i * 50 # Stagger initial positions horizontally slightly
        # Calculate the world Y for the blueprint's origin (0,0) so the lowest point lands at GROUND_Y
        start_y_for_origin = (GROUND_Y - initial_spawn_y_offset) - lowest_offset_y
        # Target world position for the *Center of Mass* of this subassembly
        target_initial_com_pos = pygame.math.Vector2(start_x, start_y_for_origin + initial_com_local.y)

        # Determine if this assembly contains the original root part
        contains_original_root = original_root_part_instance and (original_root_part_instance in assembly_parts)
        # Assign primary control to the first assembly containing the root, or just the first assembly
        is_primary = (controlled_rocket is None and contains_original_root) or \
                     (controlled_rocket is None and i == 0) # Fallback

        try:
            # Create the FlyingRocket instance
            # Pass the list of parts (they retain their state like HP, temp, fuel from the blueprint load)
            # Pass the calculated target CoM position
            # Pass the current simulation frame for potential age tracking (used by grace period indirectly)
            rocket_instance = FlyingRocket(
                parts_list=list(assembly_parts), # Pass a copy of the list
                initial_world_com_pos=target_initial_com_pos,
                initial_angle=0,
                initial_vel=pygame.math.Vector2(0,0), # Start stationary
                sim_instance_id=next_sim_id,
                is_primary_control=is_primary,
                original_root_ref=original_root_part_instance,
                current_frame=current_sim_frame # Pass current frame
            )

            all_rockets.append(rocket_instance)

            if is_primary:
                 controlled_rocket = rocket_instance
                 # Double-check control status based on root part health
                 root_ref_in_instance = controlled_rocket.original_root_part_ref
                 controlled_rocket.has_active_control = (root_ref_in_instance is not None) and \
                                                        (root_ref_in_instance in controlled_rocket.parts) and \
                                                        (not root_ref_in_instance.is_broken)
                 control_status_msg = 'CONTROLLED' if controlled_rocket.has_active_control else 'NO CONTROL (Root Missing/Broken)'
                 # print(f"Created initial rocket {next_sim_id} ({control_status_msg}) with {len(assembly_parts)} parts.") # Debug
            else:
                 # Ensure non-primary rockets start without control flag
                 rocket_instance.has_active_control = False
                 # print(f"Created initial rocket {next_sim_id} (DEBRIS/UNCONTROLLED) with {len(assembly_parts)} parts.") # Debug

            next_sim_id += 1 # Increment for the next rocket instance

        except Exception as e:
            print(f"Error initializing rocket instance {next_sim_id}: {e}")
            import traceback
            traceback.print_exc()


    # Fallback control assignment if primary failed (e.g., root part was in a later assembly)
    if controlled_rocket is None and all_rockets:
         # print("Warning: No primary control assigned initially. Assigning fallback control to first rocket.") # Debug
         controlled_rocket = all_rockets[0]
         # Check if this fallback rocket actually has the root part and can be controlled
         root_ref_in_fallback = controlled_rocket.original_root_part_ref
         controlled_rocket.has_active_control = (root_ref_in_fallback is not None) and \
                                               (root_ref_in_fallback in controlled_rocket.parts) and \
                                               (not root_ref_in_fallback.is_broken)


    # --- Simulation Setup ---
    camera = Camera(SCREEN_WIDTH, SCREEN_HEIGHT)
    # Initialize camera position centered on the controlled rocket or the first rocket
    if controlled_rocket:
        camera.update(controlled_rocket.get_world_com())
    elif all_rockets:
        camera.update(all_rockets[0].get_world_com())
    else:
        camera.update(pygame.math.Vector2(0, GROUND_Y - SCREEN_HEIGHT // 3)) # Default view if no rockets

    # Create star field
    try:
        # Define bounds for star generation (make it larger than typical view)
        star_area_bounds = pygame.Rect(
            -WORLD_WIDTH*2,
            SPACE_Y_LIMIT - STAR_FIELD_DEPTH, # Start well above space limit
            WORLD_WIDTH*4, # Wide area
            abs(SPACE_Y_LIMIT) + GROUND_Y + STAR_FIELD_DEPTH * 1.5 # Large vertical extent
        )
        stars = create_stars(STAR_COUNT*2, star_area_bounds)
    except NameError:
        stars = [] # Fallback if constants missing

    # UI Fonts and Particle Manager
    ui_font = pygame.font.SysFont(None, 20)
    ui_font_large = pygame.font.SysFont(None, 36)
    particle_manager = ParticleManager()

    sim_running = True
    last_respawn_time = time.time() # Cooldown for respawn key

    # --- Main Simulation Loop ---
    while sim_running:
        dt = clock.tick(60) / 1000.0 # Delta time in seconds
        dt = min(dt, 0.05) # Clamp delta time to prevent large physics steps
        current_sim_frame += 1 # Increment frame counter

        # Lists for managing rocket creation/deletion during the frame
        newly_created_rockets_this_frame: list[FlyingRocket] = []
        rockets_to_remove_this_frame: list[FlyingRocket] = []

        # --- Update Collision Grace Period ---
        pairs_to_remove_from_grace = []
        # Iterate through a copy of the keys to allow modification during iteration
        grace_pairs_current = list(collision_grace_period_pairs.keys())
        for pair_key in grace_pairs_current:
            # Check if the key still exists (it might have been removed if a rocket was deleted)
            if pair_key in collision_grace_period_pairs:
                collision_grace_period_pairs[pair_key] -= 1 # Decrement frames remaining
                # If grace period expired, mark pair for removal
                if collision_grace_period_pairs[pair_key] <= 0:
                    pairs_to_remove_from_grace.append(pair_key)

        # Remove expired pairs
        for pair_key in pairs_to_remove_from_grace:
            # Check again before deleting, as it might have been removed by rocket deletion logic
            if pair_key in collision_grace_period_pairs:
                del collision_grace_period_pairs[pair_key]
                # print(f"Grace period ended for pair: {pair_key}") # Debug


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
                    # Toggle Master Thrust Enable
                    if event.key == pygame.K_SPACE:
                        controlled_rocket.master_thrust_enabled = not controlled_rocket.master_thrust_enabled

                    # Deploy Parachutes (Key P) - Activates all ready chutes on controlled vessel
                    if event.key == pygame.K_p:
                        chutes_deployed_this_press = 0
                        for chute in controlled_rocket.parachutes:
                            # Check if parachute is ready (not deployed, not broken)
                            if not chute.deployed and not chute.is_broken:
                                chute.deployed = True
                                chutes_deployed_this_press += 1
                        # if chutes_deployed_this_press > 0: print(f"Deployed {chutes_deployed_this_press} parachute(s) via key.") # Debug
                        # else: print("No ready parachutes to deploy.") # Debug

                # --- Respawn ---
                current_time = time.time()
                # Check for 'R' key and cooldown
                if event.key == pygame.K_r and (current_time - last_respawn_time > 1.0):
                    print("--- RESPAWNING ROCKET ---")
                    last_respawn_time = current_time
                    # --- Reset Simulation State ---
                    all_rockets.clear()
                    controlled_rocket = None
                    newly_created_rockets_this_frame.clear() # Clear temp list just in case
                    rockets_to_remove_this_frame.clear() # Clear temp list just in case
                    particle_manager.particles.clear()
                    collision_grace_period_pairs.clear() # CRITICAL: Clear grace period on respawn
                    next_sim_id = 0 # Reset sim ID counter
                    current_sim_frame = 0 # Reset frame counter

                    # --- Reload Blueprint and Re-initialize ---
                    reloaded_blueprint = RocketBlueprint.load_from_json(blueprint_file)
                    if reloaded_blueprint and reloaded_blueprint.parts:
                        # --- Repeat the initial setup logic ---
                        # Find root part reference
                        original_root_part_instance = None
                        for part in reloaded_blueprint.parts:
                             if part.part_data and part.part_data.get("type") == "CommandPod":
                                 original_root_part_instance = part
                                 break
                        if not original_root_part_instance and reloaded_blueprint.parts:
                            original_root_part_instance = reloaded_blueprint.parts[0]

                        # Find connected subassemblies
                        initial_subassemblies = reloaded_blueprint.find_connected_subassemblies()

                        # Create FlyingRocket instances for each assembly
                        for i, assembly_parts in enumerate(initial_subassemblies):
                             if not assembly_parts: continue
                             # Recalculate spawn position (spawn ON ground)
                             temp_bp = RocketBlueprint()
                             temp_bp.parts = assembly_parts
                             initial_com_local = temp_bp.calculate_subassembly_world_com(assembly_parts)
                             lowest_offset_y = temp_bp.get_lowest_point_offset_y()
                             start_x = i * 50
                             start_y_for_origin = (GROUND_Y - initial_spawn_y_offset) - lowest_offset_y
                             target_initial_com_pos = pygame.math.Vector2(start_x, start_y_for_origin + initial_com_local.y)

                             contains_original_root = original_root_part_instance and (original_root_part_instance in assembly_parts)
                             is_primary = (controlled_rocket is None and contains_original_root) or (controlled_rocket is None and i == 0)

                             try:
                                 # --- CRITICAL: Reset runtime state on blueprint parts *before* creating rocket ---
                                 for p in assembly_parts:
                                     p.current_hp = p.part_data.get("max_hp", 100)
                                     p.is_broken = False
                                     p.engine_enabled = True # Default engine state
                                     p.deployed = False # Reset parachute/etc state
                                     p.separated = False # Reset separator state
                                     p.current_temp = AMBIENT_TEMPERATURE
                                     p.is_overheating = False
                                     # Reset fuel to capacity for respawn
                                     if p.part_data.get("type") == "FuelTank":
                                         p.current_fuel = p.fuel_capacity

                                 # Create new rocket instance
                                 rocket_instance = FlyingRocket(
                                     parts_list=list(assembly_parts),
                                     initial_world_com_pos=target_initial_com_pos,
                                     initial_angle=0,
                                     initial_vel=pygame.math.Vector2(0,0),
                                     sim_instance_id=next_sim_id,
                                     is_primary_control=is_primary,
                                     original_root_ref=original_root_part_instance,
                                     current_frame=current_sim_frame # Pass current frame (0 for respawn)
                                 )

                                 # Add to a temporary list first to avoid modifying all_rockets while iterating later potentially
                                 newly_created_rockets_this_frame.append(rocket_instance)

                                 if is_primary:
                                     controlled_rocket = rocket_instance
                                     root_ref = controlled_rocket.original_root_part_ref
                                     controlled_rocket.has_active_control = (root_ref is not None) and (root_ref in controlled_rocket.parts) and (not root_ref.is_broken)
                                 else:
                                     rocket_instance.has_active_control = False
                                 next_sim_id += 1
                             except Exception as e:
                                 print(f"Respawn Error creating instance: {e}")
                                 import traceback
                                 traceback.print_exc()

                        # Fallback control assignment after creating all respawned rockets
                        if controlled_rocket is None and newly_created_rockets_this_frame:
                             controlled_rocket = newly_created_rockets_this_frame[0]
                             root_ref = controlled_rocket.original_root_part_ref
                             controlled_rocket.has_active_control = (root_ref is not None) and (root_ref in controlled_rocket.parts) and (not root_ref.is_broken)

                        # Add the newly created rockets from the temporary list to the main simulation list
                        all_rockets.extend(newly_created_rockets_this_frame)
                        newly_created_rockets_this_frame.clear() # Clear the temp list

                        print("Respawn Complete.")
                    else:
                        print("Respawn Failed: Cannot reload blueprint.")


            # --- Activate Part via Click (Engine Toggle / Deploy / Separate) ---
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: # Left Click
                 # Allow interaction only if there's a controlled rocket
                 if controlled_rocket:
                     click_screen_pos = pygame.math.Vector2(event.pos)
                     # Convert screen click to world coordinates using camera offset
                     click_world_pos = click_screen_pos + camera.offset
                     # Call the rocket's internal method to handle activation
                     # This method now correctly handles setting pending_separation
                     # and the needs_structural_update flag for separators.
                     controlled_rocket.activate_part_at_pos(click_world_pos)


        # --- Continuous Controls (Throttle) ---
        if controlled_rocket and controlled_rocket.has_active_control:
            keys = pygame.key.get_pressed()
            throttle_change = 0.0
            # Increase throttle
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                throttle_change += THROTTLE_CHANGE_RATE * dt
            # Decrease throttle
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                throttle_change -= THROTTLE_CHANGE_RATE * dt

            # Apply change and clamp throttle between 0.0 and 1.0
            if throttle_change != 0:
                controlled_rocket.throttle_level = max(0.0, min(1.0, controlled_rocket.throttle_level + throttle_change))


        # --- Update All Active Rockets ---
        for rocket in all_rockets:
            if not rocket.is_active:
                continue # Skip inactive rockets

            # Calculate air density at the rocket's current altitude
            current_air_density = 0.0
            try:
                # Calculate altitude above ground level (AGL)
                altitude_agl = max(0, GROUND_Y - rocket.get_world_com().y)
                current_air_density = get_air_density(altitude_agl)
            except NameError: # Fallback if function missing
                current_air_density = 0.0

            # Call the rocket's update method
            # This handles internal physics, fuel, heat, damage checks, ground collision, etc.
            # It also resets its *own* needs_structural_update flag internally AFTER using it.
            rocket.update(dt, current_air_density, particle_manager)

            # Mark rocket for removal if it became inactive during its update
            # (e.g., all parts destroyed by heat/damage)
            if not rocket.is_active:
                 if rocket not in rockets_to_remove_this_frame:
                    rockets_to_remove_this_frame.append(rocket)


        # --- Inter-Rocket Collision Detection & Resolution ---
        # Store pairs processed this frame to avoid duplicate impulses/damage
        collision_pairs_processed_this_frame = set()
        # Iterate through all pairs of active rockets
        for i in range(len(all_rockets)):
            r1 = all_rockets[i]
            # Skip if rocket is inactive or already marked for removal
            if r1 in rockets_to_remove_this_frame or not r1.is_active or not r1.parts:
                continue

            for j in range(i + 1, len(all_rockets)):
                r2 = all_rockets[j]
                # Skip if rocket is inactive or already marked for removal
                if r2 in rockets_to_remove_this_frame or not r2.is_active or not r2.parts:
                    continue

                # --- Apply Collision Grace Period Check ---
                # Create a sorted tuple of instance IDs for the dictionary key
                pair_key_grace = tuple(sorted((r1.sim_instance_id, r2.sim_instance_id)))
                # Check if this pair is currently in the grace period
                if pair_key_grace in collision_grace_period_pairs:
                    # If yes, skip all collision checks between r1 and r2 for this frame
                    # print(f"Skipping collision check between {r1.sim_instance_id} and {r2.sim_instance_id} (grace period)") # Debug
                    continue

                # --- Broad Phase Collision Check (Optional but recommended) ---
                # Check distance between centers of mass first
                dist_sq = (r1.get_world_com() - r2.get_world_com()).length_squared()
                # Estimate bounding radius (use larger dimension of local bounds)
                r1_radius_approx = max(r1.local_bounds.width, r1.local_bounds.height) / 2.0
                r2_radius_approx = max(r2.local_bounds.width, r2.local_bounds.height) / 2.0
                # If distance squared is greater than sum of radii squared (plus buffer), no collision possible
                if dist_sq > (r1_radius_approx + r2_radius_approx + 10)**2: # Add 10px buffer
                    continue

                # --- Narrow Phase Collision Check (Part-to-Part) ---
                collision_found_between_r1_r2 = False
                colliding_part_r1 = None
                colliding_part_r2 = None

                # Iterate through parts of r1
                for p1_idx, p1 in enumerate(r1.parts):
                    if p1.is_broken: continue # Skip broken parts
                    # Get approximate world AABB for the part
                    rect1 = r1.get_world_part_aabb(p1)

                    # Iterate through parts of r2
                    for p2_idx, p2 in enumerate(r2.parts):
                        if p2.is_broken: continue # Skip broken parts
                        # Get approximate world AABB for the part
                        rect2 = r2.get_world_part_aabb(p2)

                        # Check for overlap between part AABBs
                        if rect1.colliderect(rect2):
                            # Collision detected between p1 and p2!
                            collision_found_between_r1_r2 = True
                            colliding_part_r1 = p1
                            colliding_part_r2 = p2

                            # Break inner loops once a collision is found for this r1/r2 pair
                            break # Stop checking parts of r2
                    if collision_found_between_r1_r2:
                        break # Stop checking parts of r1

                # --- Collision Response ---
                if collision_found_between_r1_r2:
                    # Ensure this pair hasn't already been processed this frame
                    pair_key_processed = tuple(sorted((r1.sim_instance_id, r2.sim_instance_id)))
                    if pair_key_processed in collision_pairs_processed_this_frame:
                        continue # Already handled collision between r1 and r2

                    collision_pairs_processed_this_frame.add(pair_key_processed)
                    # print(f"Collision between {r1.sim_instance_id}({colliding_part_r1.part_id}) and {r2.sim_instance_id}({colliding_part_r2.part_id})") # Debug

                    # 1. Apply Damage
                    relative_velocity = r1.vel - r2.vel
                    impact_speed = relative_velocity.length()
                    # Apply damage specifically to the colliding parts found
                    r1.apply_collision_damage(impact_speed, particle_manager, specific_part_to_damage=colliding_part_r1)
                    r2.apply_collision_damage(impact_speed, particle_manager, specific_part_to_damage=colliding_part_r2)

                    # 2. Apply Physics Response (Simple Separation Impulse/Push)
                    # Calculate vector from r2 center to r1 center
                    collision_normal = r1.get_world_com() - r2.get_world_com()
                    if collision_normal.length_squared() > 1e-6: # Avoid zero vector
                        collision_normal.normalize_ip() # Normalize in place
                    else:
                        # If centers are coincident, pick an arbitrary direction (e.g., up)
                        collision_normal = pygame.math.Vector2(0, -1)

                    # Simple push based on mass ratio to separate them along the normal
                    # This is a basic response, more complex physics (impulses based on relative velocity) could be used
                    push_strength = 2.0 # How strongly to push them apart (adjust as needed)
                    total_m = r1.total_mass + r2.total_mass
                    if total_m > 0.01:
                         # Push r1 away from r2
                         r1.pos += collision_normal * push_strength * (r2.total_mass / total_m)
                         # Push r2 away from r1
                         r2.pos -= collision_normal * push_strength * (r1.total_mass / total_m)
                         # Can also add velocity impulse here based on relative velocity along normal


        # --- Process Connectivity Checks and Separations ---
        # Iterate over a *copy* of the list, as `all_rockets` might be modified
        # by adding new rockets from splits.
        rockets_to_process_for_splits = list(all_rockets)
        # Store rockets created *during this split-processing phase*
        new_rockets_created_in_split_phase: list[FlyingRocket] = []

        for rocket in rockets_to_process_for_splits:
            # Skip rockets already marked for removal or inactive
            if rocket in rockets_to_remove_this_frame or not rocket.is_active:
                continue

            processed_split_for_this_rocket = False # Flag to handle only one type of split per rocket per frame
            split_sibling_rockets: list[FlyingRocket] = [] # Track siblings *from this specific split event*

            # --- 1. Check for Splits due to Part Destruction ---
            # This is triggered if the rocket's internal update flagged a structural change
            # AND there are no pending manual separations.
            if rocket.needs_structural_update and not rocket.pending_separation:
                # Create a temporary blueprint to check connectivity of *remaining* parts
                temp_bp = RocketBlueprint()
                temp_bp.parts = rocket.parts # Use the current parts list (already filtered in handle_destroyed_parts)
                subassemblies = temp_bp.find_connected_subassemblies()

                # Check if the destruction resulted in multiple disconnected pieces
                if len(subassemblies) > 1:
                    processed_split_for_this_rocket = True
                    # print(f"[{rocket.sim_instance_id}] SPLIT (Destruction) into {len(subassemblies)} pieces!") # Debug

                    # Mark the original (now fragmented) rocket for removal
                    if rocket not in rockets_to_remove_this_frame:
                        rockets_to_remove_this_frame.append(rocket)

                    # Preserve throttle/master state if control is transferring
                    original_throttle = rocket.throttle_level
                    original_master_thrust = rocket.master_thrust_enabled

                    # Create new FlyingRocket instances for each fragment
                    for assembly_parts in subassemblies:
                        if not assembly_parts: continue # Skip empty fragments

                        try:
                            # Calculate the world CoM of this fragment *before* creating the new rocket
                            sub_com_world = rocket.calculate_subassembly_world_com(assembly_parts)

                            # Check if this fragment contains the original root part
                            contains_root = rocket.original_root_part_ref and (rocket.original_root_part_ref in assembly_parts)
                            # Assign control if the original rocket had control and this fragment has the root
                            is_primary = rocket.has_active_control and contains_root

                            # Create the new rocket instance for the fragment
                            # Parts in 'assembly_parts' retain their state (hp, temp, fuel, etc.)
                            new_fragment_rocket = FlyingRocket(
                                parts_list=list(assembly_parts), # Copy list
                                initial_world_com_pos=sub_com_world,
                                initial_angle=rocket.angle, # Inherit angle
                                initial_vel=rocket.vel, # Inherit linear velocity
                                sim_instance_id=next_sim_id,
                                is_primary_control=is_primary,
                                original_root_ref=rocket.original_root_part_ref, # Keep original root ref
                                current_frame=current_sim_frame # Pass current frame
                            )
                            # Inherit angular velocity
                            new_fragment_rocket.angular_velocity = rocket.angular_velocity
                            # If this new fragment gets control, set its throttle state
                            if new_fragment_rocket.has_active_control:
                                new_fragment_rocket.throttle_level = original_throttle
                                new_fragment_rocket.master_thrust_enabled = original_master_thrust

                            # Add to the list of rockets created *this frame*
                            new_rockets_created_in_split_phase.append(new_fragment_rocket)
                            # Add to the list of siblings *from this specific split* for grace period
                            split_sibling_rockets.append(new_fragment_rocket)
                            next_sim_id += 1

                        except Exception as e:
                            print(f"Error creating rocket from destruction split: {e}")
                            import traceback
                            traceback.print_exc()

            # --- 2. Check for Splits due to Activated Separators ---
            # This is triggered *only* if there are separators in the pending list.
            # The `needs_structural_update` flag is still important for the rocket's *internal*
            # recalculations, but it doesn't trigger *this specific block*.
            elif rocket.pending_separation and not processed_split_for_this_rocket:
                # Process all separators activated in the previous frame for this rocket
                separators_to_process = list(rocket.pending_separation)
                rocket.pending_separation.clear() # Clear the list after copying

                split_occurred_by_separator = False
                # The parts list *might* change if multiple separators cause sequential splits within one frame
                current_parts_in_rocket_being_processed = list(rocket.parts)

                # Preserve throttle/master state if control is transferring
                original_throttle = rocket.throttle_level
                original_master_thrust = rocket.master_thrust_enabled

                # Process each activated separator
                for sep_part in separators_to_process:
                    # Check if the separator still exists in the current assembly being processed
                    # (it might have been removed in a previous iteration of this loop if multiple separators fired)
                    if sep_part not in current_parts_in_rocket_being_processed:
                        continue

                    # Find world position of separator (for applying force)
                    separator_world_pos = rocket.get_world_part_center(sep_part)
                    separation_force = sep_part.part_data.get("separation_force", 1000) # N

                    # --- Check Connectivity *After* Removing Separator ---
                    # Temporarily remove the separator to see if the structure splits
                    parts_without_this_separator = [p for p in current_parts_in_rocket_being_processed if p != sep_part]
                    # Create a temporary blueprint with the remaining parts
                    temp_bp = RocketBlueprint()
                    temp_bp.parts = parts_without_this_separator
                    # Check connectivity of the remaining structure
                    subassemblies = temp_bp.find_connected_subassemblies()

                    # --- Handle Split ---
                    if len(subassemblies) > 1: # Separation caused a split!
                        split_occurred_by_separator = True
                        processed_split_for_this_rocket = True # Mark this rocket as having split
                        # print(f"  > SPLIT by Separator {sep_part.part_id} into {len(subassemblies)} pieces!") # Debug

                        # Mark the original combined rocket instance for removal
                        if rocket not in rockets_to_remove_this_frame:
                            rockets_to_remove_this_frame.append(rocket)
                        # The original instance is being replaced, clear its parts conceptually for this loop
                        current_parts_in_rocket_being_processed = []

                        # Create new rocket instances for the separated pieces
                        for assembly_parts in subassemblies:
                            if not assembly_parts: continue
                            try:
                                # Calculate world CoM of the new piece
                                sub_com_world = rocket.calculate_subassembly_world_com(assembly_parts)

                                # Check for root part and control transfer
                                contains_root = rocket.original_root_part_ref and (rocket.original_root_part_ref in assembly_parts)
                                is_primary = rocket.has_active_control and contains_root

                                # Create the new rocket instance
                                new_separated_rocket = FlyingRocket(
                                    parts_list=list(assembly_parts),
                                    initial_world_com_pos=sub_com_world,
                                    initial_angle=rocket.angle,
                                    initial_vel=rocket.vel, # Start with same velocity
                                    sim_instance_id=next_sim_id,
                                    is_primary_control=is_primary,
                                    original_root_ref=rocket.original_root_part_ref,
                                    current_frame=current_sim_frame # Pass frame
                                )
                                new_separated_rocket.angular_velocity = rocket.angular_velocity
                                if new_separated_rocket.has_active_control:
                                    new_separated_rocket.throttle_level = original_throttle
                                    new_separated_rocket.master_thrust_enabled = original_master_thrust

                                # --- Apply Separation Impulse ---
                                # Calculate direction vector from separator center to new piece's CoM
                                separation_vector = new_separated_rocket.get_world_com() - separator_world_pos
                                # Normalize the direction vector
                                if separation_vector.length_squared() > 1e-6:
                                    separation_direction = separation_vector.normalize()
                                else:
                                    # If CoM is coincident with separator, push outwards based on rocket angle
                                    separation_direction = pygame.math.Vector2(0, -1).rotate(-rocket.angle + random.uniform(-5, 5)) # Add slight randomness

                                # Calculate impulse magnitude (Force * time / mass = delta_v)
                                # Apply force over a very short effective time (e.g., fraction of a frame)
                                effective_impulse_time = 0.06 # Adjust as needed
                                impulse_magnitude = (separation_force / max(0.1, new_separated_rocket.total_mass)) * effective_impulse_time
                                delta_velocity = separation_direction * impulse_magnitude
                                # Apply the velocity change to the new piece
                                new_separated_rocket.vel += delta_velocity

                                # Add to lists
                                new_rockets_created_in_split_phase.append(new_separated_rocket)
                                split_sibling_rockets.append(new_separated_rocket) # Track siblings
                                next_sim_id += 1

                            except Exception as e:
                                print(f"Error creating rocket from separation split: {e}")
                                import traceback
                                traceback.print_exc()

                        # Stop processing further separators on *this specific original rocket*
                        # because it has been replaced by the new fragments.
                        break # Exit the loop iterating through sep_part

                    else:
                        # --- No Split Occurred ---
                        # The structure remained connected after removing this separator.
                        # Update the current list of parts being processed for the *next* separator check.
                        current_parts_in_rocket_being_processed = parts_without_this_separator
                        # Mark the separator part itself as 'separated' (visually/logically used)
                        # It might still be in the rocket's part list if it wasn't the one causing a split yet.
                        if sep_part in rocket.parts: # Check if it wasn't removed by destruction simultaneously
                            sep_part.separated = True # Mark as used


                # --- Post-Separator Processing for the *Original* Rocket ---
                # If a split occurred *at any point* while processing separators for this rocket,
                # the original 'rocket' instance is already marked for removal, so we do nothing more here.
                # If *no split occurred* but separators were processed (and thus removed from the structure),
                # we need to update the original 'rocket' instance's state.
                if not split_occurred_by_separator and len(current_parts_in_rocket_being_processed) < len(rocket.parts):
                    # Update the original rocket's part list to the final state after removing separators
                    rocket.parts = current_parts_in_rocket_being_processed
                    # --- CRITICAL: Update internal component lists ---
                    rocket.engines = [e for e in rocket.engines if e in rocket.parts]
                    rocket.fuel_tanks = [t for t in rocket.fuel_tanks if t in rocket.parts]
                    rocket.parachutes = [pc for pc in rocket.parachutes if pc in rocket.parts]
                    # Remove used separators from the list
                    rocket.separators = [s for s in rocket.separators if s in rocket.parts and s not in separators_to_process]
                    rocket.engine_firing_status = {e: False for e in rocket.engines} # Reset status

                    # Check if the rocket became empty after removing separators
                    if not rocket.parts:
                        rocket.is_active = False # Deactivate if empty
                        if rocket not in rockets_to_remove_this_frame:
                             rockets_to_remove_this_frame.append(rocket)
                    else:
                        # --- Flag for internal update ---
                        # Since parts were removed (the separators), the rocket needs
                        # to recalculate its physics/fuel map on its *next* update cycle.
                        rocket.needs_structural_update = True

            # --- Add Grace Period for Newly Created Siblings ---
            # This happens *after* processing either destruction or separation for 'rocket'.
            if split_sibling_rockets: # If any new rockets were created from this event
                # Iterate through all unique pairs of the newly created sibling rockets
                for i_sib, r_sib1 in enumerate(split_sibling_rockets):
                    for j_sib in range(i_sib + 1, len(split_sibling_rockets)):
                        r_sib2 = split_sibling_rockets[j_sib]
                        # Create the sorted pair key for the dictionary
                        grace_pair_key = tuple(sorted((r_sib1.sim_instance_id, r_sib2.sim_instance_id)))
                        # Add the pair to the grace period dictionary with the defined frame count
                        collision_grace_period_pairs[grace_pair_key] = COLLISION_GRACE_FRAMES
                        # print(f"Adding grace period ({COLLISION_GRACE_FRAMES} frames) for pair: {grace_pair_key}") # Debug

                # Clear the list for the next rocket being processed
                split_sibling_rockets.clear()

            # --- Reset internal structural update flag if needed ---
            # This flag is reset *inside* rocket.update() after it's used for internal recalc.
            # No need to reset it here. The split check logic above correctly uses it as a trigger.


        # --- Update Rocket Lists (Add new, remove old) ---
        # Add all newly created rockets (from all splits this frame) to the main list
        if new_rockets_created_in_split_phase:
            new_potential_controlled_rocket = None
            for new_rocket in new_rockets_created_in_split_phase:
                # Check if already added (shouldn't happen with current logic, but safe)
                if new_rocket not in all_rockets:
                     all_rockets.append(new_rocket)

                # Check if this new rocket should potentially take control
                if new_rocket.has_active_control:
                    # If there was a controlled rocket, check if it lost control during the split
                    if controlled_rocket and controlled_rocket not in rockets_to_remove_this_frame:
                        # Verify the old controlled rocket still has its root part and it's not broken
                        root_ref_old = controlled_rocket.original_root_part_ref
                        if not root_ref_old or root_ref_old not in controlled_rocket.parts or root_ref_old.is_broken:
                           controlled_rocket.has_active_control = False # Mark old one as uncontrolled
                           # print(f"Previous control rocket [{controlled_rocket.sim_instance_id}] lost control post-split.") # Debug

                    # If there's no current controlled rocket, or the previous one just lost control,
                    # assign control to this new one.
                    if not controlled_rocket or not controlled_rocket.has_active_control:
                        new_potential_controlled_rocket = new_rocket
                        # print(f"Control transferring to new rocket [{new_potential_controlled_rocket.sim_instance_id}] post-split.") # Debug

            # Update the main controlled rocket reference if a transfer occurred
            if new_potential_controlled_rocket:
                controlled_rocket = new_potential_controlled_rocket

            # Clear the temporary list after processing
            new_rockets_created_in_split_phase.clear()


        # Remove rockets marked for deletion from the main list
        if rockets_to_remove_this_frame:
            was_controlled_rocket_removed = controlled_rocket in rockets_to_remove_this_frame
            removed_ids = {r.sim_instance_id for r in rockets_to_remove_this_frame}

            # --- Clean up Grace Period Dictionary ---
            # Remove any pairs involving the rockets being deleted
            pairs_to_delete_from_grace = []
            for pair in collision_grace_period_pairs.keys():
                 if pair[0] in removed_ids or pair[1] in removed_ids:
                     pairs_to_delete_from_grace.append(pair)
            for pair in pairs_to_delete_from_grace:
                 if pair in collision_grace_period_pairs: # Check if exists before deleting
                      del collision_grace_period_pairs[pair]
                      # print(f"Removed grace pair {pair} due to rocket removal.") # Debug

            # Filter the main list
            all_rockets = [r for r in all_rockets if r not in rockets_to_remove_this_frame]
            # print(f"Removed {len(rockets_to_remove_this_frame)} rocket instances.") # Debug
            rockets_to_remove_this_frame.clear() # Clear the removal list

            # Handle loss of the controlled rocket
            if was_controlled_rocket_removed:
                # print("Controlled rocket instance was removed.") # Debug
                controlled_rocket = None # Clear reference
                # Try to find a new controllable rocket among the remaining ones
                # Prioritize rockets explicitly marked with has_active_control
                for rkt in all_rockets:
                    if rkt.has_active_control:
                        controlled_rocket = rkt
                        # print(f"Found remaining controlled rocket [{controlled_rocket.sim_instance_id}].") # Debug
                        break
                # If none found, fallback to checking for root part presence/health
                if not controlled_rocket:
                    # print("No explicitly controlled rocket found, checking for root part presence...") # Debug
                    for rkt in all_rockets:
                        root_ref = rkt.original_root_part_ref
                        if root_ref and root_ref in rkt.parts and not root_ref.is_broken:
                            controlled_rocket = rkt
                            # Mark it as controllable now
                            controlled_rocket.has_active_control = True
                            # print(f"Fallback control assigned to Rocket {controlled_rocket.sim_instance_id} (has root).") # Debug
                            break
                # if not controlled_rocket: print("No controllable rocket found after removal.") # Debug

        # --- Camera Update ---
        if controlled_rocket:
            # Follow the controlled rocket's center of mass
            camera.update(controlled_rocket.get_world_com())
        elif all_rockets:
            # If no controlled rocket, follow the first active rocket in the list
            camera.update(all_rockets[0].get_world_com())
        # Else: Camera stays where it was (e.g., if all rockets are destroyed)


        # --- Drawing ---
        screen.fill(BLACK) # Clear screen with black first

        # Draw background (sky gradient, stars)
        try:
            draw_earth_background(screen, camera, stars)
        except NameError: pass # Ignore if function missing

        # Draw terrain (ground plane)
        try:
            draw_terrain(screen, camera)
        except NameError: pass # Ignore if function missing

        # Draw all active rockets
        total_parts_drawn = 0
        total_broken_drawn = 0
        for rocket in all_rockets:
            if rocket.is_active:
                # The draw method returns the count of broken parts in that rocket
                broken_count = rocket.draw(screen, camera, particle_manager)
                total_parts_drawn += len(rocket.parts)
                total_broken_drawn += broken_count

        # Update and draw particles (explosions, reentry)
        particle_manager.update(dt)
        particle_manager.draw(screen, camera)

        # --- Draw UI Elements ---
        if controlled_rocket:
            # Throttle Bar Display
            bar_w = 20
            bar_h = 100
            bar_x = 15
            bar_y = SCREEN_HEIGHT - bar_h - 40
            # Background
            pygame.draw.rect(screen, COLOR_UI_BAR_BG, (bar_x, bar_y, bar_w, bar_h))
            # Fill based on throttle level
            fill_h = bar_h * controlled_rocket.throttle_level
            pygame.draw.rect(screen, COLOR_UI_BAR, (bar_x, bar_y + bar_h - fill_h, bar_w, fill_h))
            # Border
            pygame.draw.rect(screen, WHITE, (bar_x, bar_y, bar_w, bar_h), 1)
            # Labels
            th_label = ui_font.render("Thr", True, WHITE)
            screen.blit(th_label, (bar_x, bar_y + bar_h + 5))
            th_value = ui_font.render(f"{controlled_rocket.throttle_level*100:.0f}%", True, WHITE)
            screen.blit(th_value, (bar_x, bar_y - 18))

            # Telemetry Data Display
            alt_agl = max(0, GROUND_Y - controlled_rocket.get_lowest_point_world().y) # Altitude above ground
            alt_msl = GROUND_Y - controlled_rocket.get_world_com().y # Altitude relative to world origin Y=0
            ctrl_status = "OK" if controlled_rocket.has_active_control else "NO CTRL"
            thrust_status = "ON" if controlled_rocket.master_thrust_enabled else "OFF"
            landed_status = "LANDED" if controlled_rocket.landed else "FLYING"
            max_temp_k = controlled_rocket.max_temp_reading # Max temp recorded this frame

            # Determine color for temperature readout based on thresholds
            temp_color = WHITE
            hottest_part_max_temp_allowed = DEFAULT_MAX_TEMP
            if controlled_rocket.parts: # Find max allowable temp on current parts
                 allowable_temps = [p.part_data.get('max_temp', DEFAULT_MAX_TEMP) for p in controlled_rocket.parts if p.part_data]
                 if allowable_temps:
                     hottest_part_max_temp_allowed = max(allowable_temps)

            # Color transitions for temperature
            if max_temp_k > REENTRY_EFFECT_THRESHOLD_TEMP: temp_color = (255, 255, 0) # Yellow warning
            if max_temp_k > hottest_part_max_temp_allowed * 0.9: temp_color = (255, 100, 0) # Orange critical
            if max_temp_k > hottest_part_max_temp_allowed: temp_color = RED # Red danger

            # Get total current fuel
            total_fuel = controlled_rocket.get_total_current_fuel()

            # List of telemetry strings
            status_texts = [
                f"Alt(AGL): {alt_agl:.1f}m",
                f"Alt(MSL): {alt_msl:.1f}m",
                f"Vvel: {controlled_rocket.vel.y:.1f} m/s",
                f"Hvel: {controlled_rocket.vel.x:.1f} m/s",
                f"Speed: {controlled_rocket.vel.length():.1f} m/s",
                f"Angle: {controlled_rocket.angle:.1f} deg",
                f"AngVel: {controlled_rocket.angular_velocity:.1f} d/s",
                f"Thr: {controlled_rocket.throttle_level*100:.0f}% [{thrust_status}]",
                f"Fuel: {total_fuel:.1f} units",
                f"Mass: {controlled_rocket.total_mass:.1f} kg",
                f"Control: {ctrl_status}",
                f"Status: {landed_status}",
                f"ID: {controlled_rocket.sim_instance_id}",
                f"MaxTemp: {max_temp_k:.0f} K"
            ]
            text_y_start = 10
            # Determine text color based on content
            control_color = WHITE if controlled_rocket.has_active_control else RED
            # Draw each line of telemetry
            for i, text in enumerate(status_texts):
                # Special colors for certain lines
                line_color = temp_color if "MaxTemp" in text else (control_color if "Control" in text else WHITE)
                text_surf = ui_font.render(text, True, line_color)
                screen.blit(text_surf, (bar_x + bar_w + 10, text_y_start + i * 18))

        elif not all_rockets: # If all rockets are destroyed/gone
            # Display message indicating failure
            destroyed_text = ui_font_large.render("ALL ROCKETS DESTROYED", True, RED)
            text_rect = destroyed_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            screen.blit(destroyed_text, text_rect)
            # Add prompt to respawn
            respawn_text = ui_font.render("Press 'R' to Respawn", True, WHITE)
            respawn_rect = respawn_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 40))
            screen.blit(respawn_text, respawn_rect)

        # --- Draw Debug Info ---
        fps = clock.get_fps()
        debug_y = 10
        debug_x = SCREEN_WIDTH - 120 # Position debug info top-right
        # FPS counter
        fps_text = ui_font.render(f"FPS: {fps:.1f}", True, WHITE)
        screen.blit(fps_text, (debug_x, debug_y))
        debug_y += 18
        # Object count
        obj_text = ui_font.render(f"Rockets: {len(all_rockets)}", True, WHITE)
        screen.blit(obj_text, (debug_x, debug_y))
        debug_y += 18
        # Total parts count
        parts_text = ui_font.render(f"Parts: {total_parts_drawn}", True, WHITE)
        screen.blit(parts_text, (debug_x, debug_y))
        debug_y += 18
        # Particle count
        particle_text = ui_font.render(f"Particles: {len(particle_manager.particles)}", True, WHITE)
        screen.blit(particle_text, (debug_x, debug_y))
        debug_y += 18
        # Grace period pair count (for debugging the fix)
        grace_text = ui_font.render(f"Grace Pairs: {len(collision_grace_period_pairs)}", True, WHITE)
        screen.blit(grace_text, (debug_x, debug_y))


        # --- Update Display ---
        pygame.display.flip() # Show the drawn frame

    print("--- Exiting Simulation ---")


# --- Direct Run Logic (for testing, unchanged) ---
if __name__ == '__main__':
     pygame.init()
     screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
     pygame.display.set_caption("PySpaceFlight - Direct Sim Test")
     clock = pygame.time.Clock()

     # Define test blueprint path
     assets_dir = "assets"
     # *** Use a blueprint that involves separators ***
     test_blueprint = os.path.join(assets_dir, "grid_rocket_separator_test.json") # Or current_build.json

     # Create a default blueprint if the test one doesn't exist
     if not os.path.exists(test_blueprint):
         print(f"Test blueprint '{test_blueprint}' not found, creating default with separator...")
         os.makedirs(assets_dir, exist_ok=True) # Ensure assets directory exists
         # Create a simple default rocket with a separator
         bp = RocketBlueprint("Default Separator Test")
         pod_pos = pygame.math.Vector2(0,0); pod_data = get_part_data("pod_mk1")
         bp.add_part("pod_mk1", pod_pos)
         # Add parachute on top
         para_data = get_part_data("parachute_mk1"); para_rel_pos = pod_data['logical_points']['top'] - para_data['logical_points']['bottom']
         bp.add_part("parachute_mk1", pod_pos + para_rel_pos)
         # Add separator below pod
         sep_data = get_part_data("separator_tr_s1"); sep_rel_pos = pod_data['logical_points']['bottom'] - sep_data['logical_points']['top']
         bp.add_part("separator_tr_s1", pod_pos + sep_rel_pos)
         # Add tank below separator
         tank_data = get_part_data("tank_small"); tank_rel_pos = sep_data['logical_points']['bottom'] - tank_data['logical_points']['top']
         bp.add_part("tank_small", pod_pos + sep_rel_pos + tank_rel_pos)
         # Add engine below tank
         engine_data = get_part_data("engine_basic"); engine_rel_pos = tank_data['logical_points']['bottom'] - engine_data['logical_points']['top']
         bp.add_part("engine_basic", pod_pos + sep_rel_pos + tank_rel_pos + engine_rel_pos)

         bp.save_to_json(test_blueprint)
         print(f"Saved default blueprint to {test_blueprint}")

     # Run the simulation using the potentially modified run_simulation function
     run_simulation(screen, clock, test_blueprint)

     pygame.quit()
     sys.exit()