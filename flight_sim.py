# flight_sim.py
import pygame
import math
import sys
import random
import time
import queue
import os
import json # --- MP Change --- Added for blueprint loading/saving from string
from collections import deque

# Import necessary classes/functions
from parts import draw_part_shape, get_part_data
from rocket_data import RocketBlueprint, PlacedPart, AMBIENT_TEMPERATURE, FUEL_MASS_PER_UNIT
from ui_elements import (SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, BLACK, GRAY, RED, GREEN, BLUE, LIGHT_GRAY,
                         COLOR_SKY_BLUE, COLOR_SPACE_BLACK, COLOR_HORIZON, COLOR_GROUND,
                         COLOR_FLAME, COLOR_UI_BAR, COLOR_UI_BAR_BG, COLOR_EXPLOSION,
                         COLOR_ENGINE_ENABLED, COLOR_ENGINE_DISABLED, COLOR_ACTIVATABLE_READY, COLOR_ACTIVATABLE_USED)
# --- MP Change --- Import network constants and classes
import network

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
STAR_COUNT = 20000 # Adjusted from previous (was 2000)
STAR_FIELD_DEPTH = 10000
COLLISION_GRACE_FRAMES = 150
POSITIONAL_NUDGE_FACTOR = 1.0
# --- MP Change --- How often clients send state updates (in seconds) - less frequent for less traffic
CLIENT_STATE_UPDATE_INTERVAL = 0.1 # e.g., 10 times per second
# --- MP Change --- How far apart to spawn multiplayer launchpads
MP_LAUNCHPAD_SPACING = 200

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
    # ... (EffectParticle class code remains unchanged) ...
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
            time_ratio = max(0.0, min(1.0, 1.0 - (self.life / self.max_life)))
            try: self.current_color = self.start_color.lerp(self.end_color, time_ratio)
            except ValueError: self.current_color = self.start_color
            self.current_radius = self.start_radius + (self.end_radius - self.start_radius) * time_ratio
            return True
        else: return False
    def draw(self, surface, camera):
        if self.life > 0 and self.current_radius >= 0.5:
            screen_pos = camera.apply(self.pos)
            if -self.current_radius < screen_pos.x < SCREEN_WIDTH + self.current_radius and \
               -self.current_radius < screen_pos.y < SCREEN_HEIGHT + self.current_radius:
                 try:
                     draw_color = (int(self.current_color.r), int(self.current_color.g), int(self.current_color.b))
                     pygame.draw.circle(surface, draw_color, screen_pos, int(self.current_radius))
                 except ValueError: pass

class ParticleManager:
    # ... (ParticleManager class code remains unchanged) ...
    def __init__(self):
        self.particles: list[EffectParticle] = []
    def add_explosion(self, pos, num_particles=15, max_life=0.5, max_speed=100, colors=COLOR_EXPLOSION):
        for _ in range(num_particles):
            angle = random.uniform(0, 360); speed = random.uniform(max_speed * 0.2, max_speed)
            vel = pygame.math.Vector2(speed, 0).rotate(angle); life = random.uniform(max_life * 0.3, max_life)
            color = random.choice(colors); radius = random.uniform(2, 5)
            self.particles.append(EffectParticle(pos, vel, life, color, start_radius=radius, end_radius=0))
    def add_reentry_spark(self, pos, base_vel, intensity_factor):
        angle_offset = random.uniform(-25, 25)
        if base_vel.length() > 0: spark_dir = -base_vel.normalize()
        else: spark_dir = pygame.math.Vector2(0, 1)
        spark_vel = spark_dir.rotate(angle_offset) * REENTRY_PARTICLE_SPEED * (0.5 + intensity_factor) + base_vel * 0.1
        life = REENTRY_PARTICLE_LIFETIME * (0.7 + random.random() * 0.6); start_rad = 1 + 3 * intensity_factor; end_rad = 0
        start_col = REENTRY_PARTICLE_COLOR_START.lerp(REENTRY_PARTICLE_COLOR_END, intensity_factor * 0.8); end_col = REENTRY_PARTICLE_COLOR_START
        self.particles.append(EffectParticle(pos, spark_vel, life, start_col, end_col, start_rad, end_rad))
    def update(self, dt): self.particles = [p for p in self.particles if p.update(dt)]
    def draw(self, surface, camera):
        for p in self.particles: p.draw(surface, camera)

# --- Camera Class (Unchanged) ---
class Camera:
    # ... (Camera class code remains unchanged) ...
    def __init__(self, width, height):
        self.camera_rect = pygame.Rect(0, 0, width, height)
        self.width = width; self.height = height; self.offset = pygame.math.Vector2(0, 0)
    def apply(self, target_pos: pygame.math.Vector2) -> pygame.math.Vector2: return target_pos - self.offset
    def apply_rect(self, target_rect: pygame.Rect) -> pygame.Rect: return target_rect.move(-self.offset.x, -self.offset.y)
    def update(self, target_pos: pygame.math.Vector2):
        x = target_pos.x - self.width // 2; y = target_pos.y - self.height // 2
        self.offset = pygame.math.Vector2(x, y)

# --- FlyingRocket Class ---
class FlyingRocket:
    # Added player_id and name for MP identification
    def __init__(self, parts_list: list[PlacedPart], initial_world_com_pos: pygame.math.Vector2,
                 initial_angle=0, initial_vel=pygame.math.Vector2(0,0),
                 sim_instance_id=0, is_primary_control=False, original_root_ref=None,
                 current_frame=0, player_id=0, player_name="Unknown"): # --- MP Change --- Added player_id, player_name
        self.sim_instance_id = sim_instance_id
        self.player_id = player_id # --- MP Change --- ID of the player controlling/owning this rocket
        self.player_name = player_name # --- MP Change --- Name of the player
        self.parts = parts_list
        if not self.parts: raise ValueError("Cannot initialize FlyingRocket with empty parts list.")
        self.blueprint_name = f"Rocket_{sim_instance_id}_P{player_id}" # --- MP Change --- Include player ID
        self.original_root_part_ref = original_root_ref
        # --- MP Change --- Local control flag (True only for the rocket this client instance controls)
        self.is_local_player = is_primary_control
        # --- MP Change --- Server/Host control flag (True if this instance represents the locally controlled rocket on the host/client)
        # This might be redundant with is_local_player, simplifying. Keep is_local_player.
        # self.has_active_control = is_primary_control # Replaced by is_local_player check where needed

        self.creation_frame = current_frame

        # Physics State
        self.pos = pygame.math.Vector2(initial_world_com_pos)
        self.vel = pygame.math.Vector2(initial_vel)
        self.acc = pygame.math.Vector2(0, 0)
        self.angle = initial_angle
        self.angular_velocity = 0.0

        # Control State (Only relevant/updated for local player's rocket)
        self.throttle_level = 0.0
        self.master_thrust_enabled = False

        # Component References & State
        self.engines: list[PlacedPart] = []
        self.fuel_tanks: list[PlacedPart] = []
        self.parachutes: list[PlacedPart] = []
        self.separators: list[PlacedPart] = []
        self.engine_fuel_sources: dict[PlacedPart, list[PlacedPart]] = {}
        self._part_connections_cache: dict[PlacedPart, list[PlacedPart]] | None = None

        # Populate components and assign indices
        for i, part in enumerate(self.parts):
            part.part_index = i
            part_type = part.part_data.get("type")
            if part_type == "Engine": self.engines.append(part)
            elif part_type == "FuelTank":
                self.fuel_tanks.append(part)
                part.current_fuel = max(0.0, min(part.current_fuel, part.fuel_capacity)) # Clamp fuel
            elif part_type == "Parachute": self.parachutes.append(part)
            elif part_type == "Separator": self.separators.append(part)

        # Calculated Physics Properties
        self.total_mass = 0.01
        self.dry_mass = 0.0
        self.moment_of_inertia = 10000.0
        self.center_of_mass_offset = pygame.math.Vector2(0, 0)
        self.local_bounds = pygame.Rect(0,0,1,1)

        self.calculate_physics_properties()
        self.calculate_bounds()
        self._build_fuel_source_map()

        # Correct initial world position based on calculated CoM offset
        initial_com_offset_rotated = self.center_of_mass_offset.rotate(-self.angle)
        self.pos = initial_world_com_pos - initial_com_offset_rotated

        # Status Flags
        self.landed = False
        self.thrusting = False # Is the rocket *trying* to thrust (master on, throttle > 0)
        self.is_active = True
        self.pending_separation: list[PlacedPart] = []
        self.needs_structural_update = False
        self.was_landed_last_frame = False
        self.max_temp_reading = AMBIENT_TEMPERATURE
        # Dict tracks which engines *actually* fired (had fuel)
        self.engine_firing_status: dict[PlacedPart, bool] = {e: False for e in self.engines}
        # --- MP Change --- Last time state was sent (for client throttling)
        self.last_state_send_time = 0.0

    # --- Connectivity & Fuel Map Methods (Unchanged) ---
    # _build_connection_cache, _invalidate_connection_cache, _get_connected_parts
    # _get_rule_name_for_point, _check_rule_compatibility
    # _find_accessible_tanks_for_engine, _build_fuel_source_map
    # ... (These methods remain exactly the same as before) ...
    def _build_connection_cache(self):
        self._part_connections_cache = {}
        all_parts_in_assembly = set(self.parts); CONNECTION_TOLERANCE_SQ = (3.0)**2
        part_local_points = {}
        for part in self.parts:
            points = part.part_data.get("logical_points", {}); local_points = {}
            for name, local_pos_rel_part in points.items(): local_points[name] = part.relative_pos + local_pos_rel_part.rotate(-part.relative_angle)
            part_local_points[part] = local_points
        for part in self.parts:
            self._part_connections_cache[part] = []
            current_part_data = part.part_data; current_rules = current_part_data.get("attachment_rules", {}); current_points = part_local_points.get(part, {})
            for other_part in all_parts_in_assembly:
                if other_part == part: continue
                other_part_data = other_part.part_data; other_rules = other_part_data.get("attachment_rules", {}); other_points = part_local_points.get(other_part, {}); is_connected = False
                for cpn, clp in current_points.items():
                    for opn, olp in other_points.items():
                        if (clp - olp).length_squared() < CONNECTION_TOLERANCE_SQ:
                            crn = self._get_rule_name_for_point(cpn, current_rules); orn = self._get_rule_name_for_point(opn, other_rules)
                            if crn and orn and self._check_rule_compatibility(current_part_data.get("type"), crn, current_rules, other_part_data.get("type"), orn, other_rules):
                                is_connected = True; break
                    if is_connected: break
                if is_connected: self._part_connections_cache[part].append(other_part)
    def _invalidate_connection_cache(self): self._part_connections_cache = None
    def _get_connected_parts(self, part: PlacedPart) -> list[PlacedPart]:
        if self._part_connections_cache is None: self._build_connection_cache()
        return self._part_connections_cache.get(part, [])
    def _get_rule_name_for_point(self, point_name: str, rules: dict) -> str | None:
        if "bottom" in point_name and "bottom_center" in rules: return "bottom_center"
        if "top" in point_name and "top_center" in rules: return "top_center"
        if "left" in point_name and "left_center" in rules: return "left_center"
        if "right" in point_name and "right_center" in rules: return "right_center"
        return None
    def _check_rule_compatibility(self, type1, rule_name1, rules1, type2, rule_name2, rules2) -> bool:
        allowed_on_1 = rules1.get(rule_name1, {}).get("allowed_types", []); allowed_on_2 = rules2.get(rule_name2, {}).get("allowed_types", [])
        if type2 not in allowed_on_1 or type1 not in allowed_on_2: return False
        if ("top" in rule_name1 and "bottom" in rule_name2) or ("bottom" in rule_name1 and "top" in rule_name2) or \
           ("left" in rule_name1 and "right" in rule_name2) or ("right" in rule_name1 and "left" in rule_name2): return True
        return False
    def _find_accessible_tanks_for_engine(self, start_engine: PlacedPart) -> list[PlacedPart]:
        accessible_tanks: list[PlacedPart] = []; queue = deque(); visited_in_search = {start_engine}
        initial_neighbors = self._get_connected_parts(start_engine)
        for neighbor in initial_neighbors:
             if neighbor not in visited_in_search: queue.append(neighbor); visited_in_search.add(neighbor)
        while queue:
            current_part = queue.popleft(); part_type = current_part.part_data.get("type")
            if part_type == "FuelTank":
                if current_part not in accessible_tanks: accessible_tanks.append(current_part)
            is_blocker = (part_type == "Separator") or (part_type == "Engine" and current_part != start_engine)
            if not is_blocker:
                neighbors = self._get_connected_parts(current_part)
                for neighbor in neighbors:
                    if neighbor not in visited_in_search: visited_in_search.add(neighbor); queue.append(neighbor)
        return accessible_tanks
    def _build_fuel_source_map(self):
        self._invalidate_connection_cache(); self.engine_fuel_sources = {}
        for engine in self.engines: self.engine_fuel_sources[engine] = self._find_accessible_tanks_for_engine(engine)


    # --- Physics Calculation Methods (Unchanged) ---
    # calculate_physics_properties, calculate_bounds
    # ... (These methods remain exactly the same as before) ...
    def calculate_physics_properties(self):
        total_m = 0.0; com_numerator = pygame.math.Vector2(0, 0); moi_sum = 0.0; current_dry_mass = 0.0
        for part in self.parts:
            part_mass_static = part.part_data.get("mass", 0); current_dry_mass += part_mass_static; part_fuel_mass = 0.0
            if part.part_data.get("type") == "FuelTank": part_fuel_mass = part.current_fuel * FUEL_MASS_PER_UNIT
            part_mass_current = part_mass_static + part_fuel_mass; total_m += part_mass_current
            com_numerator += part.relative_pos * part_mass_current
        self.dry_mass = current_dry_mass; self.total_mass = max(0.01, total_m)
        if self.total_mass > 0.01: self.center_of_mass_offset = com_numerator / self.total_mass
        else: self.center_of_mass_offset = self.parts[0].relative_pos if self.parts else pygame.math.Vector2(0, 0)
        for part in self.parts:
             part_mass_static = part.part_data.get("mass", 0); part_fuel_mass = 0.0
             if part.part_data.get("type") == "FuelTank": part_fuel_mass = part.current_fuel * FUEL_MASS_PER_UNIT
             part_mass_current = part_mass_static + part_fuel_mass
             w = part.part_data.get("width", 1); h = part.part_data.get("height", 1)
             i_part_center = (1/12.0) * part_mass_current * (w**2 + h**2)
             dist_vec = part.relative_pos - self.center_of_mass_offset; d_sq = dist_vec.length_squared()
             moi_sum += i_part_center + part_mass_current * d_sq
        self.moment_of_inertia = max(1.0, moi_sum)
    def calculate_bounds(self):
        if not self.parts: self.local_bounds = pygame.Rect(0, 0, 0, 0); return
        min_x, max_x = float('inf'), float('-inf'); min_y, max_y = float('inf'), float('-inf')
        for p in self.parts:
            half_w = p.part_data['width'] / 2.0; half_h = p.part_data['height'] / 2.0
            center_x = p.relative_pos.x; center_y = p.relative_pos.y
            min_x = min(min_x, center_x - half_w); max_x = max(max_x, center_x + half_w)
            min_y = min(min_y, center_y - half_h); max_y = max(max_y, center_y + half_h)
        if min_x == float('inf'): self.local_bounds = pygame.Rect(0,0,0,0)
        else: self.local_bounds = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)

    # --- Coordinate and State Access Methods (Mostly Unchanged) ---
    # get_world_com, get_world_part_center, get_parts_near_world_pos
    # get_lowest_point_world, get_world_part_aabb, get_thrust_and_consumption
    # --- get_thrust_and_consumption needs slight modification for remote rockets ---
    def get_world_com(self) -> pygame.math.Vector2:
        com_offset_rotated = self.center_of_mass_offset.rotate(-self.angle)
        return self.pos + com_offset_rotated
    def get_world_part_center(self, part: PlacedPart) -> pygame.math.Vector2:
        part_offset_rotated = part.relative_pos.rotate(-self.angle)
        return self.pos + part_offset_rotated
    def get_parts_near_world_pos(self, world_pos: pygame.math.Vector2, radius: float = 20.0) -> list[PlacedPart]:
        nearby_parts = []; radius_sq = radius * radius
        for part in self.parts:
             part_world_center = self.get_world_part_center(part)
             if (part_world_center - world_pos).length_squared() < radius_sq: nearby_parts.append(part)
        return nearby_parts
    def get_lowest_point_world(self) -> pygame.math.Vector2:
        if not self.parts: return self.pos
        lowest_y = -float('inf'); lowest_point_world = self.pos
        for part in self.parts:
             part_center_world = self.get_world_part_center(part)
             w = part.part_data.get('width', 0); h = part.part_data.get('height', 0)
             if w <= 0 or h <= 0: continue
             part_world_angle = self.angle
             corners_local = [pygame.math.Vector2(-w/2,-h/2),pygame.math.Vector2(w/2,-h/2),pygame.math.Vector2(w/2,h/2),pygame.math.Vector2(-w/2,h/2)]
             corners_world = [part_center_world + corner.rotate(-part_world_angle) for corner in corners_local]
             for corner in corners_world:
                 if corner.y > lowest_y: lowest_y = corner.y; lowest_point_world = corner
        return lowest_point_world if lowest_y > -float('inf') else self.get_world_com()
    def get_world_part_aabb(self, part: PlacedPart) -> pygame.Rect:
         part_data = part.part_data; w = part_data.get('width', 1); h = part_data.get('height', 1)
         world_center = self.get_world_part_center(part); max_dim = math.sqrt((w/2)**2 + (h/2)**2) * 2.1
         aabb = pygame.Rect(0, 0, max_dim, max_dim); aabb.center = world_center; return aabb
    # --- get_total_current_fuel (Unchanged) ---
    def get_total_current_fuel(self) -> float:
        return sum(tank.current_fuel for tank in self.fuel_tanks if not tank.is_broken)

    def get_thrust_and_consumption(self, dt: float) -> tuple[pygame.math.Vector2, pygame.math.Vector2 | None, dict[PlacedPart, float]]:
        """ Calculates potential thrust vector, application point offset, and fuel needs.
            For remote rockets, this might just return based on current known throttle/engine states,
            without checking local fuel (fuel sync happens via state updates).
            For local rockets, it checks fuel.
        """
        total_force_vector = pygame.math.Vector2(0, 0)
        thrust_torque_numerator = pygame.math.Vector2(0, 0)
        total_thrust_magnitude_applied = 0.0
        fuel_consumption_request: dict[PlacedPart, float] = {}
        active_engine_count_this_frame = 0

        # Reset firing status for all engines before calculation
        for engine in self.engines:
            self.engine_firing_status[engine] = False

        # If this rocket is NOT controlled locally, we might skip fuel check here,
        # assuming the controlling client/server manages fuel and sends updates.
        # However, for accurate physics simulation *locally* (even for remote objects),
        # it's better to calculate based on known state, including fuel.
        # The `update` loop will handle actual fuel consumption only for the local player.

        # Check master switch and throttle (relevant for both local & remote state)
        if not self.master_thrust_enabled or self.throttle_level <= 0:
            self.thrusting = False # Is the rocket *trying* to thrust
            return pygame.math.Vector2(0,0), None, {}

        # Iterate through engines
        for engine in self.engines:
            if engine.is_broken or not engine.engine_enabled:
                continue # Skip broken or manually disabled engines

            # Check fuel *only if this is the locally controlled rocket*
            # Remote rockets' fuel is updated via network messages
            can_fire_based_on_fuel = True
            if self.is_local_player:
                accessible_tanks = self.engine_fuel_sources.get(engine, [])
                available_fuel_for_engine = sum(tank.current_fuel for tank in accessible_tanks if not tank.is_broken and tank.current_fuel > 1e-6)
                if available_fuel_for_engine <= 1e-6:
                    can_fire_based_on_fuel = False # Local player has no fuel for this

            # If locally controlled and no fuel, skip thrust calc for this engine
            if self.is_local_player and not can_fire_based_on_fuel:
                self.engine_firing_status[engine] = False # Mark as not firing
                continue

            # --- Calculate potential thrust (assuming fuel is ok OR it's remote) ---
            engine_thrust_potential = engine.part_data.get("thrust", 0)
            engine_consumption_rate = engine.part_data.get("fuel_consumption", 0)
            thrust_this_engine = engine_thrust_potential * self.throttle_level
            fuel_needed_this_engine = engine_consumption_rate * self.throttle_level * dt

            thrust_direction = pygame.math.Vector2(0, -1).rotate(-self.angle)
            force_this_engine_vec = thrust_direction * thrust_this_engine
            total_force_vector += force_this_engine_vec

            engine_world_center = self.get_world_part_center(engine)
            world_com = self.get_world_com()
            engine_pos_offset_local_rotated = engine.relative_pos.rotate(-self.angle)
            thrust_torque_numerator += engine_pos_offset_local_rotated * thrust_this_engine
            total_thrust_magnitude_applied += thrust_this_engine
            active_engine_count_this_frame += 1

            # Request fuel consumption (only relevant for local player's update loop)
            if self.is_local_player:
                 fuel_consumption_request[engine] = fuel_needed_this_engine

            # Mark as potentially firing (actual status updated in main loop based on fuel draw for local)
            # For remote players, we assume it's firing if enabled/throttle on, until state update says otherwise.
            self.engine_firing_status[engine] = True

        # Calculate average thrust application point offset
        avg_thrust_application_point_offset = None
        if active_engine_count_this_frame > 0 and total_thrust_magnitude_applied > 1e-6:
            avg_thrust_application_point_offset = thrust_torque_numerator / total_thrust_magnitude_applied

        self.thrusting = active_engine_count_this_frame > 0

        return total_force_vector, avg_thrust_application_point_offset, fuel_consumption_request


    # --- Damage & Destruction Methods (Unchanged) ---
    # apply_collision_damage, handle_destroyed_parts
    # ... (These methods remain exactly the same as before) ...
    def apply_collision_damage(self, impact_velocity_magnitude, particle_manager: ParticleManager, specific_part_to_damage: PlacedPart | None = None):
        if impact_velocity_magnitude < MIN_IMPACT_VEL_DAMAGE: return
        base_damage = (impact_velocity_magnitude**1.8) * COLLISION_DAMAGE_FACTOR
        parts_to_damage = []; damage_multipliers = {}
        if specific_part_to_damage and specific_part_to_damage in self.parts:
            parts_to_damage = [specific_part_to_damage]; damage_multipliers[specific_part_to_damage] = 1.0
        elif not specific_part_to_damage and self.parts:
            lowest_world_point = self.get_lowest_point_world(); lowest_world_y = lowest_world_point.y
            world_com_y = self.get_world_com().y; rocket_impact_height = max(1.0, abs(lowest_world_y - world_com_y) * 1.5)
            for part in self.parts:
                part_center_y = self.get_world_part_center(part).y
                relative_y_from_bottom = lowest_world_y - part_center_y
                damage_factor = max(0.0, min(1.0, 1.0 - (relative_y_from_bottom / rocket_impact_height)))**2
                if damage_factor > 0.01: parts_to_damage.append(part); damage_multipliers[part] = damage_factor
            if not parts_to_damage and self.parts:
                 lowest_part = min(self.parts, key=lambda p: -self.get_world_part_center(p).y)
                 parts_to_damage = [lowest_part]; damage_multipliers[lowest_part] = 1.0
        parts_destroyed_this_impact = []
        for part in parts_to_damage:
            if part.is_broken: continue
            multiplier = damage_multipliers.get(part, 0.0); scaled_damage = base_damage * multiplier
            if scaled_damage < 0.1: continue
            part.current_hp -= scaled_damage
            if part.current_hp <= 0 and not part.is_broken:
                 part.is_broken = True; part.current_hp = 0; parts_destroyed_this_impact.append(part)
                 particle_manager.add_explosion(self.get_world_part_center(part))
        if parts_destroyed_this_impact: self.handle_destroyed_parts(parts_destroyed_this_impact)
    def handle_destroyed_parts(self, destroyed_parts: list[PlacedPart]):
        if not destroyed_parts: return
        original_part_count = len(self.parts); self.parts = [p for p in self.parts if p not in destroyed_parts]
        is_controlled_locally = self.is_local_player # Check if this instance is the player's direct control
        root_ref = self.original_root_part_ref
        root_destroyed = root_ref and (root_ref in destroyed_parts)

        # If the locally controlled rocket loses its root, player loses control
        if is_controlled_locally and root_destroyed:
             # self.has_active_control = False # (Replaced by checking is_local_player flag)
             print(f"[{self.sim_instance_id}] Player {self.player_id} lost control: Root part destroyed.")

        if not self.parts: self.is_active = False; return
        if len(self.parts) < original_part_count:
            self.needs_structural_update = True # Flag for split check AND internal recalc
            self.engines = [e for e in self.engines if e in self.parts]; self.fuel_tanks = [t for t in self.fuel_tanks if t in self.parts]
            self.parachutes = [pc for pc in self.parachutes if pc in self.parts]; self.separators = [s for s in self.separators if s in self.parts]
            self.engine_firing_status = {e: False for e in self.engines}

    # --- Part Activation Method (Only call for local player) ---
    def activate_part_at_pos(self, click_world_pos: pygame.math.Vector2):
        """ Handles activating parts via clicking *for the local player*.
            Returns the action details if an action was taken, else None.
        """
        # --- MP Change --- This should ONLY be called on the locally controlled rocket instance.
        if not self.is_local_player:
             print("Warning: activate_part_at_pos called on non-local rocket.")
             return None

        clicked_part: PlacedPart | None = None; min_dist_sq = 20**2
        for part in self.parts:
            if part.is_broken: continue
            dist_sq = (self.get_world_part_center(part) - click_world_pos).length_squared()
            if dist_sq < min_dist_sq: clicked_part = part; min_dist_sq = dist_sq
        if not clicked_part: return None

        part_type = clicked_part.part_data.get("type")
        is_activatable = clicked_part.part_data.get("activatable", False)
        action_taken = False
        action_details = None

        # --- MP Change --- Identify part uniquely for network message
        part_identifier = clicked_part.part_index # Use index within the rocket's parts list

        # Activatable parts (Parachutes, Separators)
        if is_activatable:
            if part_type == "Parachute" and not clicked_part.deployed:
                clicked_part.deployed = True; action_taken = True
                action_details = {"action": "deploy", "part_idx": part_identifier}
                print(f"Local Activate: Deploy Parachute idx {part_identifier}")
            elif part_type == "Separator" and not clicked_part.separated:
                # Mark visually immediately, add to pending, flag structure check
                clicked_part.separated = True
                if clicked_part not in self.pending_separation:
                     self.pending_separation.append(clicked_part)
                self.needs_structural_update = True # Flag potential split
                action_taken = True
                action_details = {"action": "separate", "part_idx": part_identifier}
                print(f"Local Activate: Separate idx {part_identifier}")

        # Toggleable parts (Engines)
        if not action_taken and part_type == "Engine":
            clicked_part.engine_enabled = not clicked_part.engine_enabled
            action_taken = True
            action_details = {"action": "toggle_engine", "part_idx": part_identifier, "enabled": clicked_part.engine_enabled}
            print(f"Local Activate: Toggle Engine idx {part_identifier} {'ON' if clicked_part.engine_enabled else 'OFF'}")

        # Return details for network transmission
        return action_details

    # --- MP Change --- Apply Action received from network ---
    def apply_network_action(self, action_data):
        """Applies an action (deploy, toggle, etc.) received over the network."""
        action_type = action_data.get("action")
        part_idx = action_data.get("part_idx")

        if part_idx is None or part_idx < 0 or part_idx >= len(self.parts):
            print(f"Warning: Received action for invalid part index {part_idx} on rocket {self.sim_instance_id}")
            return

        part = self.parts[part_idx]
        if part.is_broken:
             # print(f"Ignoring action '{action_type}' on broken part idx {part_idx}")
             return # Ignore actions on broken parts

        print(f"Applying network action '{action_type}' to part idx {part_idx} on rocket {self.sim_instance_id}")

        if action_type == "deploy" and part.part_data.get("type") == "Parachute":
            part.deployed = True
        elif action_type == "separate" and part.part_data.get("type") == "Separator":
            # Mark visually, add to pending list for processing in main loop (by host/locally)
            part.separated = True
            if part not in self.pending_separation:
                self.pending_separation.append(part)
            self.needs_structural_update = True # Flag potential split
        elif action_type == "toggle_engine" and part.part_data.get("type") == "Engine":
            part.engine_enabled = action_data.get("enabled", False)
        # --- MP Change --- Handle throttle and master thrust actions
        elif action_type == "set_throttle":
            self.throttle_level = max(0.0, min(1.0, action_data.get("value", 0.0)))
        elif action_type == "set_master_thrust":
            self.master_thrust_enabled = action_data.get("value", False)
        else:
            print(f"Warning: Received unknown network action type '{action_type}'")

    # --- Main Update Method ---
    # --- MP Change --- Added network_send_queue and current_time for throttling sends
    def update(self, dt, current_air_density, particle_manager: ParticleManager, network_send_queue: queue.Queue | None = None, current_time=0.0):
        """ Main physics and state update loop. Only the local player sends updates. """
        if not self.is_active or not self.parts: return

        # --- Structural Integrity Check & Recalculation ---
        if self.needs_structural_update:
             self._build_fuel_source_map()
             self.calculate_physics_properties()
             self.calculate_bounds()
             self.needs_structural_update = False # Reset flag AFTER recalc

        # --- Reset Forces/Torque ---
        self.acc = pygame.math.Vector2(0, 0); net_torque = 0.0
        current_world_com = self.get_world_com()
        velocity_sq = self.vel.length_squared()
        velocity_mag = math.sqrt(velocity_sq) if velocity_sq > 1e-9 else 0.0

        # --- Apply Gravity ---
        if self.total_mass > 0.01:
            gravity_force = pygame.math.Vector2(0, GRAVITY * self.total_mass)
            self.acc += gravity_force / self.total_mass

        # --- Calculate Potential Thrust & Fuel Needs ---
        thrust_force_potential, thrust_app_local_offset_rotated, fuel_consumption_request = self.get_thrust_and_consumption(dt)

        # --- Consume Fuel (ONLY for local player) ---
        total_fuel_drawn_this_frame = 0.0; mass_changed_by_fuel = False
        engines_actually_fired_this_frame = set() # Track engines that got fuel this frame

        if self.is_local_player and fuel_consumption_request:
            for engine, fuel_needed in fuel_consumption_request.items():
                if fuel_needed <= 1e-9: self.engine_firing_status[engine] = False; continue
                tanks = self.engine_fuel_sources.get(engine, []); valid_tanks = [t for t in tanks if not t.is_broken and t.current_fuel > 1e-9]
                available_fuel_for_engine = sum(tank.current_fuel for tank in valid_tanks)
                actual_fuel_to_draw = min(fuel_needed, available_fuel_for_engine)

                if actual_fuel_to_draw > 1e-9:
                    fuel_drawn_this_engine = 0.0
                    if available_fuel_for_engine > 1e-9:
                        for tank in valid_tanks:
                            proportion = tank.current_fuel / available_fuel_for_engine
                            draw_amount = min(actual_fuel_to_draw * proportion, tank.current_fuel)
                            tank.current_fuel -= draw_amount; fuel_drawn_this_engine += draw_amount
                            tank.current_fuel = max(0.0, tank.current_fuel)
                    total_fuel_drawn_this_frame += fuel_drawn_this_engine; mass_changed_by_fuel = True
                    engines_actually_fired_this_frame.add(engine) # Mark as fired
                    self.engine_firing_status[engine] = True # Update status used by thrust application below
                else:
                    self.engine_firing_status[engine] = False # Did not fire due to lack of fuel

            if mass_changed_by_fuel:
                self.calculate_physics_properties(); current_world_com = self.get_world_com()

        # --- Apply Actual Thrust Force & Torque ---
        actual_thrust_force_this_frame = pygame.math.Vector2(0, 0); actual_thrust_torque_numerator = pygame.math.Vector2(0, 0); actual_total_thrust_magnitude = 0.0
        # Use the engine_firing_status dict which was updated above (or set during get_thrust for remote)
        for engine, is_firing in self.engine_firing_status.items():
             if is_firing: # Only apply force if engine is marked as firing
                engine_thrust_potential = engine.part_data.get("thrust", 0)
                thrust_this_engine = engine_thrust_potential * self.throttle_level
                thrust_direction = pygame.math.Vector2(0, -1).rotate(-self.angle)
                force_this_engine_vec = thrust_direction * thrust_this_engine
                actual_thrust_force_this_frame += force_this_engine_vec
                engine_pos_offset_local_rotated = engine.relative_pos.rotate(-self.angle)
                actual_thrust_torque_numerator += engine_pos_offset_local_rotated * thrust_this_engine
                actual_total_thrust_magnitude += thrust_this_engine

        if self.total_mass > 0.01: self.acc += actual_thrust_force_this_frame / self.total_mass
        if actual_total_thrust_magnitude > 1e-6:
            avg_thrust_app_local_offset_rotated = actual_thrust_torque_numerator / actual_total_thrust_magnitude
            thrust_app_offset_from_com = (self.pos + avg_thrust_app_local_offset_rotated) - current_world_com
            net_torque += thrust_app_offset_from_com.cross(actual_thrust_force_this_frame)

        # --- Aerodynamics & Heating (Largely Unchanged, runs for all rockets) ---
        total_drag_force = pygame.math.Vector2(0, 0); parts_destroyed_by_heat = []
        current_max_temp = AMBIENT_TEMPERATURE
        for part in self.parts:
            if part.is_broken: continue
            part_data = part.part_data; part_world_center = self.get_world_part_center(part)
            drag_force_on_part = pygame.math.Vector2(0, 0)
            effective_area = (part_data['width'] + part_data['height']) / 2.0 * 0.1
            drag_coeff = part_data['base_drag_coeff']
            is_parachute = part_data['type'] == 'Parachute'; parachute_active = is_parachute and part.deployed and not self.landed
            if parachute_active: drag_coeff += part_data['deploy_drag']; effective_area += (part_data['width'] * part_data['deploy_area_factor']) * 0.1
            if current_air_density > AIR_DENSITY_VACUUM and velocity_sq > 0.1 and effective_area > 0 and drag_coeff > 0:
                drag_magnitude = 0.5 * current_air_density * velocity_sq * effective_area * drag_coeff
                if velocity_mag > 0.01:
                     drag_direction = -self.vel.normalize(); drag_force_on_part = drag_direction * drag_magnitude
                     total_drag_force += drag_force_on_part
                     drag_app_offset_from_com = part_world_center - current_world_com
                     net_torque += drag_app_offset_from_com.cross(drag_force_on_part)
            # Thermal Update
            heat_generated = 0.0
            if effective_area > 0 and velocity_mag > 50: heat_generated = AERO_HEATING_FACTOR * current_air_density * (velocity_mag**3) * effective_area * dt
            temp_diff = part.current_temp - AMBIENT_TEMPERATURE; cooling_rate = HEAT_DISSIPATION_FACTOR_VACUUM + HEAT_DISSIPATION_FACTOR_CONVECTION * current_air_density
            part_surface_area_approx = (part_data['width'] * part_data['height']) * 0.01
            heat_lost = cooling_rate * temp_diff * dt * part_surface_area_approx
            thermal_mass = part_data['thermal_mass']
            if thermal_mass > 1e-6: delta_temp = (heat_generated - heat_lost) / thermal_mass; part.current_temp += delta_temp; part.current_temp = max(AMBIENT_TEMPERATURE, part.current_temp)
            max_temp = part_data['max_temp']; part.is_overheating = part.current_temp > REENTRY_EFFECT_THRESHOLD_TEMP
            if part.current_temp > max_temp:
                overheat_amount = part.current_temp - max_temp; damage_factor = max(0.0, overheat_amount / OVERHEAT_DAMAGE_THRESHOLD_K)
                damage = OVERHEAT_DAMAGE_RATE * damage_factor * dt; part.current_hp -= damage
                if part.current_hp <= 0 and not part.is_broken:
                    part.is_broken = True; part.current_hp = 0; parts_destroyed_by_heat.append(part)
                    particle_manager.add_explosion(part_world_center)
            current_max_temp = max(current_max_temp, part.current_temp)
        self.max_temp_reading = current_max_temp
        if self.total_mass > 0.01: self.acc += total_drag_force / self.total_mass
        if parts_destroyed_by_heat: self.handle_destroyed_parts(parts_destroyed_by_heat) # Sets needs_structural_update

        # --- Control Input (Reaction Wheels - ONLY for local player) ---
        control_torque = 0.0
        can_control_locally = False
        if self.is_local_player:
             root_ref = self.original_root_part_ref
             can_control_locally = (root_ref is not None) and (root_ref in self.parts) and (not root_ref.is_broken)
             if can_control_locally:
                 keys = pygame.key.get_pressed()
                 if keys[pygame.K_LEFT] or keys[pygame.K_a]: control_torque += REACTION_WHEEL_TORQUE
                 if keys[pygame.K_RIGHT] or keys[pygame.K_d]: control_torque -= REACTION_WHEEL_TORQUE
                 # --- MP Change: Send control input actions ---
                 # For simplicity, send angular velocity directly? Or send torque input?
                 # Sending desired torque/turn direction might be better.
                 # Let's send periodic state updates instead for now. Controls affect state.

        # Add control torque to net torque (Only non-zero if local player)
        net_torque += control_torque

        # --- Physics Integration (Runs for all rockets) ---
        self.vel += self.acc * dt; self.pos += self.vel * dt
        if self.moment_of_inertia > 1e-6:
            angular_acceleration_rad = net_torque / self.moment_of_inertia
            angular_acceleration_deg = math.degrees(angular_acceleration_rad)
            self.angular_velocity += angular_acceleration_deg * dt
        self.angular_velocity *= (1.0 - ANGULAR_DAMPING * dt)
        self.angle = (self.angle + self.angular_velocity * dt) % 360

        # --- Ground Collision / Landing Logic (Runs for all rockets) ---
        lowest_point = self.get_lowest_point_world(); currently_on_ground = lowest_point.y >= GROUND_Y
        just_landed = currently_on_ground and not self.was_landed_last_frame
        just_took_off = not currently_on_ground and self.was_landed_last_frame
        if just_landed:
            impact_vel_mag = self.vel.length(); vertical_impact_threshold = MIN_IMPACT_VEL_DAMAGE * 0.5
            should_apply_damage = (self.vel.y > 1.0 and impact_vel_mag >= vertical_impact_threshold) or (impact_vel_mag >= MIN_IMPACT_VEL_DAMAGE)
            if should_apply_damage: self.apply_collision_damage(impact_vel_mag, particle_manager, specific_part_to_damage=None)
        if currently_on_ground:
            self.landed = True
            if self.is_active and self.parts:
                correction = lowest_point.y - GROUND_Y
                if correction > 0: self.pos.y -= correction
                self.vel.y = 0; self.vel.x *= 0.6; self.angular_velocity = 0
            else: self.vel = pygame.math.Vector2(0,0); self.angular_velocity = 0
        else: self.landed = False
        if just_took_off:
            destroyed_chutes_on_takeoff = []
            for chute in self.parachutes:
                if chute.deployed and not chute.is_broken:
                    chute.is_broken = True; chute.deployed = False; chute.current_hp = 0; destroyed_chutes_on_takeoff.append(chute)
                    particle_manager.add_explosion(self.get_world_part_center(chute), num_particles=5, max_life=0.3, max_speed=50)
            if destroyed_chutes_on_takeoff: self.handle_destroyed_parts(destroyed_chutes_on_takeoff)
        self.was_landed_last_frame = self.landed

        # --- MP Change: Send State Update (Local player only, throttled) ---
        if self.is_local_player and network_send_queue is not None:
             if current_time - self.last_state_send_time >= CLIENT_STATE_UPDATE_INTERVAL:
                 state_data = self.get_state()
                 state_msg = {
                     "type": network.MSG_TYPE_ROCKET_UPDATE,
                     "pid": self.player_id,
                     "action": "state_update", # Specific action type for full state
                     "data": state_data
                 }
                 network_send_queue.put(state_msg)
                 self.last_state_send_time = current_time


    # --- MP Change: State Synchronization Methods ---
    def get_state(self) -> dict:
        """ Gathers the essential state of the rocket for network synchronization. """
        part_states = []
        for i, part in enumerate(self.parts):
            part_states.append({
                "idx": i, # Use index for identification within this rocket instance
                # "part_id": part.part_id, # Not strictly needed if structure doesn't change often? Maybe add later.
                "hp": part.current_hp,
                "temp": part.current_temp,
                "fuel": part.current_fuel if part.part_data.get("type") == "FuelTank" else 0.0,
                "enabled": getattr(part, 'engine_enabled', None), # Include if exists
                "deployed": getattr(part, 'deployed', False),
                "separated": getattr(part, 'separated', False),
                "broken": part.is_broken,
                "overheating": part.is_overheating,
            })

        state = {
            "pos_x": self.pos.x, "pos_y": self.pos.y,
            "vel_x": self.vel.x, "vel_y": self.vel.y,
            "angle": self.angle,
            "ang_vel": self.angular_velocity,
            "throttle": self.throttle_level,
            "master_thrust": self.master_thrust_enabled,
            "landed": self.landed,
            "thrusting_status": {e.part_index: firing for e, firing in self.engine_firing_status.items()}, # Send firing status
            "parts": part_states
        }
        return state

    def apply_state(self, state_data: dict):
        """ Applies state received from the network to this rocket instance. """
        # --- MP Change --- Do not apply state to the locally controlled rocket, only remotes.
        if self.is_local_player:
             # Maybe apply *some* state if host correction is needed? Complex.
             # For now, local player state is authoritative locally.
             # print("Skipping apply_state for local player.")
             return

        self.pos.x = state_data.get("pos_x", self.pos.x)
        self.pos.y = state_data.get("pos_y", self.pos.y)
        self.vel.x = state_data.get("vel_x", self.vel.x)
        self.vel.y = state_data.get("vel_y", self.vel.y)
        self.angle = state_data.get("angle", self.angle)
        self.angular_velocity = state_data.get("ang_vel", self.angular_velocity)
        self.throttle_level = state_data.get("throttle", self.throttle_level)
        self.master_thrust_enabled = state_data.get("master_thrust", self.master_thrust_enabled)
        self.landed = state_data.get("landed", self.landed)
        self.was_landed_last_frame = self.landed # Assume consistency

        # Apply engine firing status
        firing_status = state_data.get("thrusting_status", {})
        self.engine_firing_status.clear() # Clear old status
        for engine in self.engines:
            # Use part index from the status dict keys (convert back to int if needed)
            engine_idx_str = str(engine.part_index)
            self.engine_firing_status[engine] = firing_status.get(engine_idx_str, False) # Default to False if not in update
            # Alternative if keys are ints:
            # self.engine_firing_status[engine] = firing_status.get(engine.part_index, False)

        # Update part states
        part_states = state_data.get("parts", [])
        part_map = {p.part_index: p for p in self.parts} # Map index to part object

        for p_state in part_states:
            part_idx = p_state.get("idx")
            part = part_map.get(part_idx)
            if part:
                part.current_hp = p_state.get("hp", part.current_hp)
                part.current_temp = p_state.get("temp", part.current_temp)
                part.is_broken = p_state.get("broken", part.is_broken)
                part.is_overheating = p_state.get("overheating", part.is_overheating)
                # Fuel state
                if part.part_data.get("type") == "FuelTank":
                    part.current_fuel = max(0.0, min(p_state.get("fuel", part.current_fuel), part.fuel_capacity))
                # Enabled/Deployed/Separated states
                if hasattr(part, 'engine_enabled') and p_state.get("enabled") is not None:
                    part.engine_enabled = p_state["enabled"]
                if hasattr(part, 'deployed') and p_state.get("deployed") is not None:
                    part.deployed = p_state["deployed"]
                if hasattr(part, 'separated') and p_state.get("separated") is not None:
                    part.separated = p_state["separated"]
            else:
                print(f"Warning: Received state for unknown part index {part_idx} on rocket {self.sim_instance_id}")

        # If structure changed (parts became broken), might need internal recalc
        old_part_count = len(self.parts)
        self.parts = [p for p in self.parts if not p.is_broken] # Filter broken parts based on received state
        if len(self.parts) < old_part_count:
             self.needs_structural_update = True # Flag for internal recalc on next update
             # Update component lists immediately after applying state
             self.engines = [e for e in self.engines if e in self.parts]; self.fuel_tanks = [t for t in self.fuel_tanks if t in self.parts]
             self.parachutes = [pc for pc in self.parachutes if pc in self.parts]; self.separators = [s for s in self.separators if s in self.parts]
             self.engine_firing_status = {e: firing_status.get(str(e.part_index), False) for e in self.engines} # Rebuild firing status for remaining engines


    # --- Drawing Method ---
    # --- MP Change --- Added optional player name drawing
    def draw(self, surface, camera, particle_manager: ParticleManager, draw_name=False):
        """ Draws the rocket parts and effects. Returns count of visually broken parts. """
        num_broken_visually = 0
        if not self.is_active: return 0

        # --- Reentry/Overheating Effects (Unchanged) ---
        spawn_reentry_particles = False; hottest_part_temp = AMBIENT_TEMPERATURE; hottest_part_max_temp = AMBIENT_TEMPERATURE; has_overheating_parts = False
        for part in self.parts:
             if not part.is_broken:
                 hottest_part_max_temp = max(hottest_part_max_temp, part.part_data.get('max_temp', DEFAULT_MAX_TEMP))
                 if part.is_overheating: spawn_reentry_particles = True; hottest_part_temp = max(hottest_part_temp, part.current_temp); has_overheating_parts = True
        if spawn_reentry_particles and self.vel.length_squared() > 50**2:
             reentry_temp_range = max(1.0, (hottest_part_max_temp * REENTRY_EFFECT_MAX_TEMP_SCALE) - REENTRY_EFFECT_THRESHOLD_TEMP)
             intensity = max(0.0, min(1.0, (hottest_part_temp - REENTRY_EFFECT_THRESHOLD_TEMP) / reentry_temp_range ))
             leading_point = self.get_lowest_point_world(); num_sparks = int(1 + intensity * 5)
             for _ in range(num_sparks):
                 spark_pos = leading_point + pygame.math.Vector2(random.uniform(-5, 5), random.uniform(-5, 5))
                 particle_manager.add_reentry_spark(spark_pos, self.vel, intensity)

        # --- Draw Individual Parts ---
        for part in self.parts:
            part_center_world = self.get_world_part_center(part); part_screen_pos = camera.apply(part_center_world)
            part_world_angle = self.angle; indicator_color = None; is_activatable = part.part_data.get("activatable", False)
            part_type = part.part_data.get("type"); is_parachute = part_type == "Parachute"; is_engine = part_type == "Engine"; is_separator = part_type == "Separator"
            show_deployed_visual = is_parachute and part.deployed and not part.is_broken and not self.landed
            heat_factor = 0.0
            if part.is_overheating and not part.is_broken:
                max_temp_visual = part.part_data.get('max_temp', DEFAULT_MAX_TEMP) * REENTRY_EFFECT_MAX_TEMP_SCALE
                heat_glow_range = max(1.0, max_temp_visual - REENTRY_EFFECT_THRESHOLD_TEMP)
                heat_factor = max(0.0, min(1.0, (part.current_temp - REENTRY_EFFECT_THRESHOLD_TEMP) / heat_glow_range))
            if part.is_broken: num_broken_visually += 1
            elif is_parachute and is_activatable: indicator_color = COLOR_ACTIVATABLE_USED if part.deployed else COLOR_ACTIVATABLE_READY
            elif is_engine: indicator_color = COLOR_ENGINE_ENABLED if part.engine_enabled else COLOR_ENGINE_DISABLED
            elif is_separator and is_activatable: indicator_color = COLOR_ACTIVATABLE_USED if part.separated else COLOR_ACTIVATABLE_READY
            try:
                draw_part_shape(surface, part.part_data, part_screen_pos, part_world_angle, broken=part.is_broken, deployed=show_deployed_visual, heat_factor=heat_factor)
            except NameError: fallback_color = RED if part.is_broken else GREEN; pygame.draw.circle(surface, fallback_color, part_screen_pos, 5)
            except ValueError: pass # Ignore potential drawing errors
            if indicator_color: pygame.draw.circle(surface, indicator_color, part_screen_pos, 5); pygame.draw.circle(surface, BLACK, part_screen_pos, 5, 1)

        # --- Draw Engine Flames ---
        # Use self.thrusting (overall attempt flag) AND self.engine_firing_status (actual fuel success)
        if self.thrusting: # Check if trying to thrust
            flame_scale = 0.5 + 0.5 * self.throttle_level
            for engine, is_firing in self.engine_firing_status.items():
                if is_firing and not engine.is_broken: # Check if THIS engine actually fired
                    engine_center_world = self.get_world_part_center(engine); engine_world_angle = self.angle
                    flame_base_offset_local = pygame.math.Vector2(0, engine.part_data["height"] / 2.0)
                    flame_base_offset_rotated = flame_base_offset_local.rotate(-engine_world_angle)
                    flame_base_world = engine_center_world + flame_base_offset_rotated
                    flame_length = (15 + random.uniform(-2, 2)) * flame_scale; flame_width = engine.part_data["width"] * 0.8 * flame_scale
                    flame_dir_world = pygame.math.Vector2(0, 1).rotate(-engine_world_angle); flame_side_world = pygame.math.Vector2(1, 0).rotate(-engine_world_angle)
                    flame_tip_world = flame_base_world + flame_dir_world * flame_length; flame_left_world = flame_base_world - flame_side_world * flame_width / 2.0; flame_right_world = flame_base_world + flame_side_world * flame_width / 2.0
                    flame_points_screen = [camera.apply(p) for p in [flame_left_world, flame_right_world, flame_tip_world]]
                    try: pygame.draw.polygon(surface, COLOR_FLAME, flame_points_screen)
                    except NameError: pygame.draw.line(surface, RED, camera.apply(flame_base_world), camera.apply(flame_tip_world), 3)
                    except ValueError: pass

        # --- MP Change: Draw Player Name Tag ---
        if draw_name and self.parts: # Only draw if requested and rocket has parts
             # Find highest point (lowest Y) to draw name above
             highest_world_y = float('inf')
             highest_point_world_x = self.get_world_com().x # Default X to CoM
             for part in self.parts:
                 part_center_world = self.get_world_part_center(part)
                 half_h = part.part_data.get('height', 0) / 2.0
                 # Simple AABB top estimation
                 part_top_y = part_center_world.y - half_h
                 if part_top_y < highest_world_y:
                      highest_world_y = part_top_y
                      highest_point_world_x = part_center_world.x # Use X of highest part

             name_pos_world = pygame.math.Vector2(highest_point_world_x, highest_world_y - 20) # Offset above highest point
             name_pos_screen = camera.apply(name_pos_world)

             # Basic screen bounds check
             if 0 < name_pos_screen.x < SCREEN_WIDTH and 0 < name_pos_screen.y < SCREEN_HEIGHT:
                 font = pygame.font.SysFont(None, 18)
                 name_color = WHITE if self.is_local_player else (180, 220, 255) # Highlight local player differently
                 name_surf = font.render(f"{self.player_name}", True, name_color) # Use stored player name
                 name_rect = name_surf.get_rect(center=name_pos_screen)
                 # Optional background for readability
                 bg_rect = name_rect.inflate(6, 4)
                 bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
                 bg_surf.fill((0, 0, 0, 120))
                 surface.blit(bg_surf, bg_rect)
                 surface.blit(name_surf, name_rect)


        return num_broken_visually

    # --- Helper for calculating CoM of a subassembly (Unchanged) ---
    # calculate_subassembly_world_com
    # ... (This method remains exactly the same as before) ...
    def calculate_subassembly_world_com(self, assembly_parts: list[PlacedPart]) -> pygame.math.Vector2:
        if not assembly_parts: return self.pos # Fallback
        com_numerator = pygame.math.Vector2(0, 0); total_assembly_mass = 0.0
        for part in assembly_parts:
            if not part.part_data: continue
            part_mass_static = float(part.part_data.get("mass", 0.0))
            if part_mass_static <= 1e-6: continue
            part_world_center = self.get_world_part_center(part) # Use current world center
            com_numerator += part_world_center * part_mass_static; total_assembly_mass += part_mass_static
        if total_assembly_mass <= 1e-6:
             if assembly_parts: return self.get_world_part_center(assembly_parts[0]) # Use first part's center
             else: return self.pos # Fallback again
        else: return com_numerator / total_assembly_mass


# --- Background/Terrain Functions (Unchanged) ---
# create_stars, get_air_density, draw_earth_background, draw_stars, draw_terrain
# ... (These functions remain exactly the same as before) ...
def create_stars(count, bounds: pygame.Rect):
    stars = []; depth_range = bounds.height
    for _ in range(count):
        x = random.uniform(bounds.left, bounds.right); y = random.uniform(bounds.top, bounds.bottom)
        z = random.uniform(1, max(2, depth_range)); stars.append((pygame.math.Vector2(x, y), z))
    return stars
def get_air_density(altitude_agl):
    scale_height = ATMOSPHERE_SCALE_HEIGHT
    if altitude_agl < 0: return AIR_DENSITY_SEA_LEVEL
    elif 0 <= altitude_agl <= ATMOSPHERE_EXP_LIMIT:
        density = AIR_DENSITY_SEA_LEVEL * math.exp(-altitude_agl / scale_height)
        return max(AIR_DENSITY_VACUUM, density)
    elif ATMOSPHERE_EXP_LIMIT < altitude_agl <= ATMOSPHERE_LINEAR_LIMIT:
        density_at_35k = max(AIR_DENSITY_VACUUM, AIR_DENSITY_SEA_LEVEL * math.exp(-ATMOSPHERE_EXP_LIMIT / scale_height))
        density_at_70k_target = AIR_DENSITY_SEA_LEVEL * ATMOSPHERE_TARGET_DENSITY_FACTOR
        interp_factor = max(0.0, min(1.0, (altitude_agl - ATMOSPHERE_EXP_LIMIT) / (ATMOSPHERE_LINEAR_LIMIT - ATMOSPHERE_EXP_LIMIT)))
        density = density_at_35k * (1.0 - interp_factor) + density_at_70k_target * interp_factor
        return max(AIR_DENSITY_VACUUM, density)
    else: return AIR_DENSITY_VACUUM
def draw_earth_background(surface, camera, stars):
    screen_rect = surface.get_rect(); avg_world_y = camera.offset.y + camera.height / 2
    ground_screen_y = camera.apply(pygame.math.Vector2(0, GROUND_Y)).y
    try: _ = BLUE_SKY_Y_LIMIT; _ = SPACE_Y_LIMIT; _ = COLOR_SKY_BLUE; _ = COLOR_SPACE_BLACK
    except NameError: BLUE_SKY_Y_LIMIT=-2000; SPACE_Y_LIMIT=-15000; COLOR_SKY_BLUE=(135,206,250); COLOR_SPACE_BLACK=(0,0,0)
    if avg_world_y > BLUE_SKY_Y_LIMIT:
        if ground_screen_y < screen_rect.bottom: pygame.draw.rect(surface, COLOR_SKY_BLUE, (0, 0, screen_rect.width, ground_screen_y))
        else: surface.fill(COLOR_SKY_BLUE)
    elif avg_world_y < SPACE_Y_LIMIT: surface.fill(COLOR_SPACE_BLACK); draw_stars(surface, stars, camera, alpha=255)
    else:
        interp = max(0.0, min(1.0, (avg_world_y - BLUE_SKY_Y_LIMIT) / (SPACE_Y_LIMIT - BLUE_SKY_Y_LIMIT)))
        bg_color = pygame.Color(COLOR_SKY_BLUE).lerp(COLOR_SPACE_BLACK, interp); surface.fill(bg_color)
        star_alpha = int(255 * interp)
        if star_alpha > 10: draw_stars(surface, stars, camera, alpha=star_alpha)
def draw_stars(surface, stars, camera, alpha=255):
    if alpha <= 0: return
    screen_rect = surface.get_rect(); base_color = pygame.Color(200, 200, 200)
    try: depth_scaling = STAR_FIELD_DEPTH
    except NameError: depth_scaling = 10000
    for world_pos, z in stars:
        parallax_factor = 1.0 / ((z / (depth_scaling / 20.0)) + 1.0)
        effective_camera_offset = camera.offset * parallax_factor; screen_pos = world_pos - effective_camera_offset
        sx, sy = int(screen_pos.x), int(screen_pos.y)
        if 0 <= sx < screen_rect.width and 0 <= sy < screen_rect.height:
            size = max(1, int(2.5 * (1.0 - z / max(1, depth_scaling))))
            alpha_factor = alpha / 255.0
            final_color_tuple = (int(base_color.r*alpha_factor), int(base_color.g*alpha_factor), int(base_color.b*alpha_factor))
            if final_color_tuple != (0,0,0):
                 try: pygame.draw.circle(surface, final_color_tuple, (sx, sy), size)
                 except ValueError: pass
def draw_terrain(surface, camera):
    world_width = WORLD_WIDTH; ground_y = GROUND_Y; ground_color = COLOR_GROUND
    view_rect_world = pygame.Rect(camera.offset.x-world_width, ground_y, camera.width+world_width*2, SCREEN_HEIGHT*2)
    rect_screen = camera.apply_rect(view_rect_world); pygame.draw.rect(surface, ground_color, rect_screen)


# --- Single Player Simulation Runner (Largely Unchanged) ---
def run_simulation(screen, clock, blueprint_file):
    print(f"--- Starting Simulation (Single Player) ---")
    print(f"Loading blueprint: {blueprint_file}")
    if not os.path.exists(blueprint_file): print(f"Error: Blueprint file not found: {blueprint_file}"); return
    initial_blueprint = RocketBlueprint.load_from_json(blueprint_file)
    if not initial_blueprint or not initial_blueprint.parts: print("Blueprint load failed or is empty."); return

    all_rockets: list[FlyingRocket] = []
    controlled_rocket: FlyingRocket | None = None
    next_sim_id = 0
    collision_grace_period_pairs: dict[tuple[int, int], int] = {}
    current_sim_frame = 0

    original_root_part_instance = None
    if initial_blueprint.parts:
        for part in initial_blueprint.parts:
             if part.part_data and part.part_data.get("type") == "CommandPod": original_root_part_instance = part; break
        if not original_root_part_instance: original_root_part_instance = initial_blueprint.parts[0]

    initial_subassemblies = initial_blueprint.find_connected_subassemblies()
    if not initial_subassemblies: print("Error: No connected parts found."); return

    initial_spawn_y_offset = 0
    for i, assembly_parts in enumerate(initial_subassemblies):
        if not assembly_parts: continue
        temp_bp_for_calc = RocketBlueprint(); temp_bp_for_calc.parts = assembly_parts
        initial_com_local = temp_bp_for_calc.calculate_subassembly_world_com(assembly_parts)
        lowest_offset_y = temp_bp_for_calc.get_lowest_point_offset_y()
        start_x = i * 50
        start_y_for_origin = (GROUND_Y - initial_spawn_y_offset) - lowest_offset_y
        target_initial_com_pos = pygame.math.Vector2(start_x, start_y_for_origin + initial_com_local.y)
        contains_original_root = original_root_part_instance and (original_root_part_instance in assembly_parts)
        is_primary = (controlled_rocket is None and contains_original_root) or (controlled_rocket is None and i == 0)
        try:
            rocket_instance = FlyingRocket(
                parts_list=list(assembly_parts), initial_world_com_pos=target_initial_com_pos,
                initial_angle=0, initial_vel=pygame.math.Vector2(0,0), sim_instance_id=next_sim_id,
                is_primary_control=is_primary, original_root_ref=original_root_part_instance,
                current_frame=current_sim_frame,
                player_id=0, player_name="LocalPlayer" # SP uses Player ID 0
            )
            all_rockets.append(rocket_instance)
            if is_primary: controlled_rocket = rocket_instance
            else: rocket_instance.is_local_player = False # Ensure debris isn't marked local
            next_sim_id += 1
        except Exception as e: print(f"Error initializing SP rocket instance {next_sim_id}: {e}"); import traceback; traceback.print_exc()
    if controlled_rocket is None and all_rockets: controlled_rocket = all_rockets[0]; controlled_rocket.is_local_player = True # Ensure fallback has control

    camera = Camera(SCREEN_WIDTH, SCREEN_HEIGHT)
    if controlled_rocket: camera.update(controlled_rocket.get_world_com())
    elif all_rockets: camera.update(all_rockets[0].get_world_com())
    else: camera.update(pygame.math.Vector2(0, GROUND_Y - SCREEN_HEIGHT // 3))

    try:
        star_area_bounds = pygame.Rect(-WORLD_WIDTH*2, SPACE_Y_LIMIT - STAR_FIELD_DEPTH, WORLD_WIDTH*4, abs(SPACE_Y_LIMIT) + GROUND_Y + STAR_FIELD_DEPTH * 1.5)
        stars = create_stars(STAR_COUNT, star_area_bounds) # Use constant
    except NameError: stars = []
    ui_font = pygame.font.SysFont(None, 20); ui_font_large = pygame.font.SysFont(None, 36)
    particle_manager = ParticleManager()
    sim_running = True; last_respawn_time = time.time()

    while sim_running:
        dt = min(clock.tick(60) / 1000.0, 0.05)
        current_sim_frame += 1
        newly_created_rockets_this_frame: list[FlyingRocket] = []
        rockets_to_remove_this_frame: list[FlyingRocket] = []

        # --- Update Collision Grace Period ---
        pairs_to_remove_from_grace = []
        for pair_key in list(collision_grace_period_pairs.keys()):
            if pair_key in collision_grace_period_pairs:
                collision_grace_period_pairs[pair_key] -= 1
                if collision_grace_period_pairs[pair_key] <= 0: pairs_to_remove_from_grace.append(pair_key)
        for pair_key in pairs_to_remove_from_grace:
            if pair_key in collision_grace_period_pairs: del collision_grace_period_pairs[pair_key]

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: sim_running = False
                if controlled_rocket and controlled_rocket.is_local_player:
                    if event.key == pygame.K_SPACE: controlled_rocket.master_thrust_enabled = not controlled_rocket.master_thrust_enabled
                    if event.key == pygame.K_p:
                        for chute in controlled_rocket.parachutes:
                            if not chute.deployed and not chute.is_broken: chute.deployed = True
                current_time = time.time()
                if event.key == pygame.K_r and (current_time - last_respawn_time > 1.0):
                    print("--- RESPAWNING ROCKET (SP) ---"); last_respawn_time = current_time
                    # --- Reset state and re-initialize (Simplified version of MP respawn) ---
                    all_rockets.clear(); controlled_rocket = None; particle_manager.particles.clear()
                    collision_grace_period_pairs.clear(); next_sim_id = 0; current_sim_frame = 0
                    reloaded_blueprint = RocketBlueprint.load_from_json(blueprint_file)
                    if reloaded_blueprint and reloaded_blueprint.parts:
                        original_root_part_instance = None # Redetermine root ref
                        for part in reloaded_blueprint.parts:
                            part.current_hp = part.part_data.get("max_hp", 100); part.is_broken = False; part.engine_enabled = True
                            part.deployed = False; part.separated = False; part.current_temp = AMBIENT_TEMPERATURE; part.is_overheating = False
                            if part.part_data.get("type") == "FuelTank": part.current_fuel = part.fuel_capacity
                            if part.part_data.get("type") == "CommandPod": original_root_part_instance = part
                        if not original_root_part_instance and reloaded_blueprint.parts: original_root_part_instance = reloaded_blueprint.parts[0]

                        initial_subassemblies = reloaded_blueprint.find_connected_subassemblies()
                        for i, assembly_parts in enumerate(initial_subassemblies): # Repeat initialization logic
                            if not assembly_parts: continue
                            temp_bp = RocketBlueprint(); temp_bp.parts = assembly_parts
                            initial_com_local = temp_bp.calculate_subassembly_world_com(assembly_parts); lowest_offset_y = temp_bp.get_lowest_point_offset_y()
                            start_x = i * 50; start_y_for_origin = (GROUND_Y - initial_spawn_y_offset) - lowest_offset_y
                            target_initial_com_pos = pygame.math.Vector2(start_x, start_y_for_origin + initial_com_local.y)
                            contains_original_root = original_root_part_instance and (original_root_part_instance in assembly_parts)
                            is_primary = (controlled_rocket is None and contains_original_root) or (controlled_rocket is None and i == 0)
                            try: # Recreate rocket instances
                                rocket_instance = FlyingRocket(list(assembly_parts), target_initial_com_pos, 0, pygame.math.Vector2(0,0), next_sim_id, is_primary, original_root_part_instance, current_sim_frame, 0, "LocalPlayer")
                                newly_created_rockets_this_frame.append(rocket_instance) # Add to temp list
                                if is_primary: controlled_rocket = rocket_instance
                                else: rocket_instance.is_local_player = False
                                next_sim_id += 1
                            except Exception as e: print(f"Respawn Error (SP): {e}")
                        if controlled_rocket is None and newly_created_rockets_this_frame: controlled_rocket = newly_created_rockets_this_frame[0]; controlled_rocket.is_local_player = True
                        all_rockets.extend(newly_created_rockets_this_frame); newly_created_rockets_this_frame.clear() # Add new rockets
                        print("Respawn Complete (SP).")
                    else: print("Respawn Failed (SP): Cannot reload blueprint.")

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                 if controlled_rocket and controlled_rocket.is_local_player:
                     click_world_pos = pygame.math.Vector2(event.pos) + camera.offset
                     # Activate part directly (no network needed in SP)
                     controlled_rocket.activate_part_at_pos(click_world_pos)

        # --- Continuous Controls (SP) ---
        if controlled_rocket and controlled_rocket.is_local_player:
            keys = pygame.key.get_pressed(); throttle_change = 0.0
            if keys[pygame.K_w] or keys[pygame.K_UP]: throttle_change += THROTTLE_CHANGE_RATE * dt
            if keys[pygame.K_s] or keys[pygame.K_DOWN]: throttle_change -= THROTTLE_CHANGE_RATE * dt
            if throttle_change != 0: controlled_rocket.throttle_level = max(0.0, min(1.0, controlled_rocket.throttle_level + throttle_change))

        # --- Update All Active Rockets (SP) ---
        for rocket in all_rockets:
            if not rocket.is_active: continue
            altitude_agl = max(0, GROUND_Y - rocket.get_world_com().y); current_air_density = get_air_density(altitude_agl)
            # In SP, network_send_queue is None
            rocket.update(dt, current_air_density, particle_manager, network_send_queue=None, current_time=time.time())
            if not rocket.is_active:
                 if rocket not in rockets_to_remove_this_frame: rockets_to_remove_this_frame.append(rocket)

        # --- Inter-Rocket Collision (SP) ---
        collision_pairs_processed_this_frame = set()
        for i in range(len(all_rockets)):
            r1 = all_rockets[i];
            if r1 in rockets_to_remove_this_frame or not r1.is_active or not r1.parts: continue
            for j in range(i + 1, len(all_rockets)):
                r2 = all_rockets[j];
                if r2 in rockets_to_remove_this_frame or not r2.is_active or not r2.parts: continue
                pair_key_grace = tuple(sorted((r1.sim_instance_id, r2.sim_instance_id)))
                if pair_key_grace in collision_grace_period_pairs: continue # Skip grace period
                dist_sq = (r1.get_world_com() - r2.get_world_com()).length_squared()
                r1_radius_approx = max(r1.local_bounds.width, r1.local_bounds.height)/2.0; r2_radius_approx = max(r2.local_bounds.width, r2.local_bounds.height)/2.0
                if dist_sq > (r1_radius_approx + r2_radius_approx + 10)**2: continue # Broad phase fail
                collision_found = False
                colliding_p1 = None
                colliding_p2 = None
                for p1 in r1.parts:
                    if p1.is_broken: continue  # Skip broken parts
                    # --- FIX: Add the missing line below ---
                    rect1 = r1.get_world_part_aabb(p1)  # Calculate AABB for p1

                    for p2 in r2.parts:
                        if p2.is_broken: continue  # Skip broken parts
                        rect2 = r2.get_world_part_aabb(p2)  # Calculate AABB for p2
                        # Now rect1 is guaranteed to be defined here
                        if rect1.colliderect(rect2):
                            collision_found = True;
                            colliding_p1 = p1;
                            colliding_p2 = p2;
                            break
                    if collision_found: break
                if collision_found: # Response
                    pair_key_processed = tuple(sorted((r1.sim_instance_id, r2.sim_instance_id)))
                    if pair_key_processed in collision_pairs_processed_this_frame: continue
                    collision_pairs_processed_this_frame.add(pair_key_processed)
                    relative_velocity = r1.vel - r2.vel; impact_speed = relative_velocity.length()
                    r1.apply_collision_damage(impact_speed, particle_manager, colliding_p1); r2.apply_collision_damage(impact_speed, particle_manager, colliding_p2)
                    collision_normal = r1.get_world_com() - r2.get_world_com() # Simple push
                    if collision_normal.length_squared() > 1e-6: collision_normal.normalize_ip()
                    else: collision_normal = pygame.math.Vector2(0, -1)
                    push_strength = 2.0; total_m = r1.total_mass + r2.total_mass
                    if total_m > 0.01: r1.pos += collision_normal*push_strength*(r2.total_mass/total_m); r2.pos -= collision_normal*push_strength*(r1.total_mass/total_m)


        # --- Process Connectivity Checks and Separations (SP) ---
        rockets_to_process_for_splits = list(all_rockets)
        new_rockets_created_in_split_phase: list[FlyingRocket] = []
        for rocket in rockets_to_process_for_splits:
            if rocket in rockets_to_remove_this_frame or not rocket.is_active: continue
            processed_split = False; split_siblings: list[FlyingRocket] = []
            # Check splits from destruction
            if rocket.needs_structural_update and not rocket.pending_separation:
                temp_bp = RocketBlueprint(); temp_bp.parts = rocket.parts; subassemblies = temp_bp.find_connected_subassemblies()
                if len(subassemblies) > 1:
                    processed_split = True; print(f"[SP:{rocket.sim_instance_id}] SPLIT (Destruction) into {len(subassemblies)} pieces!")
                    if rocket not in rockets_to_remove_this_frame: rockets_to_remove_this_frame.append(rocket)
                    original_throttle = rocket.throttle_level; original_master_thrust = rocket.master_thrust_enabled
                    for assembly_parts in subassemblies: # Create new fragments
                        if not assembly_parts: continue
                        try:
                            sub_com_world = rocket.calculate_subassembly_world_com(assembly_parts)
                            contains_root = rocket.original_root_part_ref and (rocket.original_root_part_ref in assembly_parts)
                            is_primary = rocket.is_local_player and contains_root # Transfer control only if local player had it
                            new_frag = FlyingRocket(list(assembly_parts), sub_com_world, rocket.angle, rocket.vel, next_sim_id, is_primary, rocket.original_root_part_ref, current_sim_frame, 0, "LocalPlayer")
                            new_frag.angular_velocity = rocket.angular_velocity
                            if new_frag.is_local_player: new_frag.throttle_level = original_throttle; new_frag.master_thrust_enabled = original_master_thrust
                            new_rockets_created_in_split_phase.append(new_frag); split_siblings.append(new_frag); next_sim_id += 1
                        except Exception as e: print(f"Error SP split(dest): {e}")
            # Check splits from separators
            elif rocket.pending_separation and not processed_split:
                separators_to_process = list(rocket.pending_separation); rocket.pending_separation.clear(); split_by_sep = False
                current_parts = list(rocket.parts); original_throttle = rocket.throttle_level; original_master_thrust = rocket.master_thrust_enabled
                for sep_part in separators_to_process:
                    if sep_part not in current_parts: continue
                    sep_world_pos = rocket.get_world_part_center(sep_part); sep_force = sep_part.part_data.get("separation_force", 1000)
                    parts_without_sep = [p for p in current_parts if p != sep_part]
                    temp_bp = RocketBlueprint(); temp_bp.parts = parts_without_sep; subassemblies = temp_bp.find_connected_subassemblies()
                    if len(subassemblies) > 1: # Split occurred
                        split_by_sep = True; processed_split = True; print(f"[SP:{rocket.sim_instance_id}] SPLIT by Separator {sep_part.part_id} into {len(subassemblies)} pieces!")
                        if rocket not in rockets_to_remove_this_frame: rockets_to_remove_this_frame.append(rocket)
                        current_parts = [] # Original replaced
                        for assembly_parts in subassemblies: # Create new pieces
                            if not assembly_parts: continue
                            try:
                                sub_com_world = rocket.calculate_subassembly_world_com(assembly_parts)
                                contains_root = rocket.original_root_part_ref and (rocket.original_root_part_ref in assembly_parts)
                                is_primary = rocket.is_local_player and contains_root
                                new_sep = FlyingRocket(list(assembly_parts), sub_com_world, rocket.angle, rocket.vel, next_sim_id, is_primary, rocket.original_root_part_ref, current_sim_frame, 0, "LocalPlayer")
                                new_sep.angular_velocity = rocket.angular_velocity
                                if new_sep.is_local_player: new_sep.throttle_level = original_throttle; new_sep.master_thrust_enabled = original_master_thrust
                                # Apply impulse
                                sep_vec = new_sep.get_world_com() - sep_world_pos
                                if sep_vec.length_squared() > 1e-6: sep_dir = sep_vec.normalize()
                                else: sep_dir = pygame.math.Vector2(0, -1).rotate(-rocket.angle + random.uniform(-5, 5))
                                impulse_time = 0.06; impulse_mag = (sep_force / max(0.1, new_sep.total_mass)) * impulse_time
                                new_sep.vel += sep_dir * impulse_mag
                                new_rockets_created_in_split_phase.append(new_sep); split_siblings.append(new_sep); next_sim_id += 1
                            except Exception as e: print(f"Error SP split(sep): {e}")
                        break # Stop processing separators for replaced rocket
                    else: # No split from this separator
                        current_parts = parts_without_sep
                        if sep_part in rocket.parts: sep_part.separated = True
                # Update original rocket if no split occurred but parts were removed
                if not split_by_sep and len(current_parts) < len(rocket.parts):
                    rocket.parts = current_parts
                    rocket.engines = [e for e in rocket.engines if e in rocket.parts]; rocket.fuel_tanks = [t for t in rocket.fuel_tanks if t in rocket.parts]
                    rocket.parachutes = [pc for pc in rocket.parachutes if pc in rocket.parts]; rocket.separators = [s for s in rocket.separators if s in rocket.parts and s not in separators_to_process]
                    rocket.engine_firing_status = {e: False for e in rocket.engines}
                    if not rocket.parts: rocket.is_active = False;
                    if rocket.is_active: rocket.needs_structural_update = True # Recalc next frame
                    elif rocket not in rockets_to_remove_this_frame: rockets_to_remove_this_frame.append(rocket)

            # Add grace period for siblings (SP)
            if split_siblings:
                for i_sib, r_sib1 in enumerate(split_siblings):
                    for j_sib in range(i_sib + 1, len(split_siblings)):
                        r_sib2 = split_siblings[j_sib]; grace_pair_key = tuple(sorted((r_sib1.sim_instance_id, r_sib2.sim_instance_id)))
                        collision_grace_period_pairs[grace_pair_key] = COLLISION_GRACE_FRAMES
                split_siblings.clear()

        # --- Update Rocket Lists (SP) ---
        if new_rockets_created_in_split_phase:
            new_ctrl_candidate = None
            for new_rocket in new_rockets_created_in_split_phase:
                 if new_rocket not in all_rockets: all_rockets.append(new_rocket)
                 if new_rocket.is_local_player: new_ctrl_candidate = new_rocket # If a new local player rocket was created
            if new_ctrl_candidate: # If control transferred
                 if controlled_rocket and controlled_rocket not in rockets_to_remove_this_frame:
                      controlled_rocket.is_local_player = False # Mark old one as non-local if it still exists
                 controlled_rocket = new_ctrl_candidate # Assign new controlled rocket
            new_rockets_created_in_split_phase.clear()

        if rockets_to_remove_this_frame:
            was_controlled_removed = controlled_rocket in rockets_to_remove_this_frame
            removed_ids = {r.sim_instance_id for r in rockets_to_remove_this_frame}
            # Clean grace period
            pairs_to_del_grace = [pair for pair in collision_grace_period_pairs if pair[0] in removed_ids or pair[1] in removed_ids]
            for pair in pairs_to_del_grace:
                 if pair in collision_grace_period_pairs: del collision_grace_period_pairs[pair]
            # Filter list
            all_rockets = [r for r in all_rockets if r not in rockets_to_remove_this_frame]
            rockets_to_remove_this_frame.clear()
            if was_controlled_removed: # Find new control if possible
                controlled_rocket = None
                for rkt in all_rockets:
                    root_ref = rkt.original_root_part_ref
                    if root_ref and root_ref in rkt.parts and not root_ref.is_broken:
                        controlled_rocket = rkt; controlled_rocket.is_local_player = True; break

        # --- Camera Update (SP) ---
        if controlled_rocket: camera.update(controlled_rocket.get_world_com())
        elif all_rockets: camera.update(all_rockets[0].get_world_com())

        # --- Drawing (SP) ---
        screen.fill(BLACK); draw_earth_background(screen, camera, stars); draw_terrain(screen, camera)
        total_parts_drawn = 0; total_broken_drawn = 0
        for rocket in all_rockets:
            if rocket.is_active:
                broken_count = rocket.draw(screen, camera, particle_manager, draw_name=False) # Don't draw name in SP
                total_parts_drawn += len(rocket.parts); total_broken_drawn += broken_count
        particle_manager.update(dt); particle_manager.draw(screen, camera)

        # --- Draw UI (SP) ---
        if controlled_rocket: # Draw UI only if controlled rocket exists
            bar_w=20; bar_h=100; bar_x=15; bar_y=SCREEN_HEIGHT-bar_h-40 # Throttle bar
            pygame.draw.rect(screen, COLOR_UI_BAR_BG, (bar_x, bar_y, bar_w, bar_h))
            fill_h = bar_h * controlled_rocket.throttle_level; pygame.draw.rect(screen, COLOR_UI_BAR, (bar_x, bar_y + bar_h - fill_h, bar_w, fill_h))
            pygame.draw.rect(screen, WHITE, (bar_x, bar_y, bar_w, bar_h), 1)
            th_label=ui_font.render("Thr",True,WHITE); screen.blit(th_label, (bar_x, bar_y + bar_h + 5))
            th_value=ui_font.render(f"{controlled_rocket.throttle_level*100:.0f}%",True,WHITE); screen.blit(th_value, (bar_x, bar_y - 18))
            # Telemetry
            alt_agl = max(0, GROUND_Y - controlled_rocket.get_lowest_point_world().y)
            alt_msl = GROUND_Y - controlled_rocket.get_world_com().y
            ctrl_status = "OK" if controlled_rocket.is_local_player else "NO CTRL" # Simplified for SP
            thrust_status = "ON" if controlled_rocket.master_thrust_enabled else "OFF"
            landed_status = "LANDED" if controlled_rocket.landed else "FLYING"
            max_temp_k = controlled_rocket.max_temp_reading
            temp_color = WHITE; hottest_allowable = DEFAULT_MAX_TEMP
            if controlled_rocket.parts: temps = [p.part_data.get('max_temp',DEFAULT_MAX_TEMP) for p in controlled_rocket.parts if p.part_data]; hottest_allowable = max(temps) if temps else DEFAULT_MAX_TEMP
            if max_temp_k > REENTRY_EFFECT_THRESHOLD_TEMP: temp_color = (255,255,0)
            if max_temp_k > hottest_allowable * 0.9: temp_color = (255,100,0)
            if max_temp_k > hottest_allowable: temp_color = RED
            total_fuel = controlled_rocket.get_total_current_fuel()
            status_texts = [f"Alt(AGL): {alt_agl:.1f}m", f"Alt(MSL): {alt_msl:.1f}m", f"Vvel: {controlled_rocket.vel.y:.1f} m/s", f"Hvel: {controlled_rocket.vel.x:.1f} m/s",
                            f"Speed: {controlled_rocket.vel.length():.1f} m/s", f"Angle: {controlled_rocket.angle:.1f} deg", f"AngVel: {controlled_rocket.angular_velocity:.1f} d/s",
                            f"Thr: {controlled_rocket.throttle_level*100:.0f}% [{thrust_status}]", f"Fuel: {total_fuel:.1f} units", f"Mass: {controlled_rocket.total_mass:.1f} kg",
                            f"Control: {ctrl_status}", f"Status: {landed_status}"]#, f"MaxTemp: {max_temp_k:.0f} K"] # Temp removed for space
            text_y_start = 10; control_color = WHITE if controlled_rocket.is_local_player else RED
            for i, text in enumerate(status_texts):
                line_color = temp_color if "MaxTemp" in text else (control_color if "Control" in text else WHITE)
                text_surf = ui_font.render(text, True, line_color); screen.blit(text_surf, (bar_x + bar_w + 10, text_y_start + i * 18))
            # Add Temp below main block
            temp_surf = ui_font.render(f"MaxTemp: {max_temp_k:.0f} K", True, temp_color)
            screen.blit(temp_surf, (bar_x + bar_w + 10, text_y_start + len(status_texts) * 18))

        elif not all_rockets: # All destroyed message
            destroyed_text=ui_font_large.render("ROCKET DESTROYED",True,RED); text_rect=destroyed_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2)); screen.blit(destroyed_text,text_rect)
            respawn_text=ui_font.render("Press 'R' to Respawn",True,WHITE); respawn_rect=respawn_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2+40)); screen.blit(respawn_text,respawn_rect)

        # Debug Info (SP)
        fps = clock.get_fps(); debug_y = 10; debug_x = SCREEN_WIDTH - 120
        fps_text = ui_font.render(f"FPS: {fps:.1f}", True, WHITE); screen.blit(fps_text, (debug_x, debug_y)); debug_y += 18
        obj_text = ui_font.render(f"Rockets: {len(all_rockets)}", True, WHITE); screen.blit(obj_text, (debug_x, debug_y)); debug_y += 18
        parts_text = ui_font.render(f"Parts: {total_parts_drawn}", True, WHITE); screen.blit(parts_text, (debug_x, debug_y)); debug_y += 18
        particle_text = ui_font.render(f"Particles: {len(particle_manager.particles)}", True, WHITE); screen.blit(particle_text, (debug_x, debug_y)); debug_y += 18
        grace_text = ui_font.render(f"Grace Pairs: {len(collision_grace_period_pairs)}", True, WHITE); screen.blit(grace_text, (debug_x, debug_y))

        pygame.display.flip() # Update Display

    print("--- Exiting Simulation (Single Player) ---")


# --- NEW: Multiplayer Simulation Runner ---
def run_multiplayer_simulation(screen, clock, local_blueprint_file, network_mgr, mp_mode):
    print(f"--- Starting Simulation (Multiplayer {mp_mode}) ---")
    local_player_id = network_mgr.player_id
    local_player_name = network_mgr.player_name

    # --- Game State Variables ---
    # Store rockets by player ID for easy lookup
    player_rockets: dict[int, FlyingRocket] = {}
    # Store blueprint JSON strings received from network
    player_blueprints: dict[int, str] = {}
    # Store player names received
    player_names: dict[int, str] = {local_player_id: local_player_name} # Start with local player
    # Track readiness (received blueprint AND launch ready signal)
    player_ready_status: dict[int, bool] = {} # pid -> is_ready
    # Track connection status (for lobby-like waiting)
    connected_players: dict[int, str] = {local_player_id: local_player_name} # Start with self

    # --- Simulation Phase Flags ---
    # Phase 1: Waiting for players, blueprints, ready signals
    # Phase 2: Active simulation loop
    current_phase = "WAITING" # Can be "WAITING", "RUNNING", "ENDED"
    all_players_ready = False # Flag to start simulation

    # Load local blueprint string (sent earlier, but load again for consistency)
    try:
        with open(local_blueprint_file, 'r') as f:
            player_blueprints[local_player_id] = f.read()
        player_ready_status[local_player_id] = True # Mark local player as ready (blueprint sent from builder)
        print(f"Local player {local_player_id} blueprint loaded and marked ready.")
    except Exception as e:
        print(f"FATAL ERROR: Could not load local blueprint '{local_blueprint_file}' for MP: {e}")
        # Send error to network? Exit?
        if network_mgr: network_mgr.stop()
        return # Exit simulation

    # Simulation Setup (Camera, Stars, Particles, UI)
    camera = Camera(SCREEN_WIDTH, SCREEN_HEIGHT)
    camera.update(pygame.math.Vector2(0, GROUND_Y - SCREEN_HEIGHT // 3)) # Initial view
    try:
        star_area_bounds = pygame.Rect(-WORLD_WIDTH*2, SPACE_Y_LIMIT - STAR_FIELD_DEPTH, WORLD_WIDTH*4, abs(SPACE_Y_LIMIT) + GROUND_Y + STAR_FIELD_DEPTH * 1.5)
        stars = create_stars(STAR_COUNT, star_area_bounds)
    except NameError: stars = []
    ui_font = pygame.font.SysFont(None, 20); ui_font_large = pygame.font.SysFont(None, 36)
    particle_manager = ParticleManager()

    # Multiplayer Specific Setup
    network_send_queue = queue.Queue() # Queue for messages to be sent by network thread (used by local rocket update)
    sim_running = True
    current_sim_frame = 0
    last_ping_time = time.time()
    collision_grace_period_pairs: dict[tuple[int, int], int] = {}
    next_sim_id_counter = 0 # Local counter for assigning sim_instance_id to rockets created

    # --- Helper: Function to create rocket from blueprint string ---
    def create_rocket_instance(player_id, bp_json_str, name, sim_id, frame):
        try:
            # Load blueprint from the JSON string
            bp_data = json.loads(bp_json_str)
            temp_blueprint = RocketBlueprint(bp_data.get("name", f"Player_{player_id}_Rocket"))
            temp_blueprint.parts = [PlacedPart.from_dict(part_data) for part_data in bp_data.get("parts", [])]
            if not temp_blueprint.parts:
                print(f"Warning: Blueprint for player {player_id} is empty.")
                return None

            # Find connected components (usually just one expected)
            subassemblies = temp_blueprint.find_connected_subassemblies()
            if not subassemblies or not subassemblies[0]:
                print(f"Error: No connected parts in blueprint for player {player_id}.")
                return None
            assembly_parts = subassemblies[0] # Assume first assembly is the main one

            # Determine spawn position based on player ID order (simple horizontal layout)
            # Sort current player IDs to get a consistent order
            sorted_pids = sorted(connected_players.keys())
            try: player_spawn_index = sorted_pids.index(player_id)
            except ValueError: player_spawn_index = len(sorted_pids) # Append new players

            start_x = player_spawn_index * MP_LAUNCHPAD_SPACING

            # Calculate spawn Y based on lowest point (same as SP)
            temp_bp_for_calc = RocketBlueprint(); temp_bp_for_calc.parts = assembly_parts
            initial_com_local = temp_bp_for_calc.calculate_subassembly_world_com(assembly_parts)
            lowest_offset_y = temp_bp_for_calc.get_lowest_point_offset_y()
            start_y_for_origin = GROUND_Y - lowest_offset_y # Spawn on ground
            target_initial_com_pos = pygame.math.Vector2(start_x, start_y_for_origin + initial_com_local.y)

            # Determine if this is the locally controlled rocket
            is_local = (player_id == local_player_id)

            # Find original root reference within this specific assembly
            root_ref = None
            for part in assembly_parts:
                 if part.part_data and part.part_data.get("type") == "CommandPod": root_ref = part; break
            if not root_ref and assembly_parts: root_ref = assembly_parts[0]

            # Create the FlyingRocket instance
            rocket = FlyingRocket(
                parts_list=list(assembly_parts), initial_world_com_pos=target_initial_com_pos,
                initial_angle=0, initial_vel=pygame.math.Vector2(0,0), sim_instance_id=sim_id,
                is_primary_control=is_local, original_root_ref=root_ref,
                current_frame=frame, player_id=player_id, player_name=name
            )
            print(f"Successfully created rocket instance for player {player_id} (SimID: {sim_id}, Local: {is_local}) at X={start_x}")
            return rocket
        except json.JSONDecodeError: print(f"Error decoding blueprint JSON for player {player_id}")
        except Exception as e: print(f"Error creating rocket instance for player {player_id}: {e}"); import traceback; traceback.print_exc()
        return None

    # --- Main Multiplayer Loop ---
    while sim_running:
        current_time = time.time()
        dt = min(clock.tick(60) / 1000.0, 0.05)
        current_sim_frame += 1

        # Lists for managing rockets during the frame
        newly_created_rockets_this_frame: list[FlyingRocket] = []
        rockets_to_remove_this_frame: list[FlyingRocket] = []
        rocket_splits_this_frame: list[FlyingRocket] = [] # Track original rockets that split

        # --- Network Message Processing ---
        while not network_mgr.message_queue.empty():
            try:
                msg = network_mgr.message_queue.get_nowait()
                msg_type = msg.get("type")
                sender_pid = msg.get("pid") # Player ID who sent/caused the message

                # print(f"MP Sim RX ({mp_mode}): {msg}") # Debug all messages

                if msg_type == network.MSG_TYPE_ERROR:
                    print(f"!!! Network Error: {msg.get('data')} !!!")
                    # Decide how to handle: show message, disconnect, etc.
                    # For now, just print and continue, maybe disconnect later.
                    # sim_running = False # Option: Stop sim on critical error

                elif msg_type == network.MSG_TYPE_PLAYER_JOINED:
                    pid = msg.get("pid")
                    name = msg.get("name", f"Player_{pid}")
                    if pid != local_player_id: # Don't add self again
                        connected_players[pid] = name
                        player_names[pid] = name
                        player_ready_status[pid] = False # Assume not ready until blueprint/ready signal received
                        print(f"Player {pid} ({name}) joined.")
                        # If already running, host needs to send current game state to new player
                        if current_phase == "RUNNING" and mp_mode == "HOST":
                            # Gather state from all *other* rockets
                            game_state = {}
                            for other_pid, rocket in player_rockets.items():
                                if other_pid != pid and rocket.is_active: # Exclude the new player and inactive rockets
                                    game_state[other_pid] = rocket.get_state()
                            state_msg = {"type": network.MSG_TYPE_GAME_STATE, "state": game_state}
                            # Find the socket for the new player to send directly (Server only)
                            target_socket = None
                            for sock, info in network_mgr.clients.items():
                                if info['id'] == pid: target_socket = sock; break
                            if target_socket:
                                 network_mgr.send_message(target_socket, state_msg)
                                 print(f"Sent current game state to joining player {pid}.")
                            else: print(f"Error: Could not find socket for joining player {pid} to send state.")


                elif msg_type == network.MSG_TYPE_PLAYER_LEFT:
                    pid = msg.get("pid")
                    name = player_names.get(pid, f"Player_{pid}")
                    print(f"Player {pid} ({name}) left.")
                    connected_players.pop(pid, None)
                    player_names.pop(pid, None)
                    player_blueprints.pop(pid, None)
                    player_ready_status.pop(pid, None)
                    rocket_to_remove = player_rockets.pop(pid, None)
                    if rocket_to_remove and rocket_to_remove not in rockets_to_remove_this_frame:
                        rockets_to_remove_this_frame.append(rocket_to_remove)
                    # Check if all players are ready again if someone leaves during waiting
                    if current_phase == "WAITING": all_players_ready = False


                elif msg_type == network.MSG_TYPE_BLUEPRINT:
                     pid = msg.get("pid")
                     bp_json = msg.get("json_str")
                     name = msg.get("name", f"Player_{pid}") # Use name from message if provided
                     bp_name = msg.get("bp_name", "Unknown Rocket")
                     print(f"Received blueprint for Player {pid} ({name}) - Rocket: {bp_name}")
                     if pid not in player_blueprints and bp_json:
                         player_blueprints[pid] = bp_json
                         player_names[pid] = name # Update name if sent with blueprint
                         if pid not in connected_players: connected_players[pid] = name # Add if joined silently
                         # If already running, create the rocket immediately
                         if current_phase == "RUNNING":
                              sim_id = next_sim_id_counter; next_sim_id_counter += 1
                              new_rocket = create_rocket_instance(pid, bp_json, name, sim_id, current_sim_frame)
                              if new_rocket:
                                   player_rockets[pid] = new_rocket
                                   # Host needs to broadcast the creation/state of this new rocket? Or rely on state sync?
                         # Else, wait for ready signal / launch phase

                elif msg_type == network.MSG_TYPE_LAUNCH_READY:
                    pid = msg.get("pid")
                    print(f"Player {pid} signalled Launch Ready.")
                    if pid in connected_players:
                        player_ready_status[pid] = True
                    # Check if everyone is ready only during the WAITING phase
                    if current_phase == "WAITING": all_players_ready = False # Recalculate readiness


                elif msg_type == network.MSG_TYPE_ROCKET_UPDATE:
                    pid = msg.get("pid")
                    action = msg.get("action")
                    data = msg.get("data")
                    rocket = player_rockets.get(pid)
                    if rocket and not rocket.is_local_player: # Apply only to remote rockets
                        if action == "state_update":
                            rocket.apply_state(data)
                        else:
                            # Assume other actions (deploy, toggle, etc.) are in 'data'
                            action_data = data # The 'data' field contains the action specifics
                            rocket.apply_network_action(action_data)
                    # elif not rocket: print(f"Warning: Received update for unknown rocket PID {pid}") # Less verbose


                elif msg_type == network.MSG_TYPE_GAME_STATE: # Client receives full state on join
                    if mp_mode == "CLIENT":
                         print("Received initial game state.")
                         state_map = msg.get("state", {})
                         for pid_str, state_data in state_map.items():
                              pid = int(pid_str)
                              if pid != local_player_id: # Don't apply to self
                                   # Check if blueprint exists for this player
                                   bp_json = player_blueprints.get(pid)
                                   if bp_json and pid not in player_rockets:
                                       # Create the rocket first
                                       sim_id = next_sim_id_counter; next_sim_id_counter += 1
                                       name = player_names.get(pid, f"Player_{pid}")
                                       new_rocket = create_rocket_instance(pid, bp_json, name, sim_id, current_sim_frame)
                                       if new_rocket:
                                            player_rockets[pid] = new_rocket
                                            # Now apply the received state
                                            new_rocket.apply_state(state_data)
                                            print(f"Created and applied state for existing player {pid}")
                                   elif pid in player_rockets:
                                        # Rocket already exists, just apply state
                                        player_rockets[pid].apply_state(state_data)
                                        print(f"Applied game state update to existing player {pid}")
                                   # else: print(f"Warning: Received game state for player {pid} but no blueprint.")


                # Handle other message types if needed (e.g., SET_NAME)
                elif msg_type == network.MSG_TYPE_SET_NAME:
                     pid = msg.get("pid")
                     name = msg.get("name")
                     if pid in player_names: player_names[pid] = name
                     if pid in connected_players: connected_players[pid] = name
                     if pid in player_rockets: player_rockets[pid].player_name = name
                     print(f"Updated name for Player {pid} to '{name}'")


            except queue.Empty:
                break # No more messages in queue for now
            except Exception as e:
                print(f"Error processing network message: {e}")
                import traceback
                traceback.print_exc()


        # --- Phase Logic ---
        if current_phase == "WAITING":
            # Check if all connected players have sent blueprint and ready signal
            all_players_ready = True
            if not connected_players: all_players_ready = False # Need at least one player
            for pid in connected_players:
                 # Check if blueprint received AND ready signal received
                 if pid not in player_blueprints or not player_ready_status.get(pid, False):
                     all_players_ready = False
                     break

            if all_players_ready:
                print("All players ready! Creating rockets and starting simulation...")
                current_phase = "RUNNING"
                # Create rocket instances for all players now
                player_rockets.clear() # Clear any previous instances
                next_sim_id_counter = 0 # Reset sim ID counter for this launch
                for pid, bp_json in player_blueprints.items():
                    if pid in connected_players: # Only create for currently connected players
                         sim_id = next_sim_id_counter; next_sim_id_counter += 1
                         name = player_names.get(pid, f"Player_{pid}")
                         rocket = create_rocket_instance(pid, bp_json, name, sim_id, current_sim_frame)
                         if rocket:
                             player_rockets[pid] = rocket
                         else:
                             print(f"Failed to create rocket for player {pid} at launch!")
                             # Handle this error? Kick player? Abort launch?
                             current_phase = "WAITING" # Go back to waiting if creation fails
                             all_players_ready = False
                             break # Stop creating rockets

                # Set initial camera position based on local player's rocket
                local_rocket = player_rockets.get(local_player_id)
                if local_rocket:
                     camera.update(local_rocket.get_world_com())
                elif player_rockets: # Fallback to first rocket if local failed
                     camera.update(list(player_rockets.values())[0].get_world_com())


        elif current_phase == "RUNNING":
            # --- Local Player Input Handling ---
            local_rocket = player_rockets.get(local_player_id)
            if local_rocket and local_rocket.is_active:
                # Check root part status for control
                root_ref = local_rocket.original_root_part_ref
                can_control_locally = (root_ref is not None) and (root_ref in local_rocket.parts) and (not root_ref.is_broken)

                # Event-based input (handled once per frame)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: sim_running = False; break
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE: sim_running = False; break
                        if can_control_locally:
                            # Master Thrust Toggle
                            if event.key == pygame.K_SPACE:
                                local_rocket.master_thrust_enabled = not local_rocket.master_thrust_enabled
                                # Send action immediately
                                action_msg = {"action": "set_master_thrust", "value": local_rocket.master_thrust_enabled}
                                network_send_queue.put({"type": network.MSG_TYPE_ACTION, "pid": local_player_id, "data": action_msg})

                            # Deploy All Parachutes
                            if event.key == pygame.K_p:
                                chutes_activated = []
                                for i, chute in enumerate(local_rocket.parachutes):
                                    if not chute.deployed and not chute.is_broken:
                                        chute.deployed = True # Apply locally first
                                        chutes_activated.append(i) # Send indices of activated chutes
                                if chutes_activated:
                                     # Send one message with all activated chute indices?
                                     # Or send individual messages? Individual is simpler to handle.
                                     for chute_idx_in_list in chutes_activated:
                                         # Find the original part index
                                         original_part_idx = local_rocket.parachutes[chute_idx_in_list].part_index
                                         action_msg = {"action": "deploy", "part_idx": original_part_idx}
                                         network_send_queue.put({"type": network.MSG_TYPE_ACTION, "pid": local_player_id, "data": action_msg})
                                     print(f"Sent deploy actions for {len(chutes_activated)} chutes.")

                    # Part Activation Click
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and can_control_locally:
                        click_world_pos = pygame.math.Vector2(event.pos) + camera.offset
                        # activate_part_at_pos applies locally AND returns action details
                        action_details = local_rocket.activate_part_at_pos(click_world_pos)
                        if action_details:
                            # Send the action details over the network
                            network_send_queue.put({"type": network.MSG_TYPE_ACTION, "pid": local_player_id, "data": action_details})

                if not sim_running: break # Exit outer loop if Esc was pressed

                # Continuous Controls (Throttle)
                if can_control_locally:
                    keys = pygame.key.get_pressed(); throttle_change = 0.0
                    if keys[pygame.K_w] or keys[pygame.K_UP]: throttle_change += THROTTLE_CHANGE_RATE * dt
                    if keys[pygame.K_s] or keys[pygame.K_DOWN]: throttle_change -= THROTTLE_CHANGE_RATE * dt
                    if abs(throttle_change) > 1e-6: # Check if change occurred
                        new_throttle = max(0.0, min(1.0, local_rocket.throttle_level + throttle_change))
                        # Check if throttle value actually changed before sending
                        if abs(new_throttle - local_rocket.throttle_level) > 1e-6:
                             local_rocket.throttle_level = new_throttle
                             # Send action immediately
                             action_msg = {"action": "set_throttle", "value": local_rocket.throttle_level}
                             network_send_queue.put({"type": network.MSG_TYPE_ACTION, "pid": local_player_id, "data": action_msg})


            # --- Update All Active Rockets ---
            # Create a list of rockets to iterate over, as player_rockets might change during update (splits)
            current_rockets_in_sim = list(player_rockets.values())
            for rocket in current_rockets_in_sim:
                if not rocket.is_active: continue
                # Skip update if rocket was just created this frame? No, needed for physics.
                altitude_agl = max(0, GROUND_Y - rocket.get_world_com().y); current_air_density = get_air_density(altitude_agl)
                # Pass the send queue ONLY to the local rocket's update function
                send_q = network_send_queue if rocket.is_local_player else None
                rocket.update(dt, current_air_density, particle_manager, network_send_queue=send_q, current_time=current_time)
                # Check if rocket became inactive or split during its update
                if not rocket.is_active and rocket not in rockets_to_remove_this_frame:
                    rockets_to_remove_this_frame.append(rocket)
                if rocket.needs_structural_update and rocket.is_active: # Check if update flagged a potential split
                    # Only host handles split logic authoritatively? Or client predicts?
                    # Simplest: Host detects split, creates new rockets, broadcasts creation/state.
                    # Client receives destruction/creation messages.
                    # For now, let split logic run locally, host will correct if needed.
                    # We need to track which rocket initiated the split check.
                    if rocket not in rocket_splits_this_frame:
                         rocket_splits_this_frame.append(rocket) # Mark for split processing


            # --- Send Queued Network Messages ---
            while not network_send_queue.empty():
                 try:
                     msg_to_send = network_send_queue.get_nowait()
                     network_mgr.send(msg_to_send) # Client sends to server
                     # Host needs to broadcast updates (or server handles broadcast)
                     if mp_mode == "HOST":
                          # Broadcast ROCKET_UPDATE messages
                          if msg_to_send.get("type") == network.MSG_TYPE_ROCKET_UPDATE:
                               network_mgr.broadcast(msg_to_send, exclude_socket=None) # Broadcast own state
                          # Broadcast ACTION messages as ROCKET_UPDATE
                          elif msg_to_send.get("type") == network.MSG_TYPE_ACTION:
                               update_msg = msg_to_send.copy()
                               update_msg["type"] = network.MSG_TYPE_ROCKET_UPDATE
                               # Action specific data is already in 'data' field
                               network_mgr.broadcast(update_msg, exclude_socket=None) # Broadcast own actions

                 except queue.Empty: break
                 except Exception as e: print(f"Error sending network message: {e}")


            # --- Inter-Rocket Collision (MP) ---
            # Similar logic to SP, but uses player_rockets dict values
            collision_pairs_processed_this_frame = set()
            # Get current list of active rockets
            active_rockets_list = [r for r in player_rockets.values() if r.is_active and r not in rockets_to_remove_this_frame]
            for i in range(len(active_rockets_list)):
                r1 = active_rockets_list[i]
                if not r1.parts: continue # Skip if parts somehow disappeared
                for j in range(i + 1, len(active_rockets_list)):
                    r2 = active_rockets_list[j]
                    if not r2.parts: continue
                    # Grace period check
                    pair_key_grace = tuple(sorted((r1.sim_instance_id, r2.sim_instance_id)))
                    if pair_key_grace in collision_grace_period_pairs: continue
                    # Broad phase
                    dist_sq = (r1.get_world_com() - r2.get_world_com()).length_squared()
                    r1_r = max(r1.local_bounds.width,r1.local_bounds.height)/2.0; r2_r = max(r2.local_bounds.width,r2.local_bounds.height)/2.0
                    if dist_sq > (r1_r + r2_r + 10)**2: continue
                    # Narrow phase
                    coll_found=False; c_p1=None; c_p2=None
                    for p1 in r1.parts:
                        if p1.is_broken: continue; rect1=r1.get_world_part_aabb(p1)
                        for p2 in r2.parts:
                             if p2.is_broken: continue; rect2=r2.get_world_part_aabb(p2)
                             if rect1.colliderect(rect2): coll_found=True; c_p1=p1; c_p2=p2; break
                        if coll_found: break
                    # Response
                    if coll_found:
                        pair_key_proc = tuple(sorted((r1.sim_instance_id, r2.sim_instance_id)))
                        if pair_key_proc in collision_pairs_processed_this_frame: continue
                        collision_pairs_processed_this_frame.add(pair_key_proc)
                        rel_vel = r1.vel - r2.vel; imp_spd = rel_vel.length()
                        # Apply damage locally (Host might need to broadcast damage state changes?)
                        r1.apply_collision_damage(imp_spd, particle_manager, c_p1); r2.apply_collision_damage(imp_spd, particle_manager, c_p2)
                        # Apply push locally
                        coll_norm = r1.get_world_com() - r2.get_world_com()
                        if coll_norm.length_squared() > 1e-6: coll_norm.normalize_ip()
                        else: coll_norm = pygame.math.Vector2(0, -1)
                        push = 2.0; tot_m = r1.total_mass + r2.total_mass
                        if tot_m > 0.01: r1.pos += coll_norm*push*(r2.total_mass/tot_m); r2.pos -= coll_norm*push*(r1.total_mass/tot_m)
                        # Check if damage caused structural changes
                        if r1.needs_structural_update and r1 not in rocket_splits_this_frame: rocket_splits_this_frame.append(r1)
                        if r2.needs_structural_update and r2 not in rocket_splits_this_frame: rocket_splits_this_frame.append(r2)

            # --- Process Connectivity Checks and Separations (MP) ---
            # TODO MP: This needs careful handling. Host should be authoritative.
            # Simplification: Let splits happen locally. Host broadcasts state updates which will eventually correct clients.
            # More Robust: Only host runs split logic. Sends messages like "ROCKET_DESTROYED", "NEW_ROCKET", "STATE_UPDATE" for fragments.
            # Implementing the simple local split prediction for now.
            new_rockets_created_in_split_phase: list[FlyingRocket] = []
            processed_split_rockets_this_frame = set() # Track sim_instance_ids that already split

            # Iterate through rockets flagged for potential split
            for rocket in rocket_splits_this_frame:
                 # Skip if rocket was already removed or already processed for split this frame
                 if rocket in rockets_to_remove_this_frame or not rocket.is_active or rocket.sim_instance_id in processed_split_rockets_this_frame:
                     continue

                 processed_split = False; split_siblings: list[FlyingRocket] = []
                 original_rocket_pid = rocket.player_id # Remember who owned the original rocket

                 # --- Check for splits based on current state (destruction or separation) ---
                 # We check connectivity based on the *current* parts list (already filtered by handle_destroyed_parts if needed)
                 # And consider pending separations
                 parts_to_check = list(rocket.parts)
                 separators_activated = list(rocket.pending_separation) # Use pending list
                 rocket.pending_separation.clear() # Clear pending list after copying

                 # Simulate separator removals first if any activated
                 split_due_to_separator = False
                 if separators_activated:
                      current_check_parts = list(parts_to_check)
                      for sep_part in separators_activated:
                           if sep_part not in current_check_parts: continue # Already removed?
                           parts_without_this_separator = [p for p in current_check_parts if p != sep_part]
                           temp_bp_sep = RocketBlueprint(); temp_bp_sep.parts = parts_without_this_separator
                           subassemblies_sep = temp_bp_sep.find_connected_subassemblies()
                           if len(subassemblies_sep) > 1:
                                parts_to_check = parts_without_this_separator # Update parts list for subsequent checks
                                split_due_to_separator = True
                                # Mark separator as used locally (network state handles actual removal later)
                                if sep_part in rocket.parts: sep_part.separated = True
                                break # Handle one separator split at a time per frame? Or just find if *any* split occurs? Assume any split is handled.
                           else:
                                # No split from this one, conceptually remove it for next check
                                current_check_parts = parts_without_this_separator
                                if sep_part in rocket.parts: sep_part.separated = True # Mark as used
                      # If a split was found via separator, use the remaining parts for final check
                      if split_due_to_separator:
                           parts_to_check = current_check_parts # Use parts remaining after the split-causing separator

                 # Final connectivity check on potentially modified parts list
                 temp_bp = RocketBlueprint(); temp_bp.parts = parts_to_check
                 subassemblies = temp_bp.find_connected_subassemblies()

                 # --- Handle Split Creation ---
                 if len(subassemblies) > 1:
                      processed_split = True; print(f"[MP:{rocket.sim_instance_id} P:{rocket.player_id}] SPLIT detected into {len(subassemblies)} pieces!")
                      processed_split_rockets_this_frame.add(rocket.sim_instance_id) # Mark as processed

                      # Mark original for removal
                      if rocket not in rockets_to_remove_this_frame: rockets_to_remove_this_frame.append(rocket)

                      # Preserve original state if control transfers
                      original_throttle = rocket.throttle_level; original_master_thrust = rocket.master_thrust_enabled
                      was_local = rocket.is_local_player

                      # Create new fragments
                      for assembly_parts in subassemblies:
                           if not assembly_parts: continue
                           try:
                               sub_com_world = rocket.calculate_subassembly_world_com(assembly_parts)
                               contains_root = rocket.original_root_part_ref and (rocket.original_root_part_ref in assembly_parts)
                               # Control transfers ONLY if original was local AND this piece has the root
                               is_primary = was_local and contains_root
                               # Assign new sim ID
                               sim_id = next_sim_id_counter; next_sim_id_counter += 1
                               # Create fragment
                               # Inherit player ID and name from the original rocket
                               new_frag = FlyingRocket(list(assembly_parts), sub_com_world, rocket.angle, rocket.vel, sim_id, is_primary, rocket.original_root_part_ref, current_sim_frame, original_rocket_pid, rocket.player_name)
                               new_frag.angular_velocity = rocket.angular_velocity
                               if new_frag.is_local_player: new_frag.throttle_level = original_throttle; new_frag.master_thrust_enabled = original_master_thrust
                               # Apply separation impulse if split by separator? Complex to determine which piece gets which impulse locally.
                               # Rely on host correction via state updates for now.

                               new_rockets_created_in_split_phase.append(new_frag)
                               split_siblings.append(new_frag)
                               print(f"  > Created fragment SimID:{sim_id} for Player:{original_rocket_pid} (Local:{is_primary})")

                               # --- MP TODO: Host should broadcast creation/state of new fragments ---
                               # if mp_mode == "HOST":
                               #    create_msg = {"type": "NEW_ROCKET", ...} network_mgr.broadcast(create_msg)
                               #    state_msg = {"type": "STATE_UPDATE", ...} network_mgr.broadcast(state_msg)

                           except Exception as e: print(f"Error MP split creation: {e}")

                 # --- Add Grace Period for New Siblings (MP) ---
                 if split_siblings:
                     for i_sib, r_sib1 in enumerate(split_siblings):
                         for j_sib in range(i_sib + 1, len(split_siblings)):
                             r_sib2 = split_siblings[j_sib]
                             grace_pair_key = tuple(sorted((r_sib1.sim_instance_id, r_sib2.sim_instance_id)))
                             collision_grace_period_pairs[grace_pair_key] = COLLISION_GRACE_FRAMES
                     split_siblings.clear()


            # --- Update Rocket Lists (MP) ---
            if new_rockets_created_in_split_phase:
                new_local_rocket = None
                for new_rocket in new_rockets_created_in_split_phase:
                     # Add to main dictionary using player ID (fragments keep original PID)
                     # PROBLEM: Multiple fragments per player ID. Need unique key. Use SimID?
                     # Let's keep player_rockets keyed by PID for primary rocket, store debris separately?
                     # Simpler: Keep player_rockets[pid] as the *controlled* rocket for that player. Store others elsewhere?
                     # Alternative: Modify player_rockets to store a *list* per PID? player_rockets[pid] = [rocket1, rocket2]
                     # Let's try modifying player_rockets to store lists.

                     pid = new_rocket.player_id
                     if pid not in player_rockets: # Should not happen if original existed
                          player_rockets[pid] = [] # Initialize list if needed

                     # Add the new fragment to the player's list
                     # Need to rethink how `player_rockets` is used if it becomes a list.
                     # --- REVERTING: player_rockets holds ONE rocket per PID (the main/controlled one) ---
                     # Debris needs separate handling or rely on host updates.
                     # For now, just add to a temporary 'all_sim_rockets' list for drawing/physics?

                     # --- TEMPORARY: Add new fragments directly to player_rockets, overwriting ---
                     # This is WRONG for multiple fragments, but allows drawing temporarily.
                     # Needs proper debris management system.
                     player_rockets[pid] = new_rocket # !!! This overwrites previous fragments !!!

                     if new_rocket.is_local_player: # Check if player regained control
                          new_local_rocket = new_rocket

                # Update local controlled rocket reference if needed
                if new_local_rocket:
                     # If the old controlled rocket still exists but lost control
                     old_local = player_rockets.get(local_player_id)
                     if old_local and old_local != new_local_rocket:
                          old_local.is_local_player = False # Mark old as non-local
                     # Set the new one as local (already done in FlyingRocket init)
                     # Update camera target? Done below.
                     print(f"Control transferred to new fragment SimID:{new_local_rocket.sim_instance_id}")

                new_rockets_created_in_split_phase.clear()


            if rockets_to_remove_this_frame:
                 # Remove from player_rockets dict
                 removed_pids = set()
                 for r in rockets_to_remove_this_frame:
                      if r.player_id in player_rockets and player_rockets[r.player_id] == r:
                           del player_rockets[r.player_id]
                           removed_pids.add(r.player_id)
                      # Clean up grace period involving removed rockets
                      removed_sim_id = r.sim_instance_id
                      pairs_to_del = [p for p in collision_grace_period_pairs if removed_sim_id in p]
                      for pair in pairs_to_del:
                           if pair in collision_grace_period_pairs: del collision_grace_period_pairs[pair]

                 # Handle loss of local control
                 if local_player_id in removed_pids:
                      print("Local player's rocket was removed.")
                      # Find if any *new* fragments for local player exist (This is flawed with current overwrite)
                      # Need a better way to track local player control transfer.

                 rockets_to_remove_this_frame.clear()

            # --- Camera Update (MP) ---
            local_rocket = player_rockets.get(local_player_id)
            if local_rocket and local_rocket.is_active:
                 camera.update(local_rocket.get_world_com())
            elif player_rockets: # Fallback to follow first active rocket if local is gone
                 first_active = next((r for r in player_rockets.values() if r.is_active), None)
                 if first_active: camera.update(first_active.get_world_com())
            # Else camera stays put

            # --- Periodic Ping (Client) ---
            if mp_mode == "CLIENT" and current_time - last_ping_time > 5.0:
                 network_mgr.send({"type": network.MSG_TYPE_PING})
                 last_ping_time = current_time

        # --- Drawing ---
        screen.fill(BLACK)
        # Draw background elements based on camera position (local player's view)
        draw_earth_background(screen, camera, stars); draw_terrain(screen, camera)

        # Draw all active rockets (from player_rockets dictionary)
        total_parts_drawn = 0; total_broken_drawn = 0
        # Sort rockets by Y pos for pseudo-depth? Or just draw all.
        all_rockets_to_draw = list(player_rockets.values()) # Get current rockets
        for rocket in all_rockets_to_draw:
            if rocket.is_active:
                # Draw name tag for all rockets in MP
                broken_count = rocket.draw(screen, camera, particle_manager, draw_name=True)
                total_parts_drawn += len(rocket.parts); total_broken_drawn += broken_count

        # Draw particles
        particle_manager.update(dt); particle_manager.draw(screen, camera)

        # --- Draw UI (MP) ---
        # Status text during WAITING phase
        if current_phase == "WAITING":
             status_lines = ["Waiting for Players..."]
             for pid, name in connected_players.items():
                  bp_status = "OK" if pid in player_blueprints else "Waiting"
                  rdy_status = "Ready" if player_ready_status.get(pid, False) else "Not Ready"
                  local_tag = "(You)" if pid == local_player_id else ""
                  status_lines.append(f" - P{pid} {name} {local_tag}: Blueprint[{bp_status}] Status[{rdy_status}]")
             y_pos = 50
             for line in status_lines:
                  surf = ui_font_large.render(line, True, WHITE)
                  rect = surf.get_rect(center=(SCREEN_WIDTH // 2, y_pos))
                  screen.blit(surf, rect)
                  y_pos += 40

        # Draw telemetry for local player during RUNNING phase
        elif current_phase == "RUNNING":
            local_rocket = player_rockets.get(local_player_id)
            if local_rocket: # Draw local player UI if their rocket exists
                # Throttle Bar
                bar_w=20; bar_h=100; bar_x=15; bar_y=SCREEN_HEIGHT-bar_h-40
                pygame.draw.rect(screen, COLOR_UI_BAR_BG, (bar_x, bar_y, bar_w, bar_h))
                fill_h = bar_h * local_rocket.throttle_level; pygame.draw.rect(screen, COLOR_UI_BAR, (bar_x, bar_y + bar_h - fill_h, bar_w, fill_h))
                pygame.draw.rect(screen, WHITE, (bar_x, bar_y, bar_w, bar_h), 1)
                th_label=ui_font.render("Thr",True,WHITE); screen.blit(th_label, (bar_x, bar_y + bar_h + 5))
                th_value=ui_font.render(f"{local_rocket.throttle_level*100:.0f}%",True,WHITE); screen.blit(th_value, (bar_x, bar_y - 18))
                # Telemetry
                alt_agl = max(0, GROUND_Y - local_rocket.get_lowest_point_world().y); alt_msl = GROUND_Y - local_rocket.get_world_com().y
                root_ref = local_rocket.original_root_part_ref # Check local control status
                can_control = local_rocket.is_local_player and root_ref and root_ref in local_rocket.parts and not root_ref.is_broken
                ctrl_status = "OK" if can_control else "NO CTRL"
                thrust_status = "ON" if local_rocket.master_thrust_enabled else "OFF"; landed_status = "LANDED" if local_rocket.landed else "FLYING"
                max_temp_k = local_rocket.max_temp_reading; temp_color = WHITE; hottest_allowable = DEFAULT_MAX_TEMP
                if local_rocket.parts: temps = [p.part_data.get('max_temp',DEFAULT_MAX_TEMP) for p in local_rocket.parts if p.part_data]; hottest_allowable = max(temps) if temps else DEFAULT_MAX_TEMP
                if max_temp_k > REENTRY_EFFECT_THRESHOLD_TEMP: temp_color = (255,255,0)
                if max_temp_k > hottest_allowable * 0.9: temp_color = (255,100,0)
                if max_temp_k > hottest_allowable: temp_color = RED
                total_fuel = local_rocket.get_total_current_fuel()
                status_texts = [f"Alt(AGL): {alt_agl:.1f}m", f"Alt(MSL): {alt_msl:.1f}m", f"Vvel: {local_rocket.vel.y:.1f} m/s", f"Hvel: {local_rocket.vel.x:.1f} m/s",
                                f"Speed: {local_rocket.vel.length():.1f} m/s", f"Angle: {local_rocket.angle:.1f} deg", f"AngVel: {local_rocket.angular_velocity:.1f} d/s",
                                f"Thr: {local_rocket.throttle_level*100:.0f}% [{thrust_status}]", f"Fuel: {total_fuel:.1f} units", f"Mass: {local_rocket.total_mass:.1f} kg",
                                f"Control: {ctrl_status}", f"Status: {landed_status}"] #, f"MaxTemp: {max_temp_k:.0f} K"]
                text_y_start = 10; control_color = WHITE if can_control else RED
                for i, text in enumerate(status_texts):
                    line_color = temp_color if "MaxTemp" in text else (control_color if "Control" in text else WHITE)
                    text_surf = ui_font.render(text, True, line_color); screen.blit(text_surf, (bar_x + bar_w + 10, text_y_start + i * 18))
                temp_surf = ui_font.render(f"MaxTemp: {max_temp_k:.0f} K", True, temp_color); screen.blit(temp_surf, (bar_x + bar_w + 10, text_y_start + len(status_texts) * 18))

            else: # Local rocket destroyed or not found
                 destroyed_text=ui_font_large.render("LOCAL ROCKET LOST",True,RED); text_rect=destroyed_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2)); screen.blit(destroyed_text,text_rect)


        # Debug Info (MP)
        fps = clock.get_fps(); debug_y = 10; debug_x = SCREEN_WIDTH - 140 # Move left slightly
        fps_text = ui_font.render(f"FPS: {fps:.1f}", True, WHITE); screen.blit(fps_text, (debug_x, debug_y)); debug_y += 18
        mode_text = ui_font.render(f"Mode: {mp_mode}", True, WHITE); screen.blit(mode_text, (debug_x, debug_y)); debug_y += 18
        pid_text = ui_font.render(f"PID: {local_player_id}", True, WHITE); screen.blit(pid_text, (debug_x, debug_y)); debug_y += 18
        rocket_count = len(player_rockets) # Count entries in the dict
        obj_text = ui_font.render(f"Tracked: {rocket_count}", True, WHITE); screen.blit(obj_text, (debug_x, debug_y)); debug_y += 18
        parts_text = ui_font.render(f"Parts: {total_parts_drawn}", True, WHITE); screen.blit(parts_text, (debug_x, debug_y)); debug_y += 18
        particle_text = ui_font.render(f"Particles: {len(particle_manager.particles)}", True, WHITE); screen.blit(particle_text, (debug_x, debug_y)); debug_y += 18
        grace_text = ui_font.render(f"Grace Pairs: {len(collision_grace_period_pairs)}", True, WHITE); screen.blit(grace_text, (debug_x, debug_y)); debug_y += 18
        phase_text = ui_font.render(f"Phase: {current_phase}", True, WHITE); screen.blit(phase_text, (debug_x, debug_y)); debug_y += 18


        pygame.display.flip() # Update Display

    print(f"--- Exiting Simulation (Multiplayer {mp_mode}) ---")
    # Network cleanup happens in main.py after this function returns


# --- Direct Run Logic (for testing SP, unchanged) ---
if __name__ == '__main__':
     # ... (SP direct run code remains unchanged) ...
     pygame.init(); screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
     pygame.display.set_caption("PySpaceFlight - Direct Sim Test (SP)"); clock = pygame.time.Clock()
     assets_dir = "assets"; test_blueprint = os.path.join(assets_dir, "grid_rocket_separator_test.json")
     if not os.path.exists(test_blueprint):
         print(f"Test blueprint '{test_blueprint}' not found, creating default...")
         os.makedirs(assets_dir, exist_ok=True)
         bp = RocketBlueprint("Default Separator Test"); pod_pos = pygame.math.Vector2(0,0); pod_data = get_part_data("pod_mk1"); bp.add_part("pod_mk1", pod_pos)
         para_data = get_part_data("parachute_mk1"); para_rel_pos = pod_data['logical_points']['top'] - para_data['logical_points']['bottom']; bp.add_part("parachute_mk1", pod_pos + para_rel_pos)
         sep_data = get_part_data("separator_tr_s1"); sep_rel_pos = pod_data['logical_points']['bottom'] - sep_data['logical_points']['top']; bp.add_part("separator_tr_s1", pod_pos + sep_rel_pos)
         tank_data = get_part_data("tank_small"); tank_rel_pos = sep_data['logical_points']['bottom'] - tank_data['logical_points']['top']; bp.add_part("tank_small", pod_pos + sep_rel_pos + tank_rel_pos)
         engine_data = get_part_data("engine_basic"); engine_rel_pos = tank_data['logical_points']['bottom'] - engine_data['logical_points']['top']; bp.add_part("engine_basic", pod_pos + sep_rel_pos + tank_rel_pos + engine_rel_pos)
         bp.save_to_json(test_blueprint); print(f"Saved default blueprint to {test_blueprint}")
     run_simulation(screen, clock, test_blueprint) # Run the SP version
     pygame.quit(); sys.exit()