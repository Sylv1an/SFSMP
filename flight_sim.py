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
        self.separators_to_fire_this_frame: list[PlacedPart] = []
        self.landed = False
        self.thrusting = False
        self.is_active = True
        self.pending_separation: list[PlacedPart] = []
        self.needs_structural_update = False
        self.was_landed_last_frame = False
        self.max_temp_reading = AMBIENT_TEMPERATURE
        self.engine_firing_status: dict[PlacedPart, bool] = {e: False for e in self.engines}
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
                if clicked_part not in self.separators_to_fire_this_frame:
                    self.separators_to_fire_this_frame.append(clicked_part)
                    action_taken = True
                    clicked_part.separated = True
                    self.needs_structural_update = True
                    action_details = {"action": "separate", "part_idx": part_identifier}
                    print(f"Local Activate: Queued Separator idx {part_identifier}")

        # Toggleable parts (Engines)
        if not action_taken and part_type == "Engine":
            clicked_part.engine_enabled = not clicked_part.engine_enabled
            action_taken = True
            action_details = {"action": "toggle_engine", "part_idx": part_identifier, "enabled": clicked_part.engine_enabled}
            print(f"Local Activate: Toggle Engine idx {part_identifier} {'ON' if clicked_part.engine_enabled else 'OFF'}")

        # Return details for network transmission
        return action_details

    def apply_network_action(self, action_data):
        """Applies an action (deploy, toggle, etc.) received over the network."""
        action_type = action_data.get("action")
        part_idx = action_data.get("part_idx")

        if part_idx is None or part_idx < 0 or part_idx >= len(self.parts):
            print(f"Warning: Received action for invalid part index {part_idx} on rocket {self.sim_instance_id}")
            return

        # Find the part by index, careful not to crash if index is somehow invalid after check
        try:
             part = self.parts[part_idx]
        except IndexError:
             print(f"Error: Part index {part_idx} out of bounds for rocket {self.sim_instance_id} after check.")
             return

        if part.is_broken:
             # print(f"Ignoring action '{action_type}' on broken part idx {part_idx}")
             return # Ignore actions on broken parts

        print(f"Applying network action '{action_type}' to part idx {part_idx} on rocket {self.sim_instance_id}")

        if action_type == "deploy" and part.part_data.get("type") == "Parachute":
            part.deployed = True
        elif action_type == "separate" and part.part_data.get("type") == "Separator":
             # --- Apply separation action correctly ---
             if not part.separated: # Check if not already separated/fired
                 # Add to the list to be processed this frame
                 if part not in self.separators_to_fire_this_frame:
                      self.separators_to_fire_this_frame.append(part)
                 # Mark as separated state visually (actual structural change happens in main loop)
                 part.separated = True
                 # Flag that structure needs recalculation/split check
                 self.needs_structural_update = True
                 print(f"Network: Queued Separator idx {part_idx}")
             # --- End Fix ---
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
            if part.is_broken:
                num_broken_visually += 1
                if is_separator and part.separated:
                    indicator_color = COLOR_ACTIVATABLE_USED
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
        rockets_requiring_split_check: list[FlyingRocket] = []

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
            if rocket.needs_structural_update and rocket.is_active:
                if rocket not in rockets_requiring_split_check:
                    rockets_requiring_split_check.append(rocket)
                rocket.needs_structural_update = False  # Reset flag for this frame check

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

        # --- MODIFIED: Process Connectivity Checks and Separations (SP) ---
        split_siblings_this_frame: list[FlyingRocket] = []  # Track siblings created in this phase for grace period
        # Iterate through rockets needing check (destruction or separation)
        for rocket in rockets_requiring_split_check:
                    if rocket in rockets_to_remove_this_frame or not rocket.is_active: continue

                    processed_split_for_this_rocket = False
                    parts_before_check = list(rocket.parts)  # Snapshot current parts
                    separators_fired = list(rocket.separators_to_fire_this_frame)
                    rocket.separators_to_fire_this_frame.clear()  # Clear the list

                    # Determine parts list after removing fired separators
                    parts_after_sep_removal = [p for p in parts_before_check if p not in separators_fired]

                    # Check if parts list actually changed (due to destruction OR separator firing)
                    if len(parts_after_sep_removal) < len(parts_before_check):
                        # Create a temporary blueprint to check connectivity
                        temp_bp = RocketBlueprint();
                        temp_bp.parts = parts_after_sep_removal
                        subassemblies = temp_bp.find_connected_subassemblies()

                        # --- SPLIT OCCURRED ---
                        if len(subassemblies) > 1:
                            processed_split_for_this_rocket = True
                            split_cause = "Separator" if separators_fired else "Destruction"
                            print(
                                f"[SP:{rocket.sim_instance_id}] SPLIT ({split_cause}) into {len(subassemblies)} pieces!")

                            # Mark original rocket for removal
                            if rocket not in rockets_to_remove_this_frame: rockets_to_remove_this_frame.append(rocket)

                            # Preserve original state if control might transfer
                            original_throttle = rocket.throttle_level;
                            original_master_thrust = rocket.master_thrust_enabled
                            was_local = rocket.is_local_player

                            current_split_siblings: list[FlyingRocket] = []  # Siblings from *this specific* split event

                            # Create new FlyingRocket instances for each fragment
                            for assembly_parts in subassemblies:
                                if not assembly_parts: continue
                                try:
                                    # Calculate CoM of the fragment based on its parts relative to original rocket origin
                                    sub_com_world = rocket.calculate_subassembly_world_com(assembly_parts)
                                    contains_root = rocket.original_root_part_ref and (
                                                rocket.original_root_part_ref in assembly_parts)
                                    is_primary = was_local and contains_root  # Transfer control only if local player had it AND this piece has root

                                    # Create the new rocket fragment
                                    new_frag = FlyingRocket(
                                        parts_list=list(assembly_parts), initial_world_com_pos=sub_com_world,
                                        initial_angle=rocket.angle, initial_vel=pygame.math.Vector2(rocket.vel),
                                        # Copy velocity
                                        sim_instance_id=next_sim_id, is_primary_control=is_primary,
                                        original_root_ref=rocket.original_root_part_ref,  # Keep original root ref
                                        current_frame=current_sim_frame, player_id=0, player_name="LocalPlayer"
                                    )
                                    new_frag.angular_velocity = rocket.angular_velocity  # Copy angular velocity
                                    if new_frag.is_local_player:  # Apply preserved controls if it's the new primary
                                        new_frag.throttle_level = original_throttle;
                                        new_frag.master_thrust_enabled = original_master_thrust

                                    # Apply separation impulse if split by separator
                                    if separators_fired:
                                        # Find the separator most likely responsible for this fragment's separation
                                        # (Simplification: average position of fired separators)
                                        if len(separators_fired) > 0:
                                            sep_world_pos_avg = pygame.math.Vector2(0, 0)
                                            for sep in separators_fired: sep_world_pos_avg += rocket.get_world_part_center(
                                                sep)
                                            sep_world_pos_avg /= len(separators_fired)
                                            sep_force = separators_fired[0].part_data.get("separation_force",
                                                                                          1000)  # Use first separator's force

                                            sep_vec = new_frag.get_world_com() - sep_world_pos_avg
                                            if sep_vec.length_squared() > 1e-6:
                                                sep_dir = sep_vec.normalize()
                                            else:
                                                sep_dir = pygame.math.Vector2(random.uniform(-1, 1), random.uniform(-1,
                                                                                                                    1)).normalize()  # Random push if coincident

                                            impulse_time = 0.05  # Shorter impulse duration
                                            impulse_mag = (sep_force / max(0.1, new_frag.total_mass)) * impulse_time
                                            new_frag.vel += sep_dir * impulse_mag
                                            # Add small random angular impulse too
                                            new_frag.angular_velocity += random.uniform(-15, 15)

                                    newly_created_rockets_this_frame.append(new_frag)
                                    current_split_siblings.append(new_frag)  # Add to siblings for this event
                                    next_sim_id += 1
                                except Exception as e:
                                    print(f"Error SP creating split fragment: {e}")

                            # Add siblings from this event to the frame's list
                            split_siblings_this_frame.extend(current_split_siblings)

                        # --- NO SPLIT, but parts were removed (e.g., end cap separator) ---
                        elif not processed_split_for_this_rocket:  # Check flag again
                            if separators_fired:  # Only update if separators were involved
                                print(f"[SP:{rocket.sim_instance_id}] Separator fired but no structural split.")
                                # Update the original rocket's part list
                                rocket.parts = parts_after_sep_removal
                                # Rebuild internal component lists and physics properties
                                rocket.engines = [e for e in rocket.engines if e in rocket.parts]
                                rocket.fuel_tanks = [t for t in rocket.fuel_tanks if t in rocket.parts]
                                rocket.parachutes = [pc for pc in rocket.parachutes if pc in rocket.parts]
                                rocket.separators = [s for s in rocket.separators if
                                                     s in rocket.parts]  # Remove fired ones
                                rocket.engine_firing_status = {e: False for e in rocket.engines}
                                rocket._build_fuel_source_map()  # Rebuild fuel map
                                rocket.calculate_physics_properties()  # Recalculate mass, CoM, MoI
                                rocket.calculate_bounds()
                                # Mark the fired separators as 'separated' visually (though removed from list)
                                for sep in separators_fired: sep.separated = True
                                if not rocket.parts:  # Check if rocket is now empty
                                    rocket.is_active = False
                                    if rocket not in rockets_to_remove_this_frame: rockets_to_remove_this_frame.append(
                                        rocket)

        # --- Add Collision Grace Period for New Siblings ---
        if split_siblings_this_frame:
            for i_sib, r_sib1 in enumerate(split_siblings_this_frame):
                for j_sib in range(i_sib + 1, len(split_siblings_this_frame)):
                    r_sib2 = split_siblings_this_frame[j_sib]
                    grace_pair_key = tuple(sorted((r_sib1.sim_instance_id, r_sib2.sim_instance_id)))
                    collision_grace_period_pairs[grace_pair_key] = COLLISION_GRACE_FRAMES
            split_siblings_this_frame.clear()  # Clear after processing

            # --- Update Rocket Lists (SP) ---
        if newly_created_rockets_this_frame:
            # ... (logic for adding new rockets and handling control transfer remains the same) ...
            new_ctrl_candidate = None
            for new_rocket in newly_created_rockets_this_frame:
                if new_rocket not in all_rockets: all_rockets.append(new_rocket)
                if new_rocket.is_local_player: new_ctrl_candidate = new_rocket  # If a new local player rocket was created
            if new_ctrl_candidate:  # If control transferred
                if controlled_rocket and controlled_rocket not in rockets_to_remove_this_frame:
                    controlled_rocket.is_local_player = False  # Mark old one as non-local if it still exists
                controlled_rocket = new_ctrl_candidate  # Assign new controlled rocket
            newly_created_rockets_this_frame.clear()

        if rockets_to_remove_this_frame:
            # ... (logic for removing rockets and cleaning grace period remains the same) ...
            was_controlled_removed = controlled_rocket in rockets_to_remove_this_frame
            removed_ids = {r.sim_instance_id for r in rockets_to_remove_this_frame}
            # Clean grace period
            pairs_to_del_grace = [pair for pair in collision_grace_period_pairs if
                                  pair[0] in removed_ids or pair[1] in removed_ids]
            for pair in pairs_to_del_grace:
                if pair in collision_grace_period_pairs: del collision_grace_period_pairs[pair]
            # Filter list
            all_rockets = [r for r in all_rockets if r not in rockets_to_remove_this_frame]
            rockets_to_remove_this_frame.clear()
            if was_controlled_removed:  # Find new control if possible
                controlled_rocket = None
                # --- FIXED: Iterate through potentially updated all_rockets list ---
                for rkt in all_rockets:
                    # Check if it still has its original root part and is active
                    root_ref = rkt.original_root_part_ref
                    if rkt.is_active and root_ref and root_ref in rkt.parts and not root_ref.is_broken:
                        controlled_rocket = rkt
                        controlled_rocket.is_local_player = True
                        print(f"Control transferred to SimID {rkt.sim_instance_id} after original was removed.")
                        break
                if controlled_rocket is None:
                    print("Lost control: Original rocket removed and no suitable fragment found.")

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
    player_rockets: dict[int, FlyingRocket] = {} # Stores primary/controlled rocket per player
    player_blueprints: dict[int, str] = {}
    player_names: dict[int, str] = {local_player_id: local_player_name}
    player_ready_status: dict[int, bool] = {}
    connected_players: dict[int, str] = {local_player_id: local_player_name}
    all_sim_rockets: list[FlyingRocket] = [] # List of ALL active rockets/fragments

    # --- Simulation Phase Flags ---
    current_phase = "WAITING"
    all_players_ready = False

    # Load local blueprint
    try:
        with open(local_blueprint_file, 'r') as f:
            player_blueprints[local_player_id] = f.read()
        player_ready_status[local_player_id] = True
        print(f"Local player {local_player_id} blueprint loaded and marked ready.")
    except Exception as e:
        print(f"FATAL ERROR: Could not load local blueprint '{local_blueprint_file}' for MP: {e}")
        if network_mgr: network_mgr.stop()
        return

    # Simulation Setup
    camera = Camera(SCREEN_WIDTH, SCREEN_HEIGHT)
    camera.update(pygame.math.Vector2(0, GROUND_Y - SCREEN_HEIGHT // 3))
    try:
        star_area_bounds = pygame.Rect(-WORLD_WIDTH*2, SPACE_Y_LIMIT - STAR_FIELD_DEPTH, WORLD_WIDTH*4, abs(SPACE_Y_LIMIT) + GROUND_Y + STAR_FIELD_DEPTH * 1.5)
        stars = create_stars(STAR_COUNT, star_area_bounds)
    except NameError: stars = []
    ui_font = pygame.font.SysFont(None, 20); ui_font_large = pygame.font.SysFont(None, 36)
    particle_manager = ParticleManager()

    # Multiplayer Specific Setup
    network_send_queue = queue.Queue()
    sim_running = True
    current_sim_frame = 0
    last_ping_time = time.time()
    collision_grace_period_pairs: dict[tuple[int, int], int] = {}
    next_sim_id_counter = 0

    # --- Helper: Function to create rocket from blueprint string ---
    def create_rocket_instance(player_id, bp_json_str, name, sim_id, frame):
        # (This function remains the same as provided previously)
        try:
            bp_data = json.loads(bp_json_str)
            temp_blueprint = RocketBlueprint(bp_data.get("name", f"Player_{player_id}_Rocket"))
            temp_blueprint.parts = [PlacedPart.from_dict(part_data) for part_data in bp_data.get("parts", [])]
            if not temp_blueprint.parts: return None
            subassemblies = temp_blueprint.find_connected_subassemblies()
            if not subassemblies or not subassemblies[0]: return None
            assembly_parts = subassemblies[0]
            sorted_pids = sorted(connected_players.keys())
            try: player_spawn_index = sorted_pids.index(player_id)
            except ValueError: player_spawn_index = len(sorted_pids)
            start_x = player_spawn_index * MP_LAUNCHPAD_SPACING
            temp_bp_for_calc = RocketBlueprint(); temp_bp_for_calc.parts = assembly_parts
            initial_com_local = temp_bp_for_calc.calculate_subassembly_world_com(assembly_parts)
            lowest_offset_y = temp_bp_for_calc.get_lowest_point_offset_y()
            start_y_for_origin = GROUND_Y - lowest_offset_y
            target_initial_com_pos = pygame.math.Vector2(start_x, start_y_for_origin + initial_com_local.y)
            is_local = (player_id == local_player_id)
            root_ref = None
            for part in assembly_parts:
                 if part.part_data and part.part_data.get("type") == "CommandPod": root_ref = part; break
            if not root_ref and assembly_parts: root_ref = assembly_parts[0]
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

        newly_created_rockets_this_frame: list[FlyingRocket] = []
        rockets_to_remove_this_frame: list[FlyingRocket] = []
        rockets_requiring_split_check: list[FlyingRocket] = []

        # --- Network Message Processing ---
        while not network_mgr.message_queue.empty():
            try:
                msg = network_mgr.message_queue.get_nowait()
                msg_type = msg.get("type")
                sender_pid = msg.get("pid")  # Player ID who sent/caused the message

                # print(f"MP Sim RX ({mp_mode}): {msg}") # Debug all messages

                if msg_type == network.MSG_TYPE_ERROR:
                    print(f"!!! Network Error: {msg.get('data')} !!!")
                    # sim_running = False # Option: Stop sim on critical error

                elif msg_type == network.MSG_TYPE_PLAYER_JOINED:
                    pid = msg.get("pid");
                    name = msg.get("name", f"Player_{pid}")
                    if pid != local_player_id:
                        connected_players[pid] = name;
                        player_names[pid] = name
                        player_ready_status[pid] = False
                        print(f"Player {pid} ({name}) joined.")
                        if current_phase == "RUNNING" and mp_mode == "HOST":
                            game_state = {}
                            for other_rocket in all_sim_rockets:  # Send state of ALL rockets/debris
                                if other_rocket.is_active:
                                    # Use sim_instance_id as key? Or PID + fragment ID? Let's use SimID for now.
                                    # --- ADD player_id to state data ---
                                    state_data = other_rocket.get_state()
                                    state_data['player_id'] = other_rocket.player_id # Ensure PID is included
                                    game_state[other_rocket.sim_instance_id] = state_data
                                    # --- End Add ---
                            state_msg = {"type": network.MSG_TYPE_GAME_STATE, "state": game_state,
                                         "blueprints": player_blueprints}  # Include blueprints
                            target_socket = next(
                                (sock for sock, info in network_mgr.clients.items() if info['id'] == pid), None)
                            if target_socket:
                                network_mgr.send_message(target_socket, state_msg); print(
                                    f"Sent game state & BPs to player {pid}.")
                            else:
                                print(f"Error: Could not find socket for player {pid} to send state.")

                elif msg_type == network.MSG_TYPE_PLAYER_LEFT:
                    pid = msg.get("pid");
                    name = player_names.get(pid, f"Player_{pid}")
                    print(f"Player {pid} ({name}) left.")
                    connected_players.pop(pid, None);
                    player_names.pop(pid, None)
                    player_blueprints.pop(pid, None);
                    player_ready_status.pop(pid, None)
                    # Find all rockets belonging to this player and mark for removal
                    rockets_of_player = [r for r in all_sim_rockets if r.player_id == pid]
                    for r in rockets_of_player:
                        if r not in rockets_to_remove_this_frame: rockets_to_remove_this_frame.append(r)
                    if pid in player_rockets: del player_rockets[pid]  # Remove primary reference
                    if current_phase == "WAITING": all_players_ready = False

                elif msg_type == network.MSG_TYPE_BLUEPRINT:
                    pid = msg.get("pid");
                    bp_json = msg.get("json_str");
                    name = msg.get("name", f"Player_{pid}");
                    bp_name = msg.get("bp_name", "Unknown Rocket")
                    print(f"Received blueprint for Player {pid} ({name}) - Rocket: {bp_name}")
                    if pid not in player_blueprints and bp_json:
                        player_blueprints[pid] = bp_json;
                        player_names[pid] = name
                        if pid not in connected_players: connected_players[pid] = name
                        if current_phase == "RUNNING":  # Late join: create rocket now
                            sim_id = next_sim_id_counter;
                            next_sim_id_counter += 1
                            new_rocket = create_rocket_instance(pid, bp_json, name, sim_id, current_sim_frame)
                            if new_rocket:
                                player_rockets[pid] = new_rocket  # Store as primary
                                newly_created_rockets_this_frame.append(new_rocket)  # Add to simulation list
                                # Host should broadcast creation? Or rely on state sync? Rely on sync for now.

                elif msg_type == network.MSG_TYPE_LAUNCH_READY:
                    pid = msg.get("pid");
                    print(f"Player {pid} signalled Launch Ready.")
                    if pid in connected_players: player_ready_status[pid] = True
                    if current_phase == "WAITING": all_players_ready = False

                elif msg_type == network.MSG_TYPE_ROCKET_UPDATE:
                    pid = msg.get("pid");
                    action = msg.get("action");
                    data = msg.get("data")
                    # Find the relevant rocket instance (could be primary or debris)
                    # Need to identify rocket uniquely - use sim_instance_id?
                    target_sim_id = data.get("sim_id", None)  # Assume state updates might include sim_id
                    target_rocket = None
                    if target_sim_id is not None:
                        target_rocket = next((r for r in all_sim_rockets if r.sim_instance_id == target_sim_id), None)
                    # --- Fallback REMOVED: Actions/states should target specific SimIDs ---
                    # else: target_rocket = player_rockets.get(pid)

                    if target_rocket and not target_rocket.is_local_player:  # Apply only to remote rockets
                        if action == "state_update":
                            target_rocket.apply_state(data)
                        else:
                            target_rocket.apply_network_action(data)  # Assumes 'data' contains action details
                    # elif not target_rocket: print(f"Warning: ROCKET_UPDATE for unknown SimID {target_sim_id} or PID {pid}") # Reduce log spam

                elif msg_type == network.MSG_TYPE_GAME_STATE:  # Client receives full state on join
                    if mp_mode == "CLIENT":
                        print("Received initial game state.");
                        state_map = msg.get("state", {});
                        received_bps = msg.get("blueprints", {})
                        player_blueprints.update(received_bps)  # Update known blueprints
                        # Clear existing rockets before applying state? Yes.
                        all_sim_rockets.clear();
                        player_rockets.clear()
                        for sim_id_str, state_data in state_map.items():
                            sim_id = int(sim_id_str);
                            # --- Retrieve PID from state_data ---
                            pid = state_data.get("player_id") # Assume get_state now includes player_id
                            if pid is None: print(f"Warning: Game state missing PID for SimID {sim_id}"); continue
                            # --- End PID Retrieve ---
                            bp_json = player_blueprints.get(pid)
                            if bp_json:
                                name = player_names.get(pid, f"Player_{pid}")
                                # Create instance - it might already exist conceptually, but easier to recreate
                                rocket = create_rocket_instance(pid, bp_json, name, sim_id,
                                                                current_sim_frame)  # Use sim_id from state
                                if rocket:
                                    # --- ADD player_id to state BEFORE applying ---
                                    # Apply state needs the PID, but it might not be in the nested state dict
                                    # It's better if get_state() includes it *within* the state dict itself.
                                    # Assuming apply_state can handle it being missing for now.
                                    rocket.apply_state(state_data)  # Apply the state *after* creation
                                    all_sim_rockets.append(rocket)
                                    # Track primary rocket reference if it's local or first for this player
                                    if rocket.is_local_player or pid not in player_rockets:
                                        player_rockets[pid] = rocket
                                    # print(f"Created/Applied state for SimID {sim_id} (Player {pid})") # Reduce spam
                            # else: print(f"Warning: Received state for Player {pid} but no blueprint found.")
                        print("Finished processing initial game state.")


                elif msg_type == network.MSG_TYPE_SET_NAME:
                    pid = msg.get("pid");
                    name = msg.get("name")
                    if pid in player_names: player_names[pid] = name
                    if pid in connected_players: connected_players[pid] = name
                    if pid in player_rockets: player_rockets[pid].player_name = name
                    # Update name for all fragments of this player too
                    for r in all_sim_rockets:
                        if r.player_id == pid: r.player_name = name
                    print(f"Updated name for Player {pid} to '{name}'")

            except queue.Empty:
                break
            except Exception as e:
                print(f"Error processing network message: {e}"); import traceback; traceback.print_exc()

        # --- Phase Logic ---
        if current_phase == "WAITING":
            all_players_ready = bool(connected_players)
            for pid in connected_players:
                if pid not in player_blueprints or not player_ready_status.get(pid, False):
                    all_players_ready = False;
                    break
            if all_players_ready:
                print("All players ready! Creating rockets and starting simulation...");
                current_phase = "RUNNING"
                player_rockets.clear();
                all_sim_rockets.clear()
                next_sim_id_counter = 0
                for pid, bp_json in player_blueprints.items():
                    if pid in connected_players:
                        sim_id = next_sim_id_counter;
                        next_sim_id_counter += 1
                        name = player_names.get(pid, f"Player_{pid}")
                        rocket = create_rocket_instance(pid, bp_json, name, sim_id, current_sim_frame)
                        if rocket:
                            player_rockets[pid] = rocket
                            all_sim_rockets.append(rocket)
                        else:
                            print(f"Failed to create rocket for player {pid} at launch!");
                            current_phase = "WAITING";
                            all_players_ready = False;
                            break
                local_rocket = player_rockets.get(local_player_id)
                if local_rocket:
                    camera.update(local_rocket.get_world_com())
                elif all_sim_rockets:
                    camera.update(all_sim_rockets[0].get_world_com())


        elif current_phase == "RUNNING":
            # --- Local Player Input Handling ---
            local_rocket = player_rockets.get(local_player_id)
            if local_rocket and local_rocket.is_active:
                root_ref = local_rocket.original_root_part_ref;
                can_control_locally = (root_ref is not None) and (root_ref in local_rocket.parts) and (
                    not root_ref.is_broken)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: sim_running = False; break
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE: sim_running = False; break
                        if can_control_locally:
                            if event.key == pygame.K_SPACE:
                                local_rocket.master_thrust_enabled = not local_rocket.master_thrust_enabled
                                action_msg = {"action": "set_master_thrust",
                                              "value": local_rocket.master_thrust_enabled,
                                              "sim_id": local_rocket.sim_instance_id} # Add SimID
                                network_send_queue.put({"type": network.MSG_TYPE_ACTION, "pid": local_player_id,
                                                        "data": action_msg})  # Send player action
                            if event.key == pygame.K_p:
                                chutes_activated_indices = []
                                for chute in local_rocket.parachutes:
                                    if not chute.deployed and not chute.is_broken:
                                        chute.deployed = True;
                                        chutes_activated_indices.append(chute.part_index)
                                if chutes_activated_indices:
                                    for part_idx in chutes_activated_indices:
                                        action_msg = {"action": "deploy", "part_idx": part_idx,
                                                      "sim_id": local_rocket.sim_instance_id} # Add SimID
                                        network_send_queue.put(
                                            {"type": network.MSG_TYPE_ACTION, "pid": local_player_id,
                                             "data": action_msg})  # Send player action
                                    # print(f"Sent deploy actions for {len(chutes_activated_indices)} chutes.") # Reduce spam
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and can_control_locally:
                        click_world_pos = pygame.math.Vector2(event.pos) + camera.offset
                        action_details = local_rocket.activate_part_at_pos(click_world_pos)
                        if action_details:
                            action_details["sim_id"] = local_rocket.sim_instance_id # Add SimID
                            network_send_queue.put({"type": network.MSG_TYPE_ACTION, "pid": local_player_id,
                                                    "data": action_details})  # Send player action
                if not sim_running: break
                if can_control_locally:
                    keys = pygame.key.get_pressed();
                    throttle_change = 0.0
                    if keys[pygame.K_w] or keys[pygame.K_UP]: throttle_change += THROTTLE_CHANGE_RATE * dt
                    if keys[pygame.K_s] or keys[pygame.K_DOWN]: throttle_change -= THROTTLE_CHANGE_RATE * dt
                    if abs(throttle_change) > 1e-6:
                        new_throttle = max(0.0, min(1.0, local_rocket.throttle_level + throttle_change))
                        if abs(new_throttle - local_rocket.throttle_level) > 1e-6:
                            local_rocket.throttle_level = new_throttle
                            action_msg = {"action": "set_throttle", "value": local_rocket.throttle_level,
                                          "sim_id": local_rocket.sim_instance_id} # Add SimID
                            network_send_queue.put({"type": network.MSG_TYPE_ACTION, "pid": local_player_id,
                                                    "data": action_msg})  # Send player action

            # --- Update All Active Rockets ---
            current_rockets_to_update = list(all_sim_rockets)  # Iterate over copy
            for rocket in current_rockets_to_update:
                if not rocket.is_active: continue
                altitude_agl = max(0, GROUND_Y - rocket.get_world_com().y);
                current_air_density = get_air_density(altitude_agl)
                send_q = network_send_queue if rocket.is_local_player else None
                rocket.update(dt, current_air_density, particle_manager, network_send_queue=send_q,
                              current_time=current_time)
                if not rocket.is_active and rocket not in rockets_to_remove_this_frame:
                    rockets_to_remove_this_frame.append(rocket)
                # Check if the update call flagged the need for a structural check
                if rocket.needs_structural_update and rocket.is_active:
                    if rocket not in rockets_requiring_split_check:
                        rockets_requiring_split_check.append(rocket)
                    # Reset flag AFTER adding to check list for this frame
                    rocket.needs_structural_update = False

            # --- Send Queued Network Messages ---
            while not network_send_queue.empty():
                try:
                    msg_to_send = network_send_queue.get_nowait()
                    # SimID should have been added during input handling if applicable
                    if mp_mode == "CLIENT":
                        if not network_mgr.send(msg_to_send): print("Error: Failed to send message to server.")
                    elif mp_mode == "HOST":
                        msg_to_send.setdefault("pid", 0)  # Host PID is 0
                        # Convert ACTION to ROCKET_UPDATE before broadcasting for consistency
                        if msg_to_send.get("type") == network.MSG_TYPE_ACTION:
                            msg_to_send["type"] = network.MSG_TYPE_ROCKET_UPDATE
                        # Broadcast relevant messages (updates/state)
                        if msg_to_send.get("type") == network.MSG_TYPE_ROCKET_UPDATE:
                            network_mgr.broadcast(msg_to_send, exclude_socket=None)
                        # else: print(f"Host ignoring local msg type: {msg_to_send.get('type')}")
                except queue.Empty:
                    break
                except AttributeError as e:
                    print(f"Error sending network message (AttributeError): {e}")
                except Exception as e:
                    print(f"Error processing send queue: {e}"); import traceback; traceback.print_exc()

            # --- Inter-Rocket Collision (MP) ---
            collision_pairs_processed_this_frame = set()
            active_rockets_list = [r for r in all_sim_rockets if r.is_active and r not in rockets_to_remove_this_frame]
            for i in range(len(active_rockets_list)):
                r1 = active_rockets_list[i];
                if not r1.parts: continue
                for j in range(i + 1, len(active_rockets_list)):
                    r2 = active_rockets_list[j];
                    if not r2.parts: continue
                    pair_key_grace = tuple(sorted((r1.sim_instance_id, r2.sim_instance_id)))
                    if pair_key_grace in collision_grace_period_pairs: continue
                    dist_sq = (r1.get_world_com() - r2.get_world_com()).length_squared()
                    r1_r = max(r1.local_bounds.width, r1.local_bounds.height) / 2.0;
                    r2_r = max(r2.local_bounds.width, r2.local_bounds.height) / 2.0
                    if dist_sq > (r1_r + r2_r + 10) ** 2: continue
                    coll_found = False;
                    c_p1 = None;
                    c_p2 = None
                    for p1 in r1.parts:
                        if p1.is_broken: continue; rect1 = r1.get_world_part_aabb(p1)
                        for p2 in r2.parts:
                            if p2.is_broken: continue; rect2 = r2.get_world_part_aabb(p2)
                            if rect1.colliderect(rect2): coll_found = True; c_p1 = p1; c_p2 = p2; break
                        if coll_found: break
                    if coll_found:
                        pair_key_proc = tuple(sorted((r1.sim_instance_id, r2.sim_instance_id)))
                        if pair_key_proc in collision_pairs_processed_this_frame: continue
                        collision_pairs_processed_this_frame.add(pair_key_proc)
                        rel_vel = r1.vel - r2.vel;
                        imp_spd = rel_vel.length()
                        # Apply damage locally
                        r1.apply_collision_damage(imp_spd, particle_manager, c_p1);
                        r2.apply_collision_damage(imp_spd, particle_manager, c_p2)
                        # Apply push locally
                        coll_norm = r1.get_world_com() - r2.get_world_com()
                        if coll_norm.length_squared() > 1e-6:
                            coll_norm.normalize_ip()
                        else:
                            coll_norm = pygame.math.Vector2(0, -1)
                        push = 2.0;
                        tot_m = r1.total_mass + r2.total_mass
                        if tot_m > 0.01: r1.pos += coll_norm * push * (
                                r2.total_mass / tot_m); r2.pos -= coll_norm * push * (
                                r1.total_mass / tot_m)
                        # Flag for split check if damage occurred (handled by update calling handle_destroyed_parts)

            # --- Process Connectivity Checks and Separations (MP - Local Prediction) ---
            split_siblings_this_frame: list[FlyingRocket] = []
            processed_split_rockets_this_frame = set()

            for rocket in rockets_requiring_split_check:  # Iterate through rockets flagged earlier
                if rocket in rockets_to_remove_this_frame or not rocket.is_active or rocket.sim_instance_id in processed_split_rockets_this_frame:
                    continue

                # Get separators that were fired this frame for this rocket
                separators_fired_this_frame = list(rocket.separators_to_fire_this_frame)
                rocket.separators_to_fire_this_frame.clear()  # Clear the queue

                # Get the current parts list (might have been reduced by damage already)
                current_parts = list(rocket.parts)

                # Determine the list of parts *if* the fired separators were removed
                potential_parts_after_separation = [p for p in current_parts if
                                                    p not in separators_fired_this_frame]

                # Check connectivity based on the potential state after separation
                temp_bp = RocketBlueprint()
                temp_bp.parts = potential_parts_after_separation  # Check connectivity *without* the fired separators
                subassemblies = temp_bp.find_connected_subassemblies()

                # --- Case 1: Split Detected ---
                if len(subassemblies) > 1:
                    processed_split_rockets_this_frame.add(rocket.sim_instance_id)
                    split_cause = "Separator" if separators_fired_this_frame else "Destruction"
                    print(
                        f"[MP:{rocket.sim_instance_id} P:{rocket.player_id}] SPLIT ({split_cause}) into {len(subassemblies)} pieces!")

                    if rocket not in rockets_to_remove_this_frame:
                        rockets_to_remove_this_frame.append(rocket)

                    original_throttle = rocket.throttle_level;
                    original_master_thrust = rocket.master_thrust_enabled
                    was_local = rocket.is_local_player;
                    original_pid = rocket.player_id;
                    original_name = rocket.player_name
                    current_split_siblings = []

                    for assembly_parts in subassemblies:
                        if not assembly_parts: continue
                        try:
                            sub_com_world = rocket.calculate_subassembly_world_com(assembly_parts)
                            contains_root = rocket.original_root_part_ref and (
                                    rocket.original_root_part_ref in assembly_parts)
                            is_primary = was_local and contains_root
                            sim_id = next_sim_id_counter;
                            next_sim_id_counter += 1
                            new_frag = FlyingRocket(
                                list(assembly_parts), sub_com_world, rocket.angle, rocket.vel, sim_id,
                                is_primary,
                                rocket.original_root_part_ref, current_sim_frame, original_pid,
                                original_name
                            )
                            new_frag.angular_velocity = rocket.angular_velocity
                            if new_frag.is_local_player:
                                new_frag.throttle_level = original_throttle;
                                new_frag.master_thrust_enabled = original_master_thrust

                            if split_cause == "Separator" and separators_fired_this_frame:
                                sep_world_pos_avg = sum(
                                    (rocket.get_world_part_center(s) for s in separators_fired_this_frame),
                                    pygame.math.Vector2()) / len(separators_fired_this_frame)
                                sep_force = separators_fired_this_frame[0].part_data.get("separation_force",
                                                                                         1000)
                                sep_vec = new_frag.get_world_com() - sep_world_pos_avg
                                if sep_vec.length_squared() > 1e-6:
                                    sep_dir = sep_vec.normalize()
                                else:
                                    sep_dir = pygame.math.Vector2(random.uniform(-1, 1),
                                                                  random.uniform(-1, 1)).normalize()
                                impulse_time = 0.05;
                                impulse_mag = (sep_force / max(0.1, new_frag.total_mass)) * impulse_time
                                new_frag.vel += sep_dir * impulse_mag;
                                new_frag.angular_velocity += random.uniform(-15, 15)

                            newly_created_rockets_this_frame.append(new_frag)
                            current_split_siblings.append(new_frag)
                            # print(f"  > Created fragment SimID:{sim_id} for Player:{original_pid} (Local:{is_primary})") # Reduce spam
                            # --- MP TODO: Host should broadcast creation/state of new fragments ---

                        except Exception as e:
                            print(f"Error MP creating split fragment: {e}")

                    split_siblings_this_frame.extend(current_split_siblings)

                # --- Case 2: No Split, but separators were fired (e.g., end cap) ---
                elif len(subassemblies) == 1 and separators_fired_this_frame:
                    # Only modify if separators were the *reason* for the check
                    print(f"[MP:{rocket.sim_instance_id}] Separator fired but no structural split.")
                    rocket.parts = potential_parts_after_separation  # Update original rocket's part list
                    # Rebuild internal component lists and physics properties
                    rocket.engines = [e for e in rocket.engines if e in rocket.parts]
                    rocket.fuel_tanks = [t for t in rocket.fuel_tanks if t in rocket.parts]
                    rocket.parachutes = [pc for pc in rocket.parachutes if pc in rocket.parts]
                    rocket.separators = [s for s in rocket.separators if s in rocket.parts]
                    rocket.engine_firing_status = {e: False for e in rocket.engines}
                    rocket._build_fuel_source_map()
                    rocket.calculate_physics_properties()
                    rocket.calculate_bounds()
                    # Fired separators state handled visually by part.separated = True
                    if not rocket.parts:  # Check if rocket is now empty
                        rocket.is_active = False
                        if rocket not in rockets_to_remove_this_frame: rockets_to_remove_this_frame.append(
                            rocket)

                # --- Case 3: No Split and no separators fired ---
                # (Check triggered by damage, but structure held) - No action needed.

            # --- Add Collision Grace Period for New Siblings ---
            if split_siblings_this_frame:
                for i_sib, r_sib1 in enumerate(split_siblings_this_frame):
                    for j_sib in range(i_sib + 1, len(split_siblings_this_frame)):
                        r_sib2 = split_siblings_this_frame[j_sib]
                        grace_pair_key = tuple(sorted((r_sib1.sim_instance_id, r_sib2.sim_instance_id)))
                        collision_grace_period_pairs[grace_pair_key] = COLLISION_GRACE_FRAMES
                split_siblings_this_frame.clear()

            # --- Update Rocket Lists (MP) ---
            if newly_created_rockets_this_frame:
                new_local_rocket_candidate = None
                for new_rocket in newly_created_rockets_this_frame:
                    if new_rocket not in all_sim_rockets:
                        all_sim_rockets.append(new_rocket)
                    # Update primary rocket reference if needed
                    if new_rocket.is_local_player:
                        new_local_rocket_candidate = new_rocket
                    # If this fragment belongs to a player who previously had no primary rocket listed, set it
                    # This logic might need refinement if multiple fragments could be considered 'primary'
                    elif new_rocket.player_id not in player_rockets:
                        player_rockets[new_rocket.player_id] = new_rocket

                if new_local_rocket_candidate: # If control transferred to a new fragment
                    old_local = player_rockets.get(local_player_id)
                    if old_local and old_local != new_local_rocket_candidate and old_local in all_sim_rockets:
                         old_local.is_local_player = False # Mark old one non-local if it still exists
                    player_rockets[local_player_id] = new_local_rocket_candidate # Update primary reference
                    print(f"Control transferred to new fragment SimID:{new_local_rocket_candidate.sim_instance_id}")

                newly_created_rockets_this_frame.clear()

            if rockets_to_remove_this_frame:
                 removed_sim_ids = {r.sim_instance_id for r in rockets_to_remove_this_frame}
                 # Filter main simulation list
                 all_sim_rockets = [r for r in all_sim_rockets if r.sim_instance_id not in removed_sim_ids]
                 # Update primary rocket references and handle loss of local control
                 pids_affected = set(); lost_local_control = False
                 for r in rockets_to_remove_this_frame:
                     pids_affected.add(r.player_id);
                     if r.is_local_player: lost_local_control = True
                 for pid in pids_affected:
                      if pid in player_rockets and player_rockets[pid].sim_instance_id in removed_sim_ids:
                           # Find a replacement primary rocket for this player if one exists
                           replacement = next((rkt for rkt in all_sim_rockets if rkt.player_id == pid and rkt.is_active and rkt.original_root_part_ref and rkt.original_root_part_ref in rkt.parts and not rkt.original_root_part_ref.is_broken), None)
                           if replacement:
                                player_rockets[pid] = replacement
                                if pid == local_player_id: # If it was the local player losing primary
                                     replacement.is_local_player = True; lost_local_control = False
                                     print(f"Control transferred to replacement SimID {replacement.sim_instance_id}.")
                           else:
                                if pid in player_rockets: del player_rockets[pid] # Check existence before deleting
                                if pid == local_player_id: print("Lost control: No suitable replacement found.")
                 # Clean grace period
                 pairs_to_del = [p for p in collision_grace_period_pairs if p[0] in removed_sim_ids or p[1] in removed_sim_ids]
                 for pair in pairs_to_del:
                      if pair in collision_grace_period_pairs: del collision_grace_period_pairs[pair]
                 rockets_to_remove_this_frame.clear()

            # --- Camera Update (MP) ---
            local_rocket = player_rockets.get(local_player_id)
            if local_rocket and local_rocket.is_active: camera.update(local_rocket.get_world_com())
            elif all_sim_rockets: # Fallback to follow first active rocket
                 first_active = next((r for r in all_sim_rockets if r.is_active), None)
                 if first_active: camera.update(first_active.get_world_com())

            # --- Periodic Ping (Client) ---
            if mp_mode == "CLIENT" and current_time - last_ping_time > 5.0:
                 network_mgr.send({"type": network.MSG_TYPE_PING}); last_ping_time = current_time

        # --- Drawing ---
        screen.fill(BLACK); draw_earth_background(screen, camera, stars); draw_terrain(screen, camera)
        total_parts_drawn = 0; total_broken_drawn = 0
        for rocket in all_sim_rockets: # Draw all rockets from the simulation list
            if rocket.is_active:
                broken_count = rocket.draw(screen, camera, particle_manager, draw_name=True)
                total_parts_drawn += len(rocket.parts); total_broken_drawn += broken_count
        particle_manager.update(dt); particle_manager.draw(screen, camera)

        # --- Draw UI (MP) ---
        if current_phase == "WAITING":
            # ... (Waiting UI drawing) ...
            status_lines = ["Waiting for Players..."]; y_pos = 50
            for pid, name in connected_players.items():
                 bp_status = "OK" if pid in player_blueprints else "Waiting"; rdy_status = "Ready" if player_ready_status.get(pid, False) else "Not Ready"
                 local_tag = "(You)" if pid == local_player_id else ""
                 status_lines.append(f" - P{pid} {name} {local_tag}: Blueprint[{bp_status}] Status[{rdy_status}]")
            for line in status_lines: surf = ui_font_large.render(line, True, WHITE); rect = surf.get_rect(center=(SCREEN_WIDTH // 2, y_pos)); screen.blit(surf, rect); y_pos += 40
        elif current_phase == "RUNNING":
            local_rocket_ref = player_rockets.get(local_player_id) # Get current primary reference
            if local_rocket_ref and local_rocket_ref in all_sim_rockets and local_rocket_ref.is_active: # Check it's still valid & in sim
                 # ... (Local player telemetry UI drawing using local_rocket_ref) ...
                bar_w = 20; bar_h = 100; bar_x = 15; bar_y = SCREEN_HEIGHT - bar_h - 40
                pygame.draw.rect(screen, COLOR_UI_BAR_BG, (bar_x, bar_y, bar_w, bar_h)); fill_h = bar_h * local_rocket_ref.throttle_level; pygame.draw.rect(screen, COLOR_UI_BAR, (bar_x, bar_y + bar_h - fill_h, bar_w, fill_h))
                pygame.draw.rect(screen, WHITE, (bar_x, bar_y, bar_w, bar_h), 1); th_label = ui_font.render("Thr",True,WHITE); screen.blit(th_label, (bar_x, bar_y + bar_h + 5)); th_value = ui_font.render(f"{local_rocket_ref.throttle_level*100:.0f}%",True,WHITE); screen.blit(th_value, (bar_x, bar_y - 18))
                alt_agl = max(0, GROUND_Y - local_rocket_ref.get_lowest_point_world().y); alt_msl = GROUND_Y - local_rocket_ref.get_world_com().y
                root_ref = local_rocket_ref.original_root_part_ref; can_control = local_rocket_ref.is_local_player and root_ref and root_ref in local_rocket_ref.parts and not root_ref.is_broken
                ctrl_status = "OK" if can_control else "NO CTRL"; thrust_status = "ON" if local_rocket_ref.master_thrust_enabled else "OFF"; landed_status = "LANDED" if local_rocket_ref.landed else "FLYING"
                max_temp_k = local_rocket_ref.max_temp_reading; temp_color = WHITE; hottest_allowable = DEFAULT_MAX_TEMP
                if local_rocket_ref.parts: temps = [p.part_data.get('max_temp',DEFAULT_MAX_TEMP) for p in local_rocket_ref.parts if p.part_data]; hottest_allowable = max(temps) if temps else DEFAULT_MAX_TEMP
                if max_temp_k > REENTRY_EFFECT_THRESHOLD_TEMP: temp_color = (255,255,0);
                if max_temp_k > hottest_allowable * 0.9: temp_color = (255,100,0);
                if max_temp_k > hottest_allowable: temp_color = RED
                total_fuel = local_rocket_ref.get_total_current_fuel()
                status_texts = [f"Alt(AGL): {alt_agl:.1f}m", f"Alt(MSL): {alt_msl:.1f}m", f"Vvel: {local_rocket_ref.vel.y:.1f} m/s", f"Hvel: {local_rocket_ref.vel.x:.1f} m/s", f"Speed: {local_rocket_ref.vel.length():.1f} m/s", f"Angle: {local_rocket_ref.angle:.1f} deg", f"AngVel: {local_rocket_ref.angular_velocity:.1f} d/s", f"Thr: {local_rocket_ref.throttle_level*100:.0f}% [{thrust_status}]", f"Fuel: {total_fuel:.1f} units", f"Mass: {local_rocket_ref.total_mass:.1f} kg", f"Control: {ctrl_status}", f"Status: {landed_status}"]
                text_y_start = 10; control_color = WHITE if can_control else RED
                for i, text in enumerate(status_texts): line_color = temp_color if "MaxTemp" in text else (control_color if "Control" in text else WHITE); text_surf = ui_font.render(text, True, line_color); screen.blit(text_surf, (bar_x + bar_w + 10, text_y_start + i * 18))
                temp_surf = ui_font.render(f"MaxTemp: {max_temp_k:.0f} K", True, temp_color); screen.blit(temp_surf, (bar_x + bar_w + 10, text_y_start + len(status_texts) * 18))
            else: # Local player has no active primary rocket
                 destroyed_text = ui_font_large.render("LOCAL CONTROL LOST",True,RED); text_rect = destroyed_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)); screen.blit(destroyed_text,text_rect)

        # --- Debug Info (MP) ---
        fps = clock.get_fps(); debug_y = 10; debug_x = SCREEN_WIDTH - 140
        fps_text = ui_font.render(f"FPS: {fps:.1f}", True, WHITE); screen.blit(fps_text, (debug_x, debug_y)); debug_y += 18
        mode_text = ui_font.render(f"Mode: {mp_mode}", True, WHITE); screen.blit(mode_text, (debug_x, debug_y)); debug_y += 18
        pid_text = ui_font.render(f"PID: {local_player_id}", True, WHITE); screen.blit(pid_text, (debug_x, debug_y)); debug_y += 18
        rocket_count = len(all_sim_rockets) # Count all active rockets/debris
        obj_text = ui_font.render(f"Sim Rkts: {rocket_count}", True, WHITE); screen.blit(obj_text, (debug_x, debug_y)); debug_y += 18
        parts_text = ui_font.render(f"Parts: {total_parts_drawn}", True, WHITE); screen.blit(parts_text, (debug_x, debug_y)); debug_y += 18
        particle_text = ui_font.render(f"Particles: {len(particle_manager.particles)}", True, WHITE); screen.blit(particle_text, (debug_x, debug_y)); debug_y += 18
        grace_text = ui_font.render(f"Grace Pairs: {len(collision_grace_period_pairs)}", True, WHITE); screen.blit(grace_text, (debug_x, debug_y)); debug_y += 18
        phase_text = ui_font.render(f"Phase: {current_phase}", True, WHITE); screen.blit(phase_text, (debug_x, debug_y)); debug_y += 18
        conn_text = ui_font.render(f"Connected: {len(connected_players)}", True, WHITE); screen.blit(conn_text, (debug_x, debug_y)); debug_y += 18


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