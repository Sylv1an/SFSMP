# flight_sim.py
import pygame
import math
import sys
import random
import time
import os # Added os

# Make sure parts module is imported correctly after changes
from parts import draw_part_shape, get_part_data, PARTS_CATALOG # Added PARTS_CATALOG
from rocket_data import RocketBlueprint, PlacedPart
from ui_elements import SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, BLACK, GRAY, RED, GREEN, BLUE, LIGHT_GRAY

# --- Flight Sim Constants ---
# Adjusted Gravity and Torque, added Drag constants
GRAVITY = 9.81 * 6 # Slightly lower gravity?
ROTATION_SPEED = 200 # Keep this for now
REACTION_WHEEL_TORQUE = 10000 # Adjusted torque
ANGULAR_DAMPING = 0.3
COLLISION_DAMAGE_FACTOR = 0.7 # Increased damage slightly
MIN_IMPACT_VEL_DAMAGE = 2.5 # Lower damage threshold
THROTTLE_CHANGE_RATE = 0.5
GROUND_Y = 1000
WORLD_WIDTH = 5000
BLUE_SKY_Y_LIMIT = -2000
SPACE_Y_LIMIT = -15000
STAR_COUNT = 200
STAR_FIELD_DEPTH = 10000
# Air Density (simple model: sea level and vacuum)
AIR_DENSITY_SEA_LEVEL = 1.225
AIR_DENSITY_VACUUM = 0.0
ATMOSPHERE_HEIGHT = 10000 # Altitude where density becomes negligible (adjust)

COLOR_SKY_BLUE = pygame.Color(135, 206, 250); COLOR_SPACE_BLACK = pygame.Color(0, 0, 0)
COLOR_HORIZON = pygame.Color(170, 210, 230); COLOR_GROUND = pygame.Color(0, 150, 0)
COLOR_FLAME = pygame.Color(255, 100, 0); COLOR_UI_BAR = pygame.Color(0, 200, 0)
COLOR_UI_BAR_BG = pygame.Color(50, 50, 50)
COLOR_EXPLOSION = [pygame.Color(255,255,0), pygame.Color(255,150,0), pygame.Color(200,50,0), pygame.Color(GRAY)]
COLOR_ENGINE_ENABLED = GREEN; COLOR_ENGINE_DISABLED = RED
COLOR_ACTIVATABLE_READY = BLUE
COLOR_ACTIVATABLE_USED = GRAY

# --- Simple Particle Class for Explosions (keep as is) ---
class ExplosionEffect:
    def __init__(self, pos, num_particles=15, max_life=0.5, max_speed=100):
        self.particles=[]; self.pos=pygame.math.Vector2(pos); self.max_life=max_life; global COLOR_EXPLOSION; COLOR_EXPLOSION = [pygame.Color(255,255,0), pygame.Color(255,150,0), pygame.Color(200,50,0), pygame.Color(GRAY)] # Define here if not imported
        for _ in range(num_particles): a=random.uniform(0,360); s=random.uniform(max_speed*0.2,max_speed); v=pygame.math.Vector2(s,0).rotate(a); l=random.uniform(max_life*0.3,max_life); c=random.choice(COLOR_EXPLOSION); r=random.uniform(2,5); self.particles.append([self.pos.copy(),v,l,l,c,r]); self.is_alive=True
    def update(self, dt):
        if not self.is_alive: return
        apf = False
        for p in self.particles:
            if p[2]>0: p[0]+=p[1]*dt; p[1]*=0.95; p[2]-=dt; apf=True
        self.is_alive = apf
    def draw(self, surface, camera):
        if not self.is_alive: return
        for p in self.particles:
            rl = p[2];
            if rl > 0:
                sp=camera.apply(p[0]); ml=p[3]; af=max(0,rl/ml)if ml>0 else 0; bc=p[4]; br=p[5]; dc=(int(bc.r*af),int(bc.g*af),int(bc.b*af)); dr=int(br*af);
                if dc!=(0,0,0) and dr > 0: pygame.draw.circle(surface,dc,sp,dr)

# --- Camera Class (keep as is) ---
class Camera:
    def __init__(self, width, height): self.camera_rect=pygame.Rect(0,0,width,height); self.width=width; self.height=height; self.offset=pygame.math.Vector2(0,0)
    def apply(self, target_pos): return target_pos-self.offset
    def apply_rect(self, target_rect): return target_rect.move(-self.offset.x,-self.offset.y)
    def update(self, target_pos): x=target_pos.x-self.width//2; y=target_pos.y-self.height//2; self.offset=pygame.math.Vector2(x,y)

# --- FlyingRocket Class ---
class FlyingRocket:
    def __init__(self, parts_list: list[PlacedPart], initial_pos_offset: pygame.math.Vector2, initial_angle=0, initial_vel=pygame.math.Vector2(0,0), sim_instance_id=0, is_primary_control=False, original_root_ref=None):
        """
        Initializes a FlyingRocket from a list of PRE-CONNECTED PlacedPart objects.
        initial_pos_offset is applied to the calculated CoM of the parts_list.
        """
        self.sim_instance_id = sim_instance_id
        # Directly use the provided list of connected parts
        self.parts = parts_list
        if not self.parts:
            raise ValueError("Cannot initialize FlyingRocket with an empty parts list.")

        # Determine the blueprint name (e.g., from the first part's original blueprint or generic)
        # This info might be lost, use a generic name or ID
        self.blueprint_name = f"Rocket_{sim_instance_id}"

        # Reference to the absolute original command pod (passed down during splits)
        self.original_root_part_ref = original_root_ref if original_root_ref else (self.parts[0] if self.parts[0].part_data.get("type")=="CommandPod" else None)

        # Does this specific instance currently have the primary control focus?
        self.has_active_control = is_primary_control

        # --- Calculate Initial Physics State ---
        # 1. Find the geometric center of the provided parts (relative to blueprint 0,0)
        if len(self.parts) > 0:
             geom_center = sum((p.relative_pos for p in self.parts), pygame.math.Vector2()) / len(self.parts)
        else:
             geom_center = pygame.math.Vector2(0,0) # Should not happen

        # 2. The rocket's initial position 'pos' refers to the blueprint (0,0) origin's world position.
        #    We want the initial CoM to be near the target position (e.g., launchpad + offset).
        #    Need to calculate initial CoM offset relative to geom_center first.
        self.pos = pygame.math.Vector2(0,0) # Placeholder
        self.vel = pygame.math.Vector2(initial_vel)
        self.acc = pygame.math.Vector2(0,0)
        self.angle = initial_angle
        self.angular_velocity = 0.0

        # Initialize components (fuel needs recalculating based on ONLY these parts)
        self.engines = []
        self.fuel_tanks = []
        self.parachutes = []
        self.separators = []
        total_fuel_cap_this_assembly = 0

        for i, part in enumerate(self.parts):
            # Reset runtime state for these parts within this new instance
            part.current_hp = part.part_data.get("max_hp", 100) # Reset HP
            part.is_broken = False
            part.engine_enabled = True
            part.deployed = False
            part.separated = False
            part.part_index = i # Index within this instance

            pt = part.part_data.get("type")
            if pt == "Engine": self.engines.append(part)
            elif pt == "FuelTank":
                 self.fuel_tanks.append(part)
                 total_fuel_cap_this_assembly += part.part_data.get("fuel_capacity", 0)
            elif pt == "Parachute": self.parachutes.append(part)
            elif pt == "Separator": self.separators.append(part)

        # Initial fuel is the capacity of tanks *in this assembly only*
        self.current_fuel = total_fuel_cap_this_assembly
        self.fuel_mass_per_unit = 0.1

        # Calculate initial physics properties based *only* on parts_list
        self.calculate_physics_properties() # Calculates mass, CoM offset (relative to 0,0)
        self.calculate_bounds()

        # 3. Set the actual initial world position 'self.pos'
        #    So that the calculated CoM ends up at the desired start location.
        #    Target World CoM = initial_pos_offset (e.g. launchpad ground pos)
        #    self.pos = Target World CoM - (CoM Offset rotated by initial angle)
        initial_com_offset_rotated = self.center_of_mass_offset.rotate(-self.angle)
        self.pos = initial_pos_offset - initial_com_offset_rotated

        # Effects and state flags
        self.effects = []
        self.landed = False # Recalculate landing state needed
        self.thrusting = False
        self.is_active = True
        self.pending_separation = []
        # Flag to signal main loop that this rocket needs a connectivity re-check
        self.needs_connectivity_check = False
        self.throttle_level = 0.0
        self.master_thrust_enabled = False

    def calculate_subassembly_world_com(self, subassembly_parts: list[PlacedPart]) -> pygame.math.Vector2:
        """ Calculates the world CoM of a given list of parts belonging to this rocket instance. """
        if not subassembly_parts:
            # Return current CoM as fallback, though this shouldn't be called with empty list
            return self.get_world_com()

        sub_mass = 0.0
        sub_com_numerator = pygame.math.Vector2(0, 0)
        # Note: Fuel calculation here is tricky. For simplicity, assume fuel is distributed
        # proportionally based on the tanks *remaining* in the subassembly.
        # A more accurate model would track fuel per tank.

        # Quick fuel estimate for CoM calc (can be refined)
        sub_fuel_cap = sum(
            p.part_data.get("fuel_capacity", 0) for p in subassembly_parts if p.part_data.get("type") == "FuelTank")
        total_original_fuel_cap = sum(
            p.part_data.get("fuel_capacity", 0) for p in self.fuel_tanks)  # Use original tank list capacity
        total_original_fuel_cap = max(1.0, total_original_fuel_cap)
        # Estimate fuel in this subassembly based on proportion of capacity
        estimated_sub_fuel = self.current_fuel * (sub_fuel_cap / total_original_fuel_cap)
        sub_fuel_mass = estimated_sub_fuel * self.fuel_mass_per_unit
        sub_fuel_cap = max(1.0, sub_fuel_cap)  # Avoid div by zero

        for part in subassembly_parts:
            part_mass = part.part_data.get("mass", 0)
            if part.part_data.get("type") == "FuelTank":
                # Distribute estimated subassembly fuel mass among its tanks
                part_fuel_mass = sub_fuel_mass * (part.part_data.get("fuel_capacity", 0) / sub_fuel_cap)
                part_mass += part_fuel_mass

            sub_mass += part_mass
            # Use the part's relative_pos (relative to blueprint 0,0)
            sub_com_numerator += part.relative_pos * part_mass

        if sub_mass > 0.01:
            sub_com_offset_relative = sub_com_numerator / sub_mass
        else:
            # Fallback: geometric center relative to 0,0
            if subassembly_parts:
                sub_com_offset_relative = sum((p.relative_pos for p in subassembly_parts), pygame.math.Vector2()) / len(
                    subassembly_parts)
            else:
                sub_com_offset_relative = pygame.math.Vector2(0, 0)

        # Convert relative CoM offset to world position
        sub_com_offset_rotated = sub_com_offset_relative.rotate(-self.angle)
        sub_world_com = self.pos + sub_com_offset_rotated
        return sub_world_com

    def calculate_physics_properties(self):
        """ Recalculates mass, CoM, and MoI based on current parts and fuel. """
        total_m = 0.0
        com_numerator = pygame.math.Vector2(0, 0)
        moi_sum = 0.0 # Sum of I_part + m_part * d^2 (Parallel Axis Theorem)

        if not self.parts:
            self.total_mass = 0.01 # Avoid division by zero
            self.center_of_mass_offset = pygame.math.Vector2(0, 0)
            self.moment_of_inertia = 1.0
            self.dry_mass = 0.0
            return

        fuel_mass_total = self.current_fuel * self.fuel_mass_per_unit
        total_tank_capacity = sum(p.part_data.get("fuel_capacity", 0) for p in self.fuel_tanks)
        total_tank_capacity = max(1.0, total_tank_capacity)

        self.dry_mass = sum(p.part_data.get("mass", 0) for p in self.parts)

        for part in self.parts:
            part_mass = part.part_data.get("mass", 0)
            if part.part_data.get("type") == "FuelTank":
                part_fuel_mass = fuel_mass_total * (part.part_data.get("fuel_capacity", 0) / total_tank_capacity)
                part_mass += part_fuel_mass

            total_m += part_mass
            com_numerator += part.relative_pos * part_mass

        self.total_mass = max(0.01, total_m)

        if self.total_mass > 0.01:
            self.center_of_mass_offset = com_numerator / self.total_mass
        else:
            self.center_of_mass_offset = pygame.math.Vector2(0, 0)

        for part in self.parts:
             part_mass = part.part_data.get("mass", 0)
             if part.part_data.get("type") == "FuelTank":
                 part_fuel_mass = fuel_mass_total * (part.part_data.get("fuel_capacity", 0) / total_tank_capacity)
                 part_mass += part_fuel_mass

             w = part.part_data.get("width", 1)
             h = part.part_data.get("height", 1)
             i_part = (1/12.0) * part_mass * (w**2 + h**2)
             dist_vec = part.relative_pos - self.center_of_mass_offset
             d_sq = dist_vec.length_squared()
             moi_sum += i_part + part_mass * d_sq

        self.moment_of_inertia = max(1.0, moi_sum)

    def calculate_bounds(self):
        """ Calculates the AABB of the rocket in its local coordinate system (relative to blueprint 0,0). """
        if not self.parts:
            self.local_bounds = pygame.Rect(0, 0, 0, 0)
            return

        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')

        for p in self.parts:
            half_w = p.part_data['width'] / 2
            half_h = p.part_data['height'] / 2
            center_x = p.relative_pos.x
            center_y = p.relative_pos.y
            min_x = min(min_x, center_x - half_w)
            max_x = max(max_x, center_x + half_w)
            min_y = min(min_y, center_y - half_h)
            max_y = max(max_y, center_y + half_h)

        if min_x == float('inf'):
             self.local_bounds = pygame.Rect(0,0,0,0)
        else:
             self.local_bounds = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)

    def get_world_com(self):
        """ Calculates the absolute world position of the center of mass. """
        com_offset_rotated = self.center_of_mass_offset.rotate(-self.angle)
        return self.pos + com_offset_rotated

    def get_world_part_center(self, part: PlacedPart):
        """ Calculates the absolute world position of a specific part's center. """
        # Assumes part.relative_angle is 0 for AABB calculation simplicity
        part_offset_rotated = part.relative_pos.rotate(-self.angle)
        return self.pos + part_offset_rotated

    def get_parts_near_world_pos(self, world_pos: pygame.math.Vector2, radius: float = 20.0):
        """ Finds parts whose centers are within radius of world_pos. """
        nearby_parts = []
        radius_sq = radius * radius
        for part in self.parts:
            part_center_world = self.get_world_part_center(part)
            if (part_center_world - world_pos).length_squared() < radius_sq:
                nearby_parts.append(part)
        return nearby_parts

    def get_lowest_point_world(self) -> pygame.math.Vector2:
        """ Finds the lowest point of the rocket in world coordinates, considering rotation. """
        if not self.parts:
            return self.pos

        lowest_y = float('-inf')
        lowest_point_world = self.pos # Default

        for part in self.parts:
            part_center_world = self.get_world_part_center(part)
            w = part.part_data['width']
            h = part.part_data['height']
            part_world_angle = self.angle # Assumes part.relative_angle is 0

            corners_local = [
                pygame.math.Vector2(-w/2, -h/2), pygame.math.Vector2(w/2, -h/2),
                pygame.math.Vector2(w/2, h/2), pygame.math.Vector2(-w/2, h/2)
            ]
            for corner_local in corners_local:
                corner_rotated = corner_local.rotate(-part_world_angle)
                corner_world = part_center_world + corner_rotated
                if corner_world.y > lowest_y:
                    lowest_y = corner_world.y
                    lowest_point_world = corner_world

        if lowest_y == float('-inf'):
             if self.local_bounds.height > 0:
                 bottom_center_local = pygame.math.Vector2(self.local_bounds.centerx, self.local_bounds.bottom)
                 return self.pos + bottom_center_local.rotate(-self.angle)
             else:
                 return self.get_world_com()

        return lowest_point_world

    def apply_force(self, force_vector, application_point_world=None):
        """ DEPRECATED - Use calculations within update loop.
            Keeping structure for potential future use or direct impulse application.
            Currently only applies linear acceleration part. Torque is handled in update.
        """
        if self.total_mass <= 0.01: return
        # Only applies linear acceleration component here
        self.acc += force_vector / self.total_mass
        # Torque calculation is done in update based on net_torque

    def consume_fuel(self, amount):
        """ Consumes fuel, returns True if successful, updates physics. """
        consumed = min(self.current_fuel, amount)
        if consumed > 0:
            self.current_fuel -= consumed
            self.calculate_physics_properties() # Recalculate mass, CoM, MoI
            return True
        return False

    def get_thrust_data(self) -> tuple[pygame.math.Vector2, pygame.math.Vector2 | None, float]:
        """ Calculates total potential thrust vector, world application point, and fuel consumption rate. """
        total_thrust_magnitude_potential = 0.0 # Sum of potential thrust magnitudes at 100% throttle
        thrust_torque_numerator = pygame.math.Vector2(0, 0) # For CoT calc (weighted sum of positions)
        total_consumption_rate_100 = 0.0
        active_engine_count = 0

        if not self.master_thrust_enabled or self.throttle_level <= 0 or self.current_fuel <= 0:
            return pygame.math.Vector2(0,0), None, 0.0

        # Iterate through engines to sum potential thrust and consumption
        for engine in self.engines:
            if engine.is_broken or not engine.engine_enabled:
                continue

            thrust_mag = engine.part_data.get("thrust", 0)
            consumption = engine.part_data.get("fuel_consumption", 0)

            # Get engine position relative to blueprint origin and rotate to world offset
            engine_world_offset = engine.relative_pos.rotate(-self.angle)

            # Accumulate for CoT calculation and total potential thrust/consumption
            total_thrust_magnitude_potential += thrust_mag
            # Weight engine position by its thrust magnitude for CoT average
            thrust_torque_numerator += engine_world_offset * thrust_mag
            total_consumption_rate_100 += consumption
            active_engine_count += 1

        # Calculate Total Thrust Force Vector and Application Point
        total_thrust_force = pygame.math.Vector2(0, 0)
        world_thrust_application_point = None

        if active_engine_count > 0 and total_thrust_magnitude_potential > 0:
            # Calculate the magnitude of thrust actually applied this frame
            current_total_thrust_magnitude = total_thrust_magnitude_potential * self.throttle_level

            # Define thrust direction in rocket's local frame (Upwards, along -Y axis)
            local_thrust_direction = pygame.math.Vector2(0, -1)
            # Rotate this local direction by the rocket's world angle
            world_thrust_direction = local_thrust_direction.rotate(-self.angle)

            # Calculate the final thrust vector for this frame
            total_thrust_force = world_thrust_direction * current_total_thrust_magnitude

            # Calculate Center of Thrust (CoT) application point in world coordinates
            # Average offset weighted by potential thrust magnitude
            average_world_offset = thrust_torque_numerator / total_thrust_magnitude_potential
            # Add rocket's world position (origin) to get absolute CoT
            world_thrust_application_point = self.pos + average_world_offset

        # Return total force vector, world application point, and consumption rate (at 100% throttle)
        return total_thrust_force, world_thrust_application_point, total_consumption_rate_100

    # *** NEW METHOD ***
    def get_world_part_aabb(self, part: PlacedPart) -> pygame.Rect:
            """
            Calculates an approximate world Axis-Aligned Bounding Box for a part.
            NOTE: Ignores part rotation relative to the rocket body for simplicity.
                  Uses the rocket's overall angle for placement but not for box orientation.
            """
            part_data = part.part_data
            w = part_data.get('width', 1)
            h = part_data.get('height', 1)
            # Get the part's center in world coordinates
            world_center = self.get_world_part_center(part)
            # Create the AABB centered on the world position
            aabb = pygame.Rect(0, 0, w, h)
            aabb.center = world_center
            return aabb

    # *** MODIFIED METHOD SIGNATURE ***
    def apply_collision_damage(self, impact_velocity_magnitude, specific_part_to_damage: PlacedPart | None = None):
        """
        Applies damage based on impact velocity.
        If specific_part_to_damage is provided (inter-rocket collision), applies damage there primarily.
        Otherwise (ground collision), applies to lowest part(s).
        """
        # Check minimum impact threshold
        if impact_velocity_magnitude < MIN_IMPACT_VEL_DAMAGE:
            return # No damage below threshold

        # Calculate damage amount
        damage = (impact_velocity_magnitude ** 1.8) * COLLISION_DAMAGE_FACTOR
        print(f"[{self.sim_instance_id}] Impact Vel: {impact_velocity_magnitude:.1f} -> Damage: {damage:.1f}")

        # Determine which parts to apply damage to
        impacted_parts = []
        if specific_part_to_damage and specific_part_to_damage in self.parts:
             # Target the specific part from inter-rocket collision
             impacted_parts = [specific_part_to_damage]
             print(f"  Targeting specific part: {specific_part_to_damage.part_id}")
        elif not specific_part_to_damage: # Ground collision (specific_part is None)
            lowest_point_world = self.get_lowest_point_world()
            impacted_parts = self.get_parts_near_world_pos(lowest_point_world, radius=15.0)
            if not impacted_parts and self.parts: # Fallback if radius check fails
                 lowest_part = min(self.parts, key=lambda p: self.get_world_part_center(p).y)
                 if lowest_part: impacted_parts = [lowest_part]
        # Else: specific_part was provided but not found in self.parts - do nothing? Or log error?

        # Apply damage to the determined parts
        parts_destroyed_this_impact = []
        for part in impacted_parts:
            if part.is_broken: continue # Skip already broken parts

            part.current_hp -= damage
            print(f"  Part '{part.part_id}' HP: {part.current_hp:.0f} / {part.part_data.get('max_hp', 1)}")

            if part.current_hp <= 0:
                print(f"  >> Part '{part.part_id}' BROKEN! <<")
                part.is_broken = True
                part.current_hp = 0
                parts_destroyed_this_impact.append(part)
                part_center_world = self.get_world_part_center(part)
                # Create explosion effect
                try:
                    # Make sure ExplosionEffect class is defined globally or imported
                    self.effects.append(ExplosionEffect(part_center_world))
                except NameError:
                    print("Error: ExplosionEffect class not defined or imported.")

        # If parts were destroyed, trigger the handling logic
        if parts_destroyed_this_impact:
            self.handle_destroyed_parts(parts_destroyed_this_impact)

    def handle_destroyed_parts(self, destroyed_parts: list[PlacedPart]):
        """ Removes destroyed parts and sets flag for main loop connectivity check. """
        if not destroyed_parts: return

        print(f"[{self.sim_instance_id}] Handling destruction: {[p.part_id for p in destroyed_parts]}")
        # Directly remove parts from self.parts
        original_part_count = len(self.parts)
        self.parts = [p for p in self.parts if p not in destroyed_parts]

        # Check if the original root part reference was destroyed
        if self.original_root_part_ref and self.original_root_part_ref in destroyed_parts:
            print(f"[{self.sim_instance_id}] Original root part destroyed!")
            self.has_active_control = False # Lost potential control
            # The main loop will handle finding a new controllable rocket if needed

        # Check if this instance is now empty
        if not self.parts:
            print(f"[{self.sim_instance_id}] All parts destroyed!")
            self.is_active = False # Mark for removal
            self.needs_connectivity_check = False # No need to check empty list
            return # Nothing more to do

        # If parts were actually removed, signal that connectivity needs re-checking
        if len(self.parts) < original_part_count:
            self.needs_connectivity_check = True
            print(f"[{self.sim_instance_id}] Marking for connectivity check after destruction.")

            # Update component lists (engines, tanks etc.) AFTER marking for check
            self.engines = [e for e in self.engines if e in self.parts]
            self.fuel_tanks = [t for t in self.fuel_tanks if t in self.parts]
            self.parachutes = [pc for pc in self.parachutes if pc in self.parts]
            self.separators = [s for s in self.separators if s in self.parts]

            # Recalculate physics immediately for the potentially smaller remaining assembly
            # The main loop will handle splitting later if check confirms it needed
            self.calculate_physics_properties()
            self.calculate_bounds()
        else:
             # No parts removed (e.g. only broken parts were passed in, already filtered?)
             self.needs_connectivity_check = False

    def activate_part_at_pos(self, click_world_pos):
        """ Toggles state of activatable parts (engines, parachutes, separators) near click. """
        clicked_part = None
        min_dist_sq = 20**2

        parts_to_check = self.engines + self.parachutes + self.separators
        for part in parts_to_check:
            if part.is_broken: continue
            part_center_world = self.get_world_part_center(part)
            dist_sq = (part_center_world - click_world_pos).length_squared()
            if dist_sq < min_dist_sq:
                clicked_part = part
                min_dist_sq = dist_sq

        if not clicked_part: return False

        part_type = clicked_part.part_data.get("type")
        activatable = clicked_part.part_data.get("activatable", False)
        if not activatable: return False

        if part_type == "Engine":
            clicked_part.engine_enabled = not clicked_part.engine_enabled
            print(f"Toggled engine {clicked_part.part_id} {'ON' if clicked_part.engine_enabled else 'OFF'}")
            return True
        elif part_type == "Parachute":
            if not clicked_part.deployed:
                clicked_part.deployed = True
                print(f"Deployed parachute {clicked_part.part_id}!")
                return True
            else: return False # Already deployed
        elif part_type == "Separator":
            if not clicked_part.separated:
                 if clicked_part not in self.pending_separation:
                     self.pending_separation.append(clicked_part)
                     print(f"Activated separator {clicked_part.part_id} - pending separation.")
                     clicked_part.separated = True # Mark visually immediately
                 return True
            else: return False # Already separated
        return False


    def update(self, dt, current_air_density):
        """ Main physics and state update for the rocket. """
        if not self.is_active or not self.parts:
            # Still update effects even if rocket is inactive
            [e.update(dt) for e in self.effects]; self.effects = [e for e in self.effects if e.is_alive]
            return

        # --- Reset forces and torque for this frame ---
        self.acc = pygame.math.Vector2(0, 0)
        net_torque = 0.0 # Accumulate torque here

        # 1. Gravity
        gravity_force = pygame.math.Vector2(0, GRAVITY * self.total_mass)
        if self.total_mass > 0.01:
            self.acc += gravity_force / self.total_mass # Gravity acts on CoM, no torque relative to CoM

        # 2. Thrust
        thrust_force, thrust_app_point_world, consumption_rate_100 = self.get_thrust_data()
        self.thrusting = False
        if thrust_force.length_squared() > 0:
            fuel_needed = consumption_rate_100 * self.throttle_level * dt
            if self.consume_fuel(fuel_needed):
                self.thrusting = True
                if self.total_mass > 0.01:
                    # Apply linear acceleration from thrust
                    self.acc += thrust_force / self.total_mass
                    # Calculate torque from thrust
                    if thrust_app_point_world:
                        world_com = self.get_world_com()
                        radius_vec = thrust_app_point_world - world_com
                        thrust_torque = radius_vec.x * thrust_force.y - radius_vec.y * thrust_force.x
                        net_torque += thrust_torque

        # 3. Aerodynamic Drag
        drag_force = pygame.math.Vector2(0,0) # Initialize drag force
        if current_air_density > AIR_DENSITY_VACUUM and self.vel.length_squared() > 0.1:
            base_drag_coeff = 0.8
            # Use local_bounds width/height for area - crude approximation!
            # Effective area should depend on orientation relative to velocity vector
            vel_angle = self.vel.angle_to(pygame.math.Vector2(0,-1)) # Angle relative to up
            # Very basic area approximation: wider profile if sideways
            effective_width = abs(self.local_bounds.width * math.sin(math.radians(vel_angle))) + abs(self.local_bounds.height * math.cos(math.radians(vel_angle)))
            base_area = max(1.0, effective_width) # Use the calculated profile width as area estimate
            total_drag_coeff = base_drag_coeff
            effective_area = base_area

            deployed_chute_drag = 0
            for chute in self.parachutes:
                if chute.deployed and not chute.is_broken:
                     deployed_chute_drag += chute.part_data.get("deploy_drag", 0)
                     effective_area += chute.part_data.get("width",5) * chute.part_data.get("deploy_area_factor", 10)

            total_drag_coeff += deployed_chute_drag

            vel_mag_sq = self.vel.length_squared()
            vel_mag = math.sqrt(vel_mag_sq)
            drag_force_magnitude = 0.5 * current_air_density * vel_mag_sq * total_drag_coeff * effective_area
            if vel_mag > 0: # Avoid division by zero if velocity is zero
                 drag_force = -self.vel.normalize() * drag_force_magnitude

            # Apply drag force (linear)
            if self.total_mass > 0.01:
                self.acc += drag_force / self.total_mass
            # Calculate drag torque (apply at CoM approx for now)
            drag_app_point = self.get_world_com()
            world_com = self.get_world_com()
            radius_vec = drag_app_point - world_com
            drag_torque = radius_vec.x * drag_force.y - radius_vec.y * drag_force.x
            net_torque += drag_torque


        # 4. Rotational Control (Reaction Wheels)
        root_ref = self.original_root_part_ref  # Get the reference
        self.has_active_control = (root_ref is not None) and (root_ref in self.parts) and (not root_ref.is_broken)
        control_torque_input = 0
        if self.has_active_control:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:  # A/Left rotates Counter-Clockwise (+)
                control_torque_input += REACTION_WHEEL_TORQUE
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:  # D/Right rotates Clockwise (-)
                control_torque_input -= REACTION_WHEEL_TORQUE
            net_torque += control_torque_input


        # 5. Integration
        # Update velocity based on total linear acceleration
        self.vel += self.acc * dt
        # Update position based on new velocity
        self.pos += self.vel * dt

        # Calculate angular acceleration from net torque
        if self.moment_of_inertia > 0:
            angular_acceleration_rad = net_torque / self.moment_of_inertia
            angular_acceleration_deg = math.degrees(angular_acceleration_rad)
            # Update angular velocity
            self.angular_velocity += angular_acceleration_deg * dt

        # Apply angular damping
        self.angular_velocity *= (1.0 - ANGULAR_DAMPING * dt)
        # Update angle
        self.angle = (self.angle + self.angular_velocity * dt) % 360


        # 6. Ground Collision Check
        self.landed = False # Assume not landed unless check passes
        lowest_point = self.get_lowest_point_world()

        if lowest_point.y >= GROUND_Y:
            impact_velocity = self.vel.copy()
            impact_vel_mag = impact_velocity.length()

            should_apply_damage = False
            if not self.thrusting:
                 if impact_vel_mag >= MIN_IMPACT_VEL_DAMAGE:
                      should_apply_damage = True
            elif impact_velocity.y > 1.0 and impact_vel_mag >= MIN_IMPACT_VEL_DAMAGE:
                 should_apply_damage = True

            if should_apply_damage:
                self.apply_collision_damage(impact_vel_mag, specific_part_to_damage=None)

            self.landed = True # Set landed flag *after* damage check

            if self.is_active and self.parts:
                lowest_point_after_damage = self.get_lowest_point_world()
                correction = lowest_point_after_damage.y - GROUND_Y
                if correction > 0:
                    self.pos.y -= correction
                self.vel.y = 0
                self.vel.x *= 0.6
                self.angular_velocity = 0
            else:
                 self.vel = pygame.math.Vector2(0,0)

        # 7. Update Effects
        [e.update(dt) for e in self.effects]
        self.effects = [e for e in self.effects if e.is_alive]

        # 8. Handle Pending Separations (Placeholder - needs main loop logic)
        if self.pending_separation:
             # print(f"Processing {len(self.pending_separation)} separations for rocket {self.sim_instance_id}...") # Debug
             # Actual separation logic needs to be triggered externally or return data
             # For now, just clear the pending list after marking visually
             # Keep them marked visually until cleared by external logic perhaps?
             pass # Logic moved to main sim loop ideally


    def draw(self, surface, camera):
        """ Draws all parts of the rocket and effects. """
        num_broken_visually = 0
        if not self.is_active: return 0 # Don't draw if destroyed/separated out

        for part in self.parts:
            part_center_world = self.get_world_part_center(part)
            part_screen_pos = camera.apply(part_center_world)
            part_world_angle = self.angle # Assumes part.relative_angle is 0

            # Basic culling (optional)
            # screen_rect = camera.camera_rect.copy()
            # screen_rect.topleft = (0,0) # Check against screen bounds
            # if not screen_rect.collidepoint(part_screen_pos): continue

            indicator_color = None
            activatable = part.part_data.get("activatable", False)

            if part.is_broken:
                num_broken_visually += 1
            elif activatable:
                 part_type = part.part_data.get("type")
                 if part_type == "Engine": indicator_color = COLOR_ENGINE_ENABLED if part.engine_enabled else COLOR_ENGINE_DISABLED
                 elif part_type == "Parachute": indicator_color = COLOR_ACTIVATABLE_USED if part.deployed else COLOR_ACTIVATABLE_READY
                 elif part_type == "Separator": indicator_color = COLOR_ACTIVATABLE_USED if part.separated else COLOR_ACTIVATABLE_READY # Shows blue until activated

            draw_part_shape(surface, part.part_data, part_screen_pos, part_world_angle,
                            broken=part.is_broken, deployed=part.deployed) # Pass deployed flag

            if indicator_color:
                pygame.draw.circle(surface, indicator_color, part_screen_pos, 5)


        # Draw thrust flame
        if self.thrusting:
            flame_scale = 0.5 + 0.5 * self.throttle_level
            for engine in self.engines:
                if engine.engine_enabled and not engine.is_broken:
                    engine_center_world = self.get_world_part_center(engine)
                    engine_world_angle = self.angle

                    flame_base_offset_local = pygame.math.Vector2(0, engine.part_data["height"] / 2)
                    flame_base_offset_rotated = flame_base_offset_local.rotate(-engine_world_angle)
                    flame_base_world = engine_center_world + flame_base_offset_rotated

                    flame_length = (15 + random.uniform(-2,2)) * flame_scale
                    flame_width = engine.part_data["width"] * 0.8 * flame_scale

                    flame_dir_world = pygame.math.Vector2(0, 1).rotate(-engine_world_angle)
                    flame_side_world = pygame.math.Vector2(1, 0).rotate(-engine_world_angle)

                    flame_tip_world = flame_base_world + flame_dir_world * flame_length
                    flame_left_base = flame_base_world - flame_side_world * flame_width / 2
                    flame_right_base = flame_base_world + flame_side_world * flame_width / 2

                    flame_points_screen = [camera.apply(p) for p in [flame_left_base, flame_right_base, flame_tip_world]]
                    pygame.draw.polygon(surface, COLOR_FLAME, flame_points_screen)


        # Draw effects
        [e.draw(surface, camera) for e in self.effects]

        return num_broken_visually

# --- Background/Terrain Functions (keep as is, minor tweaks maybe) ---
def create_stars(count, bounds):
    stars = []
    for _ in range(count):
        x = random.uniform(bounds.left, bounds.right)
        y = random.uniform(bounds.top, bounds.bottom)
        # Make z logarithmic/cubed to have more near stars?
        z = random.uniform(1, STAR_FIELD_DEPTH**0.5)**2 # Example adjustment
        stars.append((pygame.math.Vector2(x, y), z))
    return stars

def get_air_density(altitude_agl):
    """ Calculates simple air density based on altitude above ground level. """
    if altitude_agl < 0: # Below ground? Use sea level.
        return AIR_DENSITY_SEA_LEVEL
    if altitude_agl > ATMOSPHERE_HEIGHT:
        return AIR_DENSITY_VACUUM
    # Linear decrease for simplicity
    factor = 1.0 - (altitude_agl / ATMOSPHERE_HEIGHT)
    return AIR_DENSITY_SEA_LEVEL * factor

def draw_earth_background(surface, camera, stars):
    screen_rect = surface.get_rect()
    # Average Y position visible on screen
    avg_world_y = camera.offset.y + camera.height / 2
    ground_screen_y = camera.apply(pygame.math.Vector2(0, GROUND_Y)).y

    # Determine background color based on altitude
    if avg_world_y > BLUE_SKY_Y_LIMIT: # Close to ground / low altitude
        if ground_screen_y < screen_rect.bottom: # Ground visible
            # Draw sky
             sky_rect = pygame.Rect(0, 0, screen_rect.width, ground_screen_y)
             pygame.draw.rect(surface, COLOR_SKY_BLUE, sky_rect)
             # Draw horizon gradient (optional, simple rect for now)
             # horizon_rect = pygame.Rect(0, ground_screen_y - 10, screen_rect.width, 10)
             # pygame.draw.rect(surface, COLOR_HORIZON, horizon_rect)
        else: # Ground below screen bottom
            surface.fill(COLOR_SKY_BLUE)

    elif avg_world_y < SPACE_Y_LIMIT: # High altitude / space
        surface.fill(COLOR_SPACE_BLACK)
        draw_stars(surface, stars, camera, alpha=255)

    else: # Transition altitude
        # Interpolate between sky blue and space black
        interp_factor = max(0.0, min(1.0, (avg_world_y - BLUE_SKY_Y_LIMIT) / (SPACE_Y_LIMIT - BLUE_SKY_Y_LIMIT)))
        bg_color = COLOR_SKY_BLUE.lerp(COLOR_SPACE_BLACK, interp_factor)
        surface.fill(bg_color)
        # Fade in stars
        star_alpha = int(255 * interp_factor)
        if star_alpha > 10:
            draw_stars(surface, stars, camera, alpha=star_alpha)

def draw_stars(surface, stars, camera, alpha=255):
    if alpha <= 0: return
    screen_center_world = camera.offset + pygame.math.Vector2(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
    base_star_color = pygame.Color(200, 200, 200) # Slightly less bright gray

    for star_world_pos, z in stars:
        # Parallax effect: closer stars (smaller z) move less relative to center
        parallax_factor = 1.0 / (z / 100 + 1) # Adjust divisor for stronger/weaker effect
        star_view_pos = screen_center_world + (star_world_pos - screen_center_world) * parallax_factor
        star_screen_pos = camera.apply(star_view_pos)

        # Culling
        if 0 <= star_screen_pos.x < SCREEN_WIDTH and 0 <= star_screen_pos.y < SCREEN_HEIGHT:
            # Size based on depth (closer stars are slightly larger)
            size = max(1, int(2.5 * (1.0 - z / STAR_FIELD_DEPTH)))
            # Final color with alpha
            alpha_factor = alpha / 255.0
            final_color = (int(base_star_color.r * alpha_factor),
                           int(base_star_color.g * alpha_factor),
                           int(base_star_color.b * alpha_factor))

            if final_color != (0,0,0): # Avoid drawing black stars
                 pygame.draw.circle(surface, final_color, (int(star_screen_pos.x), int(star_screen_pos.y)), size)

def draw_terrain(surface, camera):
    # Draw a simple ground rectangle extending horizontally
    ground_view_rect = pygame.Rect(camera.offset.x - WORLD_WIDTH, # Extend far left/right
                                   GROUND_Y,
                                   camera.width + WORLD_WIDTH * 2,
                                   SCREEN_HEIGHT * 2) # Make it tall enough
    ground_rect_screen = camera.apply_rect(ground_view_rect)
    pygame.draw.rect(surface, COLOR_GROUND, ground_rect_screen)

# --- Simulation Runner Function (REVISED STRUCTURE) ---
def run_simulation(screen, clock, blueprint_file):
    print(f"--- Starting Simulation ---")
    print(f"Loading blueprint: {blueprint_file}")
    if not os.path.exists(blueprint_file):
        print(f"Error: Blueprint file not found at {blueprint_file}"); return

    initial_blueprint = RocketBlueprint.load_from_json(blueprint_file)
    if not initial_blueprint or not initial_blueprint.parts:
        print("Blueprint load failed or blueprint is empty."); return

    # --- Initial Simulation State Setup ---
    all_rockets: list[FlyingRocket] = []
    controlled_rocket: FlyingRocket | None = None
    next_sim_id = 0

    # --- *** REVISED ROOT PART IDENTIFICATION *** ---
    original_root_part_instance = None
    if initial_blueprint.parts:
        # Try finding the first CommandPod in the entire list
        for part in initial_blueprint.parts:
            # Ensure part_data is accessible; might need get_part_data if not stored directly
            # Assuming part.part_data exists from PlacedPart init
            if part.part_data and part.part_data.get("type") == "CommandPod":
                original_root_part_instance = part
                print(f"DEBUG: Identified CommandPod '{part.part_id}' as the potential root reference.")
                break # Found the first one, stop searching
        if original_root_part_instance is None:
             print("Warning: No CommandPod found in blueprint. Control will be assigned arbitrarily if possible.")
             # No specific root reference exists
    # --- *** END REVISED ROOT IDENTIFICATION *** ---


    # --- Create Initial Rocket Instances from Blueprint ---
    initial_subassemblies = initial_blueprint.find_connected_subassemblies()
    if not initial_subassemblies:
        print("Error: No parts found after connectivity check."); return

    for i, assembly_parts in enumerate(initial_subassemblies):
        if not assembly_parts: continue
        # Create temporary blueprint to get dimensions for positioning
        temp_bp = RocketBlueprint(); temp_bp.parts = assembly_parts
        lowest_offset_y = temp_bp.get_lowest_point_offset_y()
        start_x = i * 50 # Simple horizontal offset for multiple initial assemblies
        start_y = GROUND_Y - lowest_offset_y - 0.1 # Start just above ground
        target_initial_com_pos = pygame.math.Vector2(start_x, start_y) # Target CoM position

        # Check if this assembly contains the identified root part instance
        contains_original_root = original_root_part_instance and (original_root_part_instance in assembly_parts)

        # Assign control if this is the first assembly AND it contains the root
        # Or fallback to first assembly if no root pod exists at all
        is_primary = (controlled_rocket is None and contains_original_root) or \
                     (controlled_rocket is None and original_root_part_instance is None and i == 0)

        try:
            rocket_instance = FlyingRocket(
                parts_list=list(assembly_parts), # Pass a copy of the list
                initial_pos_offset=target_initial_com_pos, # Target CoM world pos
                initial_angle=0,
                initial_vel=pygame.math.Vector2(0,0),
                sim_instance_id=next_sim_id,
                is_primary_control=is_primary, # Initial hint, might be overridden below
                # Pass the found root instance (or None)
                original_root_ref=original_root_part_instance
            )
            all_rockets.append(rocket_instance)

            if is_primary:
                 controlled_rocket = rocket_instance
                 # Explicitly set the actual control flag based on whether root exists *in this instance*
                 # This uses the check logic now inside FlyingRocket.update
                 root_ref_in_instance = controlled_rocket.original_root_part_ref
                 controlled_rocket.has_active_control = (root_ref_in_instance is not None) and \
                                                        (root_ref_in_instance in controlled_rocket.parts) and \
                                                        (not root_ref_in_instance.is_broken) # Check broken state too
                 control_status_msg = 'CONTROLLED' if controlled_rocket.has_active_control else 'CONTROLLED (No Pod/Broken)'
                 print(f"Created initial rocket {next_sim_id} ({control_status_msg}) with {len(assembly_parts)} parts.")

            else:
                 # Ensure non-primary rockets start with has_active_control as False
                 rocket_instance.has_active_control = False
                 print(f"Created initial rocket {next_sim_id} (DEBRIS) with {len(assembly_parts)} parts.")
            next_sim_id += 1
        except Exception as e:
            print(f"Error initializing rocket instance {next_sim_id}: {e}")


    # Safety check if control assignment somehow failed but rockets exist
    if controlled_rocket is None and all_rockets:
         print("Warning: Fallback control assignment to first assembly.")
         controlled_rocket = all_rockets[0]
         # Grant active control only if it actually has the valid root pod ref
         root_ref_in_fallback = controlled_rocket.original_root_part_ref
         controlled_rocket.has_active_control = (root_ref_in_fallback is not None) and \
                                                (root_ref_in_fallback in controlled_rocket.parts) and \
                                                (not root_ref_in_fallback.is_broken)


    # --- Setup Camera, Stars, UI ---
    camera = Camera(SCREEN_WIDTH, SCREEN_HEIGHT)
    if controlled_rocket: camera.update(controlled_rocket.get_world_com())
    elif all_rockets: camera.update(all_rockets[0].get_world_com())
    else: camera.update(pygame.math.Vector2(0, GROUND_Y))

    # Ensure these helper functions exist or are defined above
    try:
        star_area_bounds = pygame.Rect(-WORLD_WIDTH * 2, SPACE_Y_LIMIT - STAR_FIELD_DEPTH, WORLD_WIDTH * 4, abs(SPACE_Y_LIMIT) + GROUND_Y + STAR_FIELD_DEPTH*1.5)
        stars = create_stars(STAR_COUNT * 2, star_area_bounds)
    except NameError:
        print("Warning: create_stars or related constants not found, skipping star creation.")
        stars = []

    ui_font = pygame.font.SysFont(None, 20)
    ui_font_large = pygame.font.SysFont(None, 36)

    # --- Main Simulation Loop ---
    sim_running = True
    last_respawn_time = time.time()

    while sim_running:
        dt = clock.tick(60) / 1000.0
        dt = min(dt, 0.05) # Clamp max dt

        # --- Frame State Initialization ---
        newly_created_rockets_this_frame: list[FlyingRocket] = []
        rockets_to_remove_this_frame: list[FlyingRocket] = []

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                # DEBUG PRINT BEFORE INPUT
                # if controlled_rocket: print(f"DEBUG: KeyDown Check - Controlled: {controlled_rocket.sim_instance_id}, ActiveCtrl: {controlled_rocket.has_active_control}")
                # else: print("DEBUG: KeyDown Check - No Controlled Rocket!")

                if event.key == pygame.K_ESCAPE: sim_running = False
                # Apply inputs ONLY to the controlled rocket
                if controlled_rocket and controlled_rocket.has_active_control: # Also check has_active_control here
                    if event.key == pygame.K_SPACE:
                        controlled_rocket.master_thrust_enabled = not controlled_rocket.master_thrust_enabled
                        print(f"[{controlled_rocket.sim_instance_id}] Master Thrust set to: {controlled_rocket.master_thrust_enabled}")
                    if event.key == pygame.K_p:
                         for chute in controlled_rocket.parachutes:
                             if not chute.deployed and not chute.is_broken:
                                 chute.deployed = True; print(f"Deployed {chute.part_id} via key.")

                # --- Respawn Logic ---
                current_time = time.time()
                if event.key == pygame.K_r and (current_time - last_respawn_time > 1.0):
                    print("--- RESPAWNING ROCKET ---"); last_respawn_time = current_time
                    all_rockets.clear(); controlled_rocket = None; newly_created_rockets_this_frame.clear(); rockets_to_remove_this_frame.clear()
                    reloaded_blueprint = RocketBlueprint.load_from_json(blueprint_file)
                    if reloaded_blueprint and reloaded_blueprint.parts:
                        # --- Repeat root finding on reload ---
                        original_root_part_instance = None
                        for part in reloaded_blueprint.parts:
                            if part.part_data and part.part_data.get("type") == "CommandPod":
                                original_root_part_instance = part; break
                        # --- Repeat initial rocket creation loop ---
                        initial_subassemblies = reloaded_blueprint.find_connected_subassemblies()
                        for i, assembly_parts in enumerate(initial_subassemblies):
                            if not assembly_parts: continue
                            temp_bp = RocketBlueprint(); temp_bp.parts = assembly_parts
                            lowest_offset_y = temp_bp.get_lowest_point_offset_y(); start_x = i * 50; start_y = GROUND_Y - lowest_offset_y - 0.1
                            target_initial_com_pos = pygame.math.Vector2(start_x, start_y)
                            contains_original_root = original_root_part_instance and (original_root_part_instance in assembly_parts)
                            is_primary = (controlled_rocket is None and contains_original_root) or \
                                         (controlled_rocket is None and original_root_part_instance is None and i == 0)
                            try:
                                rocket_instance = FlyingRocket(list(assembly_parts), target_initial_com_pos, 0, pygame.math.Vector2(0,0), next_sim_id, is_primary, original_root_part_instance)
                                newly_created_rockets_this_frame.append(rocket_instance)
                                if is_primary:
                                     controlled_rocket = rocket_instance
                                     root_ref_in_instance = controlled_rocket.original_root_part_ref
                                     controlled_rocket.has_active_control = (root_ref_in_instance is not None) and (root_ref_in_instance in controlled_rocket.parts) and (not root_ref_in_instance.is_broken)
                                else:
                                     rocket_instance.has_active_control = False
                                next_sim_id += 1
                            except Exception as e: print(f"Respawn Error creating instance: {e}")
                        if controlled_rocket is None and newly_created_rockets_this_frame:
                             controlled_rocket = newly_created_rockets_this_frame[0]
                             root_ref_in_fallback = controlled_rocket.original_root_part_ref
                             controlled_rocket.has_active_control = (root_ref_in_fallback is not None) and (root_ref_in_fallback in controlled_rocket.parts) and (not root_ref_in_fallback.is_broken)
                        print("Respawn Complete.")
                    else: print("Respawn Failed: Cannot reload blueprint.")
                    # Immediately process adds/removes after respawn
                    all_rockets.extend(newly_created_rockets_this_frame); newly_created_rockets_this_frame.clear()


            # Mouse Click Activation (only on controlled rocket)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                 if controlled_rocket: # Check if controlled_rocket exists
                     click_screen_pos = pygame.math.Vector2(event.pos); click_world_pos = click_screen_pos + camera.offset
                     controlled_rocket.activate_part_at_pos(click_world_pos)

        # --- Continuous Input (Throttle for controlled rocket) ---
        # Check if controlled and if it actually has control ability
        if controlled_rocket and controlled_rocket.has_active_control:
            keys = pygame.key.get_pressed()
            throttle_change = 0
            if keys[pygame.K_w] or keys[pygame.K_UP]: throttle_change += THROTTLE_CHANGE_RATE * dt
            if keys[pygame.K_s] or keys[pygame.K_DOWN]: throttle_change -= THROTTLE_CHANGE_RATE * dt
            if throttle_change != 0:
                controlled_rocket.throttle_level = max(0.0, min(1.0, controlled_rocket.throttle_level + throttle_change))

        # --- Stage 1: Update Physics for all active rockets ---
        for rocket in all_rockets:
            if not rocket.is_active: continue
            rocket_com_alt = GROUND_Y - rocket.get_world_com().y
            # Ensure get_air_density exists or provide default
            try: current_air_density = get_air_density(rocket_com_alt)
            except NameError: current_air_density = 0.0
            rocket.update(dt, current_air_density)
            # Check if rocket became inactive AFTER update
            if not rocket.is_active and rocket not in rockets_to_remove_this_frame:
                 rockets_to_remove_this_frame.append(rocket)


        # --- Stage 1.5: Inter-Rocket Collision Detection & Response ---
        collision_pairs_processed = set()
        for i, r1 in enumerate(all_rockets):
            if r1 in rockets_to_remove_this_frame or not r1.is_active: continue
            for j in range(i + 1, len(all_rockets)):
                r2 = all_rockets[j]
                if r2 in rockets_to_remove_this_frame or not r2.is_active: continue
                # Optional Broad Phase Check (keep if implemented)
                # ...
                # Narrow Phase
                collision_found_between_r1_r2 = False
                for p1 in r1.parts:
                    if p1.is_broken: continue
                    rect1 = r1.get_world_part_aabb(p1)
                    for p2 in r2.parts:
                        if p2.is_broken: continue
                        rect2 = r2.get_world_part_aabb(p2)
                        if rect1.colliderect(rect2):
                            pair_key = tuple(sorted((r1.sim_instance_id, r2.sim_instance_id)))
                            if pair_key in collision_pairs_processed: continue
                            print(f"COLLISION: R{r1.sim_instance_id}({p1.part_id}) & R{r2.sim_instance_id}({p2.part_id})")
                            collision_found_between_r1_r2 = True
                            collision_pairs_processed.add(pair_key)
                            # Collision Response (Damage + Push)
                            relative_velocity = r1.vel - r2.vel; impact_speed = relative_velocity.length()
                            r1.apply_collision_damage(impact_speed, specific_part_to_damage=p1)
                            r2.apply_collision_damage(impact_speed, specific_part_to_damage=p2)
                            # Basic Push Apart (keep if implemented)
                            # ... (push logic) ...
                            break # Stop checking r2 parts against p1
                    if collision_found_between_r1_r2: break # Stop checking r1 parts


        # --- Stage 2: Process State Changes (Destruction Splits, Separations) ---
        rockets_to_process = list(all_rockets) # Iterate over copy
        for rocket in rockets_to_process:
            if rocket in rockets_to_remove_this_frame or not rocket.is_active: continue
            processed_split_this_frame = False
            # --- Check for destruction splits ---
            if rocket.needs_connectivity_check:
                rocket.needs_connectivity_check = False; print(f"[{rocket.sim_instance_id}] Re-checking connectivity (Destruction)...")
                temp_bp = RocketBlueprint(); temp_bp.parts = rocket.parts
                subassemblies = temp_bp.find_connected_subassemblies()
                if len(subassemblies) > 1:
                    processed_split_this_frame = True; print(f"[{rocket.sim_instance_id}] SPLIT DETECTED (Destruction) into {len(subassemblies)} pieces!")
                    if rocket not in rockets_to_remove_this_frame: rockets_to_remove_this_frame.append(rocket)
                    for assembly_parts in subassemblies:
                         if not assembly_parts: continue
                         try:
                             sub_world_com = rocket.calculate_subassembly_world_com(assembly_parts)
                             contains_root = rocket.original_root_part_ref in assembly_parts
                             is_primary = rocket.has_active_control and contains_root
                             new_rocket = FlyingRocket(list(assembly_parts), sub_world_com, rocket.angle, rocket.vel, next_sim_id, is_primary, rocket.original_root_part_ref)
                             new_rocket.angular_velocity = rocket.angular_velocity
                             newly_created_rockets_this_frame.append(new_rocket)
                             print(f"  > New {'controlled' if is_primary else 'debris'} rocket {next_sim_id} (destruction split).")
                             next_sim_id += 1
                         except Exception as e: print(f"Error creating rocket from destruction split: {e}")

            # --- Check for separator splits ---
            if rocket.pending_separation and not processed_split_this_frame:
                 print(f"[{rocket.sim_instance_id}] Processing {len(rocket.pending_separation)} separations...")
                 separators_activated = list(rocket.pending_separation); rocket.pending_separation.clear()
                 parts_at_start = list(rocket.parts); parts_remaining = list(rocket.parts)
                 split_occurred_by_sep = False
                 for sep_part in separators_activated:
                     if sep_part not in parts_remaining: continue
                     print(f"  Processing separator: {sep_part.part_id}")
                     sep_world_pos = rocket.get_world_part_center(sep_part); sep_force = sep_part.part_data.get("separation_force", 1000)
                     current_check_list = [p for p in parts_remaining if p != sep_part]
                     temp_bp = RocketBlueprint(); temp_bp.parts = current_check_list
                     subassemblies = temp_bp.find_connected_subassemblies()
                     if len(subassemblies) > 1:
                         split_occurred_by_sep = True; print(f"  > SPLIT CONFIRMED by {sep_part.part_id} into {len(subassemblies)} pieces!")
                         if rocket not in rockets_to_remove_this_frame: rockets_to_remove_this_frame.append(rocket)
                         for assembly_parts in subassemblies:
                              if not assembly_parts: continue
                              try:
                                  sub_world_com = rocket.calculate_subassembly_world_com(assembly_parts)
                                  contains_root = rocket.original_root_part_ref in assembly_parts
                                  is_primary = rocket.has_active_control and contains_root
                                  new_rocket = FlyingRocket(list(assembly_parts), sub_world_com, rocket.angle, rocket.vel, next_sim_id, is_primary, rocket.original_root_part_ref)
                                  new_rocket.angular_velocity = rocket.angular_velocity
                                  # Apply Impulse
                                  sep_vec = new_rocket.get_world_com() - sep_world_pos
                                  sep_dir = sep_vec.normalize() if sep_vec.length() > 0 else pygame.math.Vector2(0,-1).rotate(-rocket.angle)
                                  impulse_mag = (sep_force / max(0.1, new_rocket.total_mass)) * 0.05 # Tune multiplier
                                  delta_v = sep_dir * impulse_mag; new_rocket.vel += delta_v
                                  print(f"    Applying impulse {delta_v.length():.1f} to new rocket {next_sim_id}")
                                  newly_created_rockets_this_frame.append(new_rocket)
                                  print(f"    > New {'controlled' if is_primary else 'debris'} rocket {next_sim_id} (separation).")
                                  next_sim_id += 1
                              except Exception as e: print(f"Error creating rocket from separation split: {e}")
                         break # Stop processing separators for this original rocket as it's being replaced
                     else:
                         print(f"  > Separator {sep_part.part_id} did not cause a split.")
                         parts_remaining = current_check_list # Update remaining list for next sep check
                 if not split_occurred_by_sep and len(parts_remaining) < len(parts_at_start):
                      print(f"[{rocket.sim_instance_id}] Updating parts after non-splitting separation(s).")
                      rocket.parts = parts_remaining
                      # Update component lists & physics
                      rocket.engines = [e for e in rocket.engines if e in rocket.parts]; rocket.fuel_tanks = [t for t in rocket.fuel_tanks if t in rocket.parts]; rocket.parachutes = [pc for pc in rocket.parachutes if pc in rocket.parts]; rocket.separators = [s for s in rocket.separators if s in rocket.parts]
                      if not rocket.parts: rocket.is_active = False; rockets_to_remove_this_frame.append(rocket)
                      else: rocket.calculate_physics_properties(); rocket.calculate_bounds()


        # --- Stage 3: Apply Additions and Removals ---
        if newly_created_rockets_this_frame:
            new_controlled_rocket_found = None
            for nr in newly_created_rockets_this_frame:
                 all_rockets.append(nr)
                 if nr.has_active_control:
                      if controlled_rocket and controlled_rocket not in rockets_to_remove_this_frame: print(f"Warning: Multiple control claim! New: {nr.sim_instance_id}, Old: {controlled_rocket.sim_instance_id}"); controlled_rocket.has_active_control = False
                      new_controlled_rocket_found = nr
            if new_controlled_rocket_found: controlled_rocket = new_controlled_rocket_found

        if rockets_to_remove_this_frame:
            was_controlled_rocket_removed = controlled_rocket in rockets_to_remove_this_frame
            all_rockets = [r for r in all_rockets if r not in rockets_to_remove_this_frame]
            if was_controlled_rocket_removed:
                print("Controlled rocket instance removed/replaced."); controlled_rocket = None
                for rkt in all_rockets: # Find new control
                    if rkt.has_active_control: controlled_rocket = rkt; break # Already marked?
                if not controlled_rocket: # If not marked, find one with root
                    for rkt in all_rockets:
                        if rkt.original_root_part_ref and rkt.original_root_part_ref in rkt.parts and not rkt.original_root_part_ref.is_broken:
                             controlled_rocket = rkt; controlled_rocket.has_active_control = True; break
                if controlled_rocket: print(f"Control transferred to rocket instance {controlled_rocket.sim_instance_id}.")
                else: print("No controllable rocket found after removals.")


        # --- Stage 4: Update Camera ---
        if controlled_rocket: camera.update(controlled_rocket.get_world_com())
        elif all_rockets: camera.update(all_rockets[0].get_world_com())


        # --- Stage 5: Drawing ---
        screen.fill(BLACK) # Use background function ideally
        try: draw_earth_background(screen, camera, stars)
        except NameError: pass
        try: draw_terrain(screen, camera)
        except NameError: pass

        total_parts_drawn = 0; total_broken_drawn = 0
        for rocket in all_rockets:
            if rocket.is_active:
                 broken_count = rocket.draw(screen, camera)
                 total_parts_drawn += len(rocket.parts); total_broken_drawn += broken_count

        # --- UI Overlay ---
        if controlled_rocket:
            # Draw UI for controlled rocket (Throttle, Status Text)
            bar_width=20; bar_height=100; bar_x=15; bar_y=SCREEN_HEIGHT-bar_height-40
            pygame.draw.rect(screen, (50,50,50), (bar_x, bar_y, bar_width, bar_height))
            fill_height=bar_height*controlled_rocket.throttle_level
            pygame.draw.rect(screen, (0,200,0), (bar_x, bar_y + bar_height - fill_height, bar_width, fill_height))
            pygame.draw.rect(screen, WHITE, (bar_x, bar_y, bar_width, bar_height), 1)
            th_txt=ui_font.render("Thr", True, WHITE); screen.blit(th_txt, (bar_x, bar_y + bar_height + 5))
            th_val_txt = ui_font.render(f"{controlled_rocket.throttle_level * 100:.0f}%", True, WHITE); screen.blit(th_val_txt, (bar_x, bar_y - 18))

            alt_agl=max(0,GROUND_Y-controlled_rocket.get_lowest_point_world().y); alt_msl=GROUND_Y-controlled_rocket.get_world_com().y
            cs="OK" if controlled_rocket.has_active_control else "NO CONTROL"
            mts="ON" if controlled_rocket.master_thrust_enabled else "OFF"; ls="LANDED" if controlled_rocket.landed else "FLYING"
            st=[f"Alt(AGL): {alt_agl:.1f}m", f"Alt(MSL): {alt_msl:.1f}m", f"Vvel: {controlled_rocket.vel.y:.1f}", f"Hvel: {controlled_rocket.vel.x:.1f}",
                f"Speed: {controlled_rocket.vel.length():.1f}", f"Angle: {controlled_rocket.angle:.1f}", f"AngVel: {controlled_rocket.angular_velocity:.1f}",
                f"Thr: {controlled_rocket.throttle_level*100:.0f}% [{mts}]", f"Fuel: {controlled_rocket.current_fuel:.1f}", f"Mass: {controlled_rocket.total_mass:.1f}kg",
                f"Ctrl: {cs}", f"Status: {ls}", f"Inst: {controlled_rocket.sim_instance_id}"]
            t_y=10; tc=WHITE if controlled_rocket.has_active_control else RED
            for i,t in enumerate(st): ts=ui_font.render(t,True,tc); screen.blit(ts,(bar_x+bar_width+10,t_y+i*18))
        elif not all_rockets:
            dt_txt=ui_font_large.render("ALL ROCKETS DESTROYED",True,RED); tr=dt_txt.get_rect(center=(SCREEN_WIDTH//2,SCREEN_HEIGHT//2)); screen.blit(dt_txt,tr)
            rt_txt=ui_font.render("Press 'R' to Respawn", True, WHITE); rtr=rt_txt.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2+40)); screen.blit(rt_txt, rtr)

        # FPS Counter
        fps = clock.get_fps(); fps_text = ui_font.render(f"FPS: {fps:.1f}", True, WHITE); screen.blit(fps_text, (SCREEN_WIDTH - 80, 10))
        obj_text = ui_font.render(f"Objs: {len(all_rockets)}", True, WHITE); screen.blit(obj_text, (SCREEN_WIDTH - 80, 30))

        pygame.display.flip()

    print("--- Exiting Simulation ---")