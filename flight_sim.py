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
from ui_elements import SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, BLACK, GRAY, RED, GREEN, BLUE, LIGHT_GRAY # Added LIGHT_GRAY

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
        self.particles=[]; self.pos=pygame.math.Vector2(pos); self.max_life=max_life
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
            rl = p[2] # remaining life
            # *** FIX: Only calculate and draw if particle is alive ***
            if rl > 0:
                sp = camera.apply(p[0]) # screen position
                ml = p[3] # max life
                # Calculate alpha factor safely
                af = max(0, rl / ml) if ml > 0 else 0
                bc = p[4] # base color
                br = p[5] # base radius
                # Calculate derived color and radius
                dc = (int(bc.r * af), int(bc.g * af), int(bc.b * af))
                dr = int(br * af)

                # *** FIX: Check derived color/radius *inside* the rl > 0 block ***
                if dc != (0, 0, 0) and dr > 0:
                    pygame.draw.circle(surface, dc, sp, dr)
            # No else needed, if rl <= 0, nothing is drawn for this particle

# --- Camera Class (keep as is) ---
class Camera:
    def __init__(self, width, height): self.camera_rect=pygame.Rect(0,0,width,height); self.width=width; self.height=height; self.offset=pygame.math.Vector2(0,0)
    def apply(self, target_pos): return target_pos-self.offset
    def apply_rect(self, target_rect): return target_rect.move(-self.offset.x,-self.offset.y)
    def update(self, target_pos): x=target_pos.x-self.width//2; y=target_pos.y-self.height//2; self.offset=pygame.math.Vector2(x,y)

# --- FlyingRocket Class ---
class FlyingRocket:
    def __init__(self, blueprint: RocketBlueprint, initial_pos, initial_angle=0, initial_vel=pygame.math.Vector2(0,0), sim_instance_id=0):
        self.sim_instance_id = sim_instance_id # To differentiate between separated rockets
        self.blueprint_name = blueprint.name
        # IMPORTANT: Create deep copies of PlacedPart data for flight state
        self.parts = [PlacedPart.from_dict(p.to_dict()) for p in blueprint.parts]
        if not self.parts: raise ValueError("Cannot initialize FlyingRocket with no parts.")

        self.original_root_part_obj = self.parts[0] # Keep track of the original command pod
        self.has_active_control = True # Assume control initially if root exists

        # Physics state
        self.pos = pygame.math.Vector2(initial_pos)
        self.vel = pygame.math.Vector2(initial_vel)
        self.acc = pygame.math.Vector2(0,0)
        self.angle = initial_angle # Degrees, counter-clockwise from vertical up (0 = up)
        self.angular_velocity = 0.0 # Degrees per second

        # Control state
        self.throttle_level = 0.0
        self.master_thrust_enabled = False

        # Resources and Components
        self.engines = []
        self.fuel_tanks = []
        self.parachutes = []
        self.separators = []
        total_fuel_cap = 0

        for i, part in enumerate(self.parts):
            # Initialize flight state on the copied part object
            part.current_hp = part.part_data.get("max_hp", 100)
            part.is_broken = False
            part.engine_enabled = True
            part.deployed = False # Reset chute state on init
            part.separated = False # Reset separator state on init
            part.part_index = i # Give each part an index within this rocket instance

            # Categorize parts
            pt = part.part_data.get("type")
            if pt == "Engine": self.engines.append(part)
            elif pt == "FuelTank": self.fuel_tanks.append(part)
            elif pt == "Parachute": self.parachutes.append(part)
            elif pt == "Separator": self.separators.append(part)

            total_fuel_cap += part.part_data.get("fuel_capacity", 0)

        self.current_fuel = total_fuel_cap
        self.fuel_mass_per_unit = 0.1 # kg per unit of fuel (adjust)

        # Physics properties (recalculated when mass changes)
        self.total_mass = 0
        self.dry_mass = 0
        self.moment_of_inertia = 10000 # Default placeholder
        self.center_of_mass_offset = pygame.math.Vector2(0, 0) # Relative to blueprint origin (0,0)
        self.local_bounds = pygame.Rect(0,0,1,1) # Placeholder

        # Recalculate physics properties based on initial state
        self.calculate_physics_properties()
        self.calculate_bounds() # Calculate visual bounds

        # Effects
        self.effects = [] # For explosions, etc.

        # State flags
        self.landed = False
        self.thrusting = False
        self.is_active = True # Flag if this rocket object is still part of the simulation
        self.pending_separation = [] # Store separators activated this frame


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


    def apply_collision_damage(self, impact_velocity_magnitude):
        """ Applies damage to parts near the lowest point based on impact velocity. """
        # This method assumes it's called only when damage conditions are met (checked in update)
        if impact_velocity_magnitude < MIN_IMPACT_VEL_DAMAGE: # Redundant check, but safe
            return

        damage = (impact_velocity_magnitude ** 1.8) * COLLISION_DAMAGE_FACTOR
        print(f"Impact Vel: {impact_velocity_magnitude:.1f} -> Damage: {damage:.1f}")

        lowest_point_world = self.get_lowest_point_world()
        impacted_parts = self.get_parts_near_world_pos(lowest_point_world, radius=15.0)

        if not impacted_parts and self.parts:
             lowest_part = min(self.parts, key=lambda p: self.get_world_part_center(p).y)
             if lowest_part: impacted_parts = [lowest_part]

        parts_destroyed_this_impact = []
        for part in impacted_parts:
            if part.is_broken: continue
            part.current_hp -= damage
            print(f"  Part '{part.part_id}' HP: {part.current_hp:.0f} / {part.part_data.get('max_hp', 1)}")
            if part.current_hp <= 0:
                print(f"  >> Part '{part.part_id}' BROKEN! <<")
                part.is_broken = True
                part.current_hp = 0
                parts_destroyed_this_impact.append(part)
                part_center_world = self.get_world_part_center(part)
                self.effects.append(ExplosionEffect(part_center_world)) # Ensure ExplosionEffect class exists

        if parts_destroyed_this_impact:
            self.handle_destroyed_parts(parts_destroyed_this_impact)

    def handle_destroyed_parts(self, destroyed_parts: list[PlacedPart]):
        """ Removes destroyed parts and checks for structural separation (basic implementation). """
        if not destroyed_parts: return

        print(f"Handling destruction of: {[p.part_id for p in destroyed_parts]}")
        self.parts = [p for p in self.parts if p not in destroyed_parts]
        self.engines = [e for e in self.engines if e not in destroyed_parts]
        self.fuel_tanks = [t for t in self.fuel_tanks if t not in destroyed_parts]
        self.parachutes = [pc for pc in self.parachutes if pc not in destroyed_parts]
        self.separators = [s for s in self.separators if s not in destroyed_parts]

        if self.original_root_part_obj in destroyed_parts:
            print("Root part destroyed! Losing primary control.")
            self.has_active_control = False

        if not self.parts:
            print("All parts destroyed!")
            self.is_active = False
            return

        # --- Basic Connectivity Check Placeholder ---
        # TODO: Implement proper graph traversal for separation detection
        connected_parts = set(self.parts) # Simplification: Assume remaining parts are connected
        separated_parts = [p for p in self.parts if p not in connected_parts]
        if separated_parts:
             print(f"Structural failure! Separated parts: {[p.part_id for p in separated_parts]}")
             # This should trigger new FlyingRocket creation in the main loop
             self.parts = list(connected_parts) # Keep only connected parts in this instance

        self.calculate_physics_properties()
        self.calculate_bounds()

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
        self.has_active_control = self.original_root_part_obj in self.parts and not self.original_root_part_obj.is_broken
        control_torque_input = 0
        if self.has_active_control:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]: control_torque_input += REACTION_WHEEL_TORQUE
            if keys[pygame.K_LEFT] or keys[pygame.K_a]: control_torque_input -= REACTION_WHEEL_TORQUE
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
                self.apply_collision_damage(impact_vel_mag)

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

# --- Simulation Runner Function ---
def run_simulation(screen, clock, blueprint_file):
    print(f"--- Starting Simulation ---")
    print(f"Loading blueprint: {blueprint_file}")
    if not os.path.exists(blueprint_file):
        print(f"Error: Blueprint file not found at {blueprint_file}")
        return # Exit if file doesn't exist

    initial_blueprint = RocketBlueprint.load_from_json(blueprint_file)
    if not initial_blueprint or not initial_blueprint.parts:
        print("Blueprint load failed or blueprint is empty.")
        return # Exit if load failed or no parts

    # Calculate initial position based on lowest point
    lowest_offset_y = initial_blueprint.get_lowest_point_offset_y()
    start_x = 0 # Start at world origin X
    start_y = GROUND_Y - lowest_offset_y - 0.1 # Place lowest point on the ground

    initial_pos = pygame.math.Vector2(start_x, start_y)
    print(f"Blueprint: {initial_blueprint.name}")
    print(f"Lowest point offset relative to root center: {lowest_offset_y:.2f}")
    print(f"Calculated Start Pos: ({initial_pos.x:.2f}, {initial_pos.y:.2f}) (Ground Y: {GROUND_Y})")

    # --- Manage Multiple Rockets ---
    all_rockets = []
    next_sim_id = 0

    try:
        player_rocket = FlyingRocket(initial_blueprint, initial_pos, sim_instance_id=next_sim_id)
        all_rockets.append(player_rocket)
        controlled_rocket = player_rocket # Initially control the first rocket
        next_sim_id += 1
        print(f"Initial rocket created. Mass: {player_rocket.total_mass:.2f} kg")
    except Exception as e:
        print(f"Fatal Error initializing rocket: {e}")
        return # Exit if creation fails

    # --- Setup Camera, Stars, UI ---
    camera = Camera(SCREEN_WIDTH, SCREEN_HEIGHT)
    # Initial camera focus on the controlled rocket
    if controlled_rocket:
         camera.update(controlled_rocket.get_world_com())
    else: # Should not happen here, but safety check
         camera.update(initial_pos)


    # Define star field bounds relative to world origin
    star_area_bounds = pygame.Rect(-WORLD_WIDTH * 2, SPACE_Y_LIMIT - STAR_FIELD_DEPTH,
                                    WORLD_WIDTH * 4, abs(SPACE_Y_LIMIT) + GROUND_Y + STAR_FIELD_DEPTH*1.5)
    stars = create_stars(STAR_COUNT * 2, star_area_bounds)
    ui_font = pygame.font.SysFont(None, 20) # Smaller font for more info
    ui_font_large = pygame.font.SysFont(None, 36)

    sim_running = True
    last_respawn_time = time.time()

    while sim_running:
        dt = clock.tick(60) / 1000.0
        dt = min(dt, 0.05) # Clamp max dt to prevent physics explosions

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    sim_running = False
                if controlled_rocket: # Only handle input if we have a controlled rocket
                    if event.key == pygame.K_SPACE:
                        controlled_rocket.master_thrust_enabled = not controlled_rocket.master_thrust_enabled
                        print(f"Master Thrust: {'ON' if controlled_rocket.master_thrust_enabled else 'OFF'}")
                    if event.key == pygame.K_p: # Deploy Parachutes Key (example)
                         for chute in controlled_rocket.parachutes:
                             if not chute.deployed and not chute.is_broken:
                                 chute.deployed = True
                                 print(f"Deployed parachute {chute.part_id} via key.")
                         # TODO: Add sound effect

                # Respawn Logic
                current_time = time.time()
                if event.key == pygame.K_r and (current_time - last_respawn_time > 2.0): # Add debounce timer
                    print("--- RESPAWNING ROCKET ---")
                    last_respawn_time = current_time
                    # Clear existing rockets
                    all_rockets.clear()
                    try:
                        # Reload blueprint in case it changed (though not expected here)
                        reloaded_blueprint = RocketBlueprint.load_from_json(blueprint_file)
                        if not reloaded_blueprint or not reloaded_blueprint.parts:
                            print("Respawn failed: Cannot reload blueprint.")
                            sim_running = False; break

                        lowest_offset_y = reloaded_blueprint.get_lowest_point_offset_y()
                        start_y = GROUND_Y - lowest_offset_y
                        initial_pos = pygame.math.Vector2(start_x, start_y)

                        player_rocket = FlyingRocket(reloaded_blueprint, initial_pos, sim_instance_id=next_sim_id)
                        all_rockets.append(player_rocket)
                        controlled_rocket = player_rocket
                        next_sim_id += 1
                        print("Rocket Respawned.")
                        camera.update(controlled_rocket.get_world_com())
                    except Exception as e:
                        print(f"Error during respawn: {e}")
                        sim_running = False # Exit if respawn fails critically
                        break # Exit event loop

            # Mouse Click for Activation
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                 if controlled_rocket: # Can only interact with the controlled rocket
                     click_screen_pos = pygame.math.Vector2(event.pos)
                     click_world_pos = click_screen_pos + camera.offset
                     controlled_rocket.activate_part_at_pos(click_world_pos)


        # --- Continuous Input (Throttle) ---
        if controlled_rocket:
            keys = pygame.key.get_pressed()
            throttle_change = 0
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                throttle_change += THROTTLE_CHANGE_RATE * dt
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                throttle_change -= THROTTLE_CHANGE_RATE * dt

            if throttle_change != 0:
                controlled_rocket.throttle_level = max(0.0, min(1.0, controlled_rocket.throttle_level + throttle_change))

        # --- Updates ---
        new_rockets_to_add = [] # Store rockets created by separation
        rockets_to_remove = []  # Store rockets that become inactive

        # Update all active rocket instances
        for rocket in all_rockets:
            if not rocket.is_active:
                if rocket not in rockets_to_remove: rockets_to_remove.append(rocket)
                continue

            # Calculate current air density for this rocket
            rocket_com_alt = GROUND_Y - rocket.get_world_com().y
            current_air_density = get_air_density(rocket_com_alt)

            # Update physics and state
            rocket.update(dt, current_air_density)

            # --- Handle Pending Separations ---
            if rocket.pending_separation:
                print(f"Processing {len(rocket.pending_separation)} separations for rocket {rocket.sim_instance_id}...")
                # This is where the complex logic goes.
                # For each separator in pending_separation:
                # 1. Find parts connected "above/below" or "radially"
                # 2. Create new RocketBlueprint(s) with the separated parts.
                # 3. Create new FlyingRocket instance(s) from these blueprints.
                # 4. Apply separation impulse/velocity change.
                # 5. Remove separated parts from the original rocket.
                # 6. Add the new FlyingRocket instance(s) to `new_rockets_to_add`.
                # 7. Recalculate physics for the original rocket.

                # --- Simplified Placeholder ---
                # This placeholder just removes the separator and assumes no actual split occurs yet
                for sep in rocket.pending_separation:
                    print(f" >> Separator {sep.part_id} activated (Placeholder: No split yet).")
                    # sep.separated = True # Already marked for visual
                    # Optionally remove the separator part after activation?
                    # rocket.parts.remove(sep)
                    # rocket.separators.remove(sep)
                rocket.pending_separation.clear()
                # rocket.calculate_physics_properties() # Recalculate if parts removed


            # Check if rocket became inactive after update (e.g., all parts broken)
            if not rocket.is_active:
                if rocket not in rockets_to_remove: rockets_to_remove.append(rocket)


        # Remove inactive rockets and add newly separated ones
        for r in rockets_to_remove:
            if r in all_rockets:
                all_rockets.remove(r)
                print(f"Removed inactive rocket instance {r.sim_instance_id}.")
                if r == controlled_rocket:
                    controlled_rocket = None # Lost control
                    # Try find another rocket with control?
                    for rkt in all_rockets:
                        if rkt.has_active_control:
                            controlled_rocket = rkt
                            print(f"Control transferred to rocket instance {rkt.sim_instance_id}")
                            break


        all_rockets.extend(new_rockets_to_add) # Add rockets from separations

        # Update camera to follow the controlled rocket's CoM
        if controlled_rocket:
            camera.update(controlled_rocket.get_world_com())
        elif all_rockets:
            # If no controlled rocket, follow the first one in the list (arbitrary)
            camera.update(all_rockets[0].get_world_com())
        # Else: no rockets left, camera stays put

        # --- Drawing ---
        # 1. Background
        draw_earth_background(screen, camera, stars)

        # 2. Terrain
        draw_terrain(screen, camera)

        # 3. Rockets
        total_parts_drawn = 0
        total_broken_drawn = 0
        for rocket in all_rockets:
            broken_count = rocket.draw(screen, camera)
            total_parts_drawn += len(rocket.parts)
            total_broken_drawn += broken_count

        # 4. UI Overlay
        # Throttle Gauge (only if controlled rocket exists)
        if controlled_rocket:
            bar_width = 20
            bar_height = 100
            bar_x = 15
            bar_y = SCREEN_HEIGHT - bar_height - 40 # Moved up slightly
            pygame.draw.rect(screen, COLOR_UI_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
            fill_height = bar_height * controlled_rocket.throttle_level
            pygame.draw.rect(screen, COLOR_UI_BAR, (bar_x, bar_y + bar_height - fill_height, bar_width, fill_height))
            pygame.draw.rect(screen, WHITE, (bar_x, bar_y, bar_width, bar_height), 1)
            th_txt = ui_font.render("Thr", True, WHITE); screen.blit(th_txt, (bar_x, bar_y + bar_height + 5))
            th_val_txt = ui_font.render(f"{controlled_rocket.throttle_level * 100:.0f}%", True, WHITE); screen.blit(th_val_txt, (bar_x, bar_y - 18))

            # Status Text (for controlled rocket)
            alt_agl = max(0, GROUND_Y - controlled_rocket.get_lowest_point_world().y)
            alt_msl = GROUND_Y - controlled_rocket.get_world_com().y # Altitude of CoM
            control_status = "OK" if controlled_rocket.has_active_control else "NO CONTROL"
            master_thrust_status = "ON" if controlled_rocket.master_thrust_enabled else "OFF"
            landed_status = "LANDED" if controlled_rocket.landed else "FLYING"

            status_lines = [
                f"Alt (AGL): {alt_agl:.1f} m",
                f"Alt (MSL): {alt_msl:.1f} m",
                f"Vert Vel: {controlled_rocket.vel.y:.1f} m/s",
                f"Horz Vel: {controlled_rocket.vel.x:.1f} m/s",
                f"Total Vel: {controlled_rocket.vel.length():.1f} m/s",
                f"Angle: {controlled_rocket.angle:.1f} deg",
                f"Ang Vel: {controlled_rocket.angular_velocity:.1f} d/s",
                f"Throttle: {controlled_rocket.throttle_level*100:.0f}% [{master_thrust_status}]",
                f"Fuel: {controlled_rocket.current_fuel:.1f}",
                f"Mass: {controlled_rocket.total_mass:.1f} kg",
                f"Control: {control_status}",
                f"Status: {landed_status}",
                # f"Parts: {len(controlled_rocket.parts)} ({total_broken_drawn} broken)", # Show total broken?
            ]
            if controlled_rocket.original_root_part_obj in controlled_rocket.parts:
                 root_part = controlled_rocket.original_root_part_obj
                 status_lines.append(f"Root HP: {root_part.current_hp:.0f}/{root_part.part_data.get('max_hp',1)}")

            text_y = 10
            text_x = bar_x + bar_width + 10
            text_color = WHITE if controlled_rocket.has_active_control else RED
            for i, line in enumerate(status_lines):
                text_surf = ui_font.render(line, True, text_color)
                screen.blit(text_surf, (text_x, text_y + i * 18)) # Smaller line spacing

        elif not all_rockets: # No rockets left at all
             destroyed_text = ui_font_large.render("ALL ROCKETS DESTROYED", True, RED)
             text_rect = destroyed_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
             screen.blit(destroyed_text, text_rect)
             respawn_text = ui_font.render("Press 'R' to Respawn", True, WHITE)
             respawn_rect = respawn_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 40))
             screen.blit(respawn_text, respawn_rect)


        # FPS Counter (optional)
        fps = clock.get_fps()
        fps_text = ui_font.render(f"FPS: {fps:.1f}", True, WHITE)
        screen.blit(fps_text, (SCREEN_WIDTH - 80, 10))

        pygame.display.flip()

    print("--- Exiting Simulation ---")