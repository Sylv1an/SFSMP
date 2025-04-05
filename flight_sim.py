# flight_sim.py
import pygame
import math
import sys
import random
import time
import os
# Import necessary classes/functions
from parts import draw_part_shape, get_part_data, PARTS_CATALOG # Make sure PARTS_CATALOG is imported if needed by other functions
from rocket_data import RocketBlueprint, PlacedPart
# Make sure all needed constants and colors are imported or defined here
from ui_elements import SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, BLACK, GRAY, RED, GREEN, BLUE, LIGHT_GRAY
# Define colors if not in ui_elements
try: from ui_elements import COLOR_SKY_BLUE, COLOR_SPACE_BLACK, COLOR_HORIZON, COLOR_GROUND
except ImportError: COLOR_SKY_BLUE, COLOR_SPACE_BLACK, COLOR_HORIZON, COLOR_GROUND = (135, 206, 250), (0,0,0), (170, 210, 230), (0, 150, 0)
try: from ui_elements import COLOR_FLAME, COLOR_UI_BAR, COLOR_UI_BAR_BG, COLOR_EXPLOSION
except ImportError: COLOR_FLAME, COLOR_UI_BAR, COLOR_UI_BAR_BG, COLOR_EXPLOSION = (255,100,0), (0,200,0), (50,50,50), [(255,255,0),(255,150,0),(200,50,0),(150,150,150)]
try: from ui_elements import COLOR_ENGINE_ENABLED, COLOR_ENGINE_DISABLED, COLOR_ACTIVATABLE_READY, COLOR_ACTIVATABLE_USED
except ImportError: COLOR_ENGINE_ENABLED, COLOR_ENGINE_DISABLED, COLOR_ACTIVATABLE_READY, COLOR_ACTIVATABLE_USED = GREEN, RED, BLUE, GRAY

# --- Flight Sim Constants ---
# Ensure these are defined
GRAVITY = 9.81 * 6
ROTATION_SPEED = 200
REACTION_WHEEL_TORQUE = 10000
ANGULAR_DAMPING = 0.3
COLLISION_DAMAGE_FACTOR = 0.7
MIN_IMPACT_VEL_DAMAGE = 8
THROTTLE_CHANGE_RATE = 0.5
GROUND_Y = 1000
WORLD_WIDTH = 5000
BLUE_SKY_Y_LIMIT = -2000
SPACE_Y_LIMIT = -15000
STAR_COUNT = 20000
STAR_FIELD_DEPTH = 10000
AIR_DENSITY_SEA_LEVEL = 0.15
AIR_DENSITY_VACUUM = 0.0
ATMOSPHERE_HEIGHT = 10000

COLOR_EXPLOSION = [
    pygame.Color(255, 255, 0),    # Yellow
    pygame.Color(255, 150, 0),    # Orange
    pygame.Color(200, 50, 0),     # Dark Orange/Red
    pygame.Color(GRAY)            # Use imported GRAY
]

# --- Simple Particle Class for Explosions ---
class ExplosionEffect:
    def __init__(self, pos, num_particles=15, max_life=0.5, max_speed=100):
        self.particles=[]; self.pos=pygame.math.Vector2(pos); self.max_life=max_life
        # *** REMOVED global and redefinition ***
        for _ in range(num_particles):
            a=random.uniform(0,360); s=random.uniform(max_speed*0.2,max_speed)
            v=pygame.math.Vector2(s,0).rotate(a); l=random.uniform(max_life*0.3,max_life)
            # Use the module-level COLOR_EXPLOSION list directly
            c=random.choice(COLOR_EXPLOSION) # c is now guaranteed to be a pygame.Color
            r=random.uniform(2,5)
            self.particles.append([self.pos.copy(),v,l,l,c,r])
        self.is_alive=True
    def update(self, dt):
        if not self.is_alive: return
        any_particle_alive = False
        for p in self.particles:
            if p[2]>0: # if life remaining > 0
                p[0]+=p[1]*dt # Update position
                p[1]*=0.95 # Apply damping/drag
                p[2]-=dt # Decrease life
                any_particle_alive=True
        self.is_alive = any_particle_alive

    def draw(self, surface, camera):
        if not self.is_alive: return
        for p in self.particles:
            rl = p[2] # remaining life
            if rl > 0:
                sp = camera.apply(p[0]) # screen position
                ml = p[3] # max life
                af = max(0, rl / ml) if ml > 0 else 0 # alpha factor
                bc = p[4] # base color
                br = p[5] # base radius
                dc = (int(bc.r * af), int(bc.g * af), int(bc.b * af))
                dr = int(br * af) # derived radius
                if dc != (0, 0, 0) and dr > 0: # Check validity before drawing
                    pygame.draw.circle(surface, dc, sp, dr)

# --- Camera Class ---
class Camera:
    def __init__(self, width, height):
        self.camera_rect=pygame.Rect(0,0,width,height)
        self.width=width; self.height=height
        self.offset=pygame.math.Vector2(0,0)
    def apply(self, target_pos):
        return target_pos-self.offset
    def apply_rect(self, target_rect):
        return target_rect.move(-self.offset.x,-self.offset.y)
    def update(self, target_pos):
        # Center camera on target_pos
        x = target_pos.x - self.width // 2
        y = target_pos.y - self.height // 2
        self.offset = pygame.math.Vector2(x,y)

# --- FlyingRocket Class ---
class FlyingRocket:
    def __init__(self, parts_list: list[PlacedPart], initial_pos_offset: pygame.math.Vector2, initial_angle=0, initial_vel=pygame.math.Vector2(0,0), sim_instance_id=0, is_primary_control=False, original_root_ref=None):
        self.sim_instance_id = sim_instance_id
        self.parts = parts_list
        if not self.parts: raise ValueError("Cannot initialize FlyingRocket with an empty parts list.")
        self.blueprint_name = f"Rocket_{sim_instance_id}"
        self.original_root_part_ref = original_root_ref
        self.has_active_control = is_primary_control

        # Physics state
        self.pos = pygame.math.Vector2(0,0)
        self.vel = pygame.math.Vector2(initial_vel)
        self.acc = pygame.math.Vector2(0,0)
        self.angle = initial_angle
        self.angular_velocity = 0.0

        # Control state
        self.throttle_level = 0.0
        self.master_thrust_enabled = False

        # Components & Resources
        self.engines = []
        self.fuel_tanks = []
        self.parachutes = []
        self.separators = []
        total_fuel_cap_this_assembly = 0
        for i, part in enumerate(self.parts):
            # Ensure runtime state is reset
            part.current_hp = part.part_data.get("max_hp", 100)
            part.is_broken = False
            part.engine_enabled = True
            part.deployed = False # Parachute specific
            part.separated = False # Separator specific
            part.part_index = i
            pt = part.part_data.get("type")
            if pt == "Engine": self.engines.append(part)
            elif pt == "FuelTank": self.fuel_tanks.append(part); total_fuel_cap_this_assembly += part.part_data.get("fuel_capacity", 0)
            elif pt == "Parachute": self.parachutes.append(part)
            elif pt == "Separator": self.separators.append(part)
        self.current_fuel = total_fuel_cap_this_assembly
        self.fuel_mass_per_unit = 0.1

        # Physics properties
        self.total_mass = 0; self.dry_mass = 0; self.moment_of_inertia = 10000
        self.center_of_mass_offset = pygame.math.Vector2(0, 0)
        self.local_bounds = pygame.Rect(0,0,1,1)
        self.calculate_physics_properties(); self.calculate_bounds()

        # Set initial position based on target CoM
        initial_com_offset_rotated = self.center_of_mass_offset.rotate(-self.angle)
        self.pos = initial_pos_offset - initial_com_offset_rotated

        # Effects and runtime flags
        self.effects = []; self.landed = False; self.thrusting = False
        self.is_active = True; self.pending_separation = []; self.needs_connectivity_check = False
        self.was_landed_last_frame = False # Track landing state change

    def calculate_physics_properties(self):
        total_m = 0.0; com_numerator = pygame.math.Vector2(0, 0); moi_sum = 0.0
        if not self.parts: self.total_mass = 0.01; self.center_of_mass_offset = pygame.math.Vector2(0, 0); self.moment_of_inertia = 1.0; self.dry_mass = 0.0; return
        fuel_mass_total = self.current_fuel * self.fuel_mass_per_unit
        total_tank_capacity = sum(p.part_data.get("fuel_capacity", 0) for p in self.fuel_tanks); total_tank_capacity = max(1.0, total_tank_capacity)
        self.dry_mass = sum(p.part_data.get("mass", 0) for p in self.parts)
        for part in self.parts:
            part_mass = part.part_data.get("mass", 0)
            if part.part_data.get("type") == "FuelTank": part_mass += fuel_mass_total * (part.part_data.get("fuel_capacity", 0) / total_tank_capacity)
            total_m += part_mass; com_numerator += part.relative_pos * part_mass
        self.total_mass = max(0.01, total_m)
        self.center_of_mass_offset = com_numerator / self.total_mass if self.total_mass > 0.01 else pygame.math.Vector2(0, 0)
        for part in self.parts:
             part_mass = part.part_data.get("mass", 0)
             if part.part_data.get("type") == "FuelTank": part_mass += fuel_mass_total * (part.part_data.get("fuel_capacity", 0) / total_tank_capacity)
             w = part.part_data.get("width", 1); h = part.part_data.get("height", 1); i_part = (1/12.0) * part_mass * (w**2 + h**2)
             dist_vec = part.relative_pos - self.center_of_mass_offset; d_sq = dist_vec.length_squared(); moi_sum += i_part + part_mass * d_sq
        self.moment_of_inertia = max(1.0, moi_sum)

    def calculate_bounds(self):
        if not self.parts: self.local_bounds = pygame.Rect(0, 0, 0, 0); return
        min_x, max_x = float('inf'), float('-inf'); min_y, max_y = float('inf'), float('-inf')
        for p in self.parts: hw=p.part_data['width']/2; hh=p.part_data['height']/2; cx=p.relative_pos.x; cy=p.relative_pos.y; min_x=min(min_x, cx-hw); max_x=max(max_x, cx+hw); min_y=min(min_y, cy-hh); max_y=max(max_y, cy+hh)
        if min_x == float('inf'): self.local_bounds = pygame.Rect(0,0,0,0)
        else: self.local_bounds = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)

    def get_world_com(self):
        return self.pos + self.center_of_mass_offset.rotate(-self.angle)

    def get_world_part_center(self, part: PlacedPart):
        return self.pos + part.relative_pos.rotate(-self.angle)

    def get_parts_near_world_pos(self, world_pos: pygame.math.Vector2, radius: float = 20.0):
        nearby_parts = []; radius_sq = radius * radius
        for part in self.parts:
            if (self.get_world_part_center(part) - world_pos).length_squared() < radius_sq: nearby_parts.append(part)
        return nearby_parts

    def get_lowest_point_world(self) -> pygame.math.Vector2:
        if not self.parts: return self.pos
        lowest_y = float('-inf'); lowest_point_world = self.pos
        for part in self.parts:
            pcw = self.get_world_part_center(part); w=part.part_data['width']; h=part.part_data['height']; pwa = self.angle
            corners = [pcw + pygame.math.Vector2(x,y).rotate(-pwa) for x in [-w/2,w/2] for y in [-h/2,h/2]]
            for corner in corners:
                if corner.y > lowest_y: lowest_y = corner.y; lowest_point_world = corner
        if lowest_y == float('-inf'): return self.get_world_com()
        return lowest_point_world

    def get_world_part_aabb(self, part: PlacedPart) -> pygame.Rect:
        pd = part.part_data; w = pd.get('width', 1); h = pd.get('height', 1)
        wc = self.get_world_part_center(part); aabb = pygame.Rect(0, 0, w, h); aabb.center = wc
        return aabb

    def consume_fuel(self, amount):
        consumed = min(self.current_fuel, amount)
        if consumed > 0: self.current_fuel -= consumed; self.calculate_physics_properties(); return True
        return False

    def get_thrust_data(self) -> tuple[pygame.math.Vector2, pygame.math.Vector2 | None, float]:
        total_thrust_potential = 0.0; thrust_torque_num = pygame.math.Vector2(0, 0); total_consumption_100 = 0.0; active_engines = 0
        if not self.master_thrust_enabled or self.throttle_level <= 0 or self.current_fuel <= 0: return pygame.math.Vector2(0,0), None, 0.0
        for engine in self.engines:
            if engine.is_broken or not engine.engine_enabled: continue
            thrust = engine.part_data.get("thrust", 0); cons = engine.part_data.get("fuel_consumption", 0)
            offset = engine.relative_pos.rotate(-self.angle); total_thrust_potential += thrust; thrust_torque_num += offset * thrust; total_consumption_100 += cons; active_engines += 1
        force = pygame.math.Vector2(0, 0); app_point = None
        if active_engines > 0 and total_thrust_potential > 0:
            actual_thrust = total_thrust_potential * self.throttle_level
            direction = pygame.math.Vector2(0, -1).rotate(-self.angle); force = direction * actual_thrust
            avg_offset = thrust_torque_num / total_thrust_potential; app_point = self.pos + avg_offset
        return force, app_point, total_consumption_100

    # *** REVISED DAMAGE APPLICATION FOR GROUND IMPACT ***
    def apply_collision_damage(self, impact_velocity_magnitude, specific_part_to_damage: PlacedPart | None = None):
        if impact_velocity_magnitude < MIN_IMPACT_VEL_DAMAGE: return
        base_damage = (impact_velocity_magnitude ** 1.8) * COLLISION_DAMAGE_FACTOR
        print(f"[{self.sim_instance_id}] Impact Vel: {impact_velocity_magnitude:.1f} -> BaseDmg: {base_damage:.1f}")

        parts_to_damage = []
        damage_multipliers = {} # Store multiplier per part

        if specific_part_to_damage and specific_part_to_damage in self.parts:
             # Inter-rocket collision - target specific part
             parts_to_damage = [specific_part_to_damage]
             damage_multipliers[specific_part_to_damage] = 1.0 # Full damage
             print(f"  Targeting specific part: {specific_part_to_damage.part_id}")
        elif not specific_part_to_damage and self.parts:
             # Ground collision - Apply damage to all parts, scaled by height
             lowest_y = self.get_lowest_point_world().y
             rocket_height = lowest_y - self.get_world_com().y # Rough height estimate based on CoM and lowest point
             if rocket_height <= 1.0: rocket_height = 1.0 # Avoid division by zero

             for part in self.parts:
                 part_center_y = self.get_world_part_center(part).y
                 # Calculate normalized height (0 = lowest point, 1 = CoM approx, higher above)
                 # We want more damage lower down.
                 # Relative position from lowest point up towards CoM
                 relative_y = lowest_y - part_center_y
                 # Factor: 1.0 at lowest point, decreasing towards 0 higher up
                 # Use exponential decay or linear? Linear for simplicity:
                 damage_factor = max(0.0, min(1.0, 1.0 - (relative_y / (rocket_height * 1.5)))) # Scale over 1.5x height
                 # Apply more aggressively: square the factor to make lower parts take much more
                 damage_factor = damage_factor ** 2

                 if damage_factor > 0.01: # Only consider parts significantly low
                     parts_to_damage.append(part)
                     damage_multipliers[part] = damage_factor
                     # print(f"  Part {part.part_id} y={part_center_y:.1f}, lowest={lowest_y:.1f}, factor={damage_factor:.2f}") # Debug

             if not parts_to_damage: # If somehow no parts were low enough, damage the absolute lowest
                  lowest_part = min(self.parts, key=lambda p: self.get_world_part_center(p).y)
                  if lowest_part: parts_to_damage = [lowest_part]; damage_multipliers[lowest_part] = 1.0


        # Apply scaled damage
        parts_destroyed_this_impact = []
        for part in parts_to_damage:
            if part.is_broken: continue
            scaled_damage = base_damage * damage_multipliers.get(part, 0.0)
            if scaled_damage < 0.1: continue # Skip negligible damage

            part.current_hp -= scaled_damage
            print(f"  Part '{part.part_id}' HP: {part.current_hp:.0f} / {part.part_data.get('max_hp', 1)} (Dmg: {scaled_damage:.1f})")
            if part.current_hp <= 0:
                print(f"  >> Part '{part.part_id}' BROKEN! <<"); part.is_broken = True; part.current_hp = 0
                parts_destroyed_this_impact.append(part)
                try: self.effects.append(ExplosionEffect(self.get_world_part_center(part)))
                except NameError: print("Error: ExplosionEffect class missing")
        if parts_destroyed_this_impact: self.handle_destroyed_parts(parts_destroyed_this_impact)

    def handle_destroyed_parts(self, destroyed_parts: list[PlacedPart]):
        if not destroyed_parts: return
        print(f"[{self.sim_instance_id}] Handling destruction: {[p.part_id for p in destroyed_parts]}")
        original_part_count = len(self.parts); self.parts = [p for p in self.parts if p not in destroyed_parts]
        if self.original_root_part_ref and self.original_root_part_ref in destroyed_parts: print(f"[{self.sim_instance_id}] Original root destroyed!"); self.has_active_control = False
        if not self.parts: print(f"[{self.sim_instance_id}] All parts destroyed!"); self.is_active = False; return
        if len(self.parts) < original_part_count:
            self.needs_connectivity_check = True; print(f"[{self.sim_instance_id}] Marking for connectivity check.")
            self.engines=[e for e in self.engines if e in self.parts]; self.fuel_tanks=[t for t in self.fuel_tanks if t in self.parts]; self.parachutes=[pc for pc in self.parachutes if pc in self.parts]; self.separators=[s for s in self.separators if s in self.parts]
            self.calculate_physics_properties(); self.calculate_bounds()

    def activate_part_at_pos(self, click_world_pos):
        clicked_part = None; min_dist_sq = 20**2
        parts_to_check = self.engines + self.parachutes + self.separators
        for part in parts_to_check:
            if part.is_broken: continue
            if (self.get_world_part_center(part) - click_world_pos).length_squared() < min_dist_sq: clicked_part = part; min_dist_sq = (self.get_world_part_center(part) - click_world_pos).length_squared() # Update min_dist_sq
        if not clicked_part or not clicked_part.part_data.get("activatable", False): return False

        part_type = clicked_part.part_data.get("type")
        if part_type == "Engine": clicked_part.engine_enabled = not clicked_part.engine_enabled; print(f"Toggled E {clicked_part.part_id} {'ON' if clicked_part.engine_enabled else 'OFF'}"); return True
        elif part_type == "Parachute":
            if not clicked_part.deployed: clicked_part.deployed = True; print(f"Deployed P {clicked_part.part_id}!"); return True
            else: return False
        elif part_type == "Separator":
            if not clicked_part.separated:
                 if clicked_part not in self.pending_separation: self.pending_separation.append(clicked_part); clicked_part.separated = True; print(f"Activated S {clicked_part.part_id}."); return True
                 else: return True # Already pending
            else: return False
        return False

    def update(self, dt, current_air_density):
        if not self.is_active or not self.parts: [e.update(dt) for e in self.effects]; self.effects = [e for e in self.effects if e.is_alive]; return

        self.acc = pygame.math.Vector2(0, 0); net_torque = 0.0

        # --- Forces ---
        if self.total_mass > 0.01: self.acc += pygame.math.Vector2(0, GRAVITY * self.total_mass) / self.total_mass # Gravity
        thrust_force, thrust_app, cons_rate = self.get_thrust_data(); self.thrusting = False
        if thrust_force.length_squared() > 0 and self.consume_fuel(cons_rate * self.throttle_level * dt):
            self.thrusting = True
            if self.total_mass > 0.01: self.acc += thrust_force / self.total_mass
            if thrust_app: net_torque += (thrust_app - self.get_world_com()).cross(thrust_force)
        drag_force = pygame.math.Vector2(0,0) # Drag (simplified)
        if current_air_density > AIR_DENSITY_VACUUM and self.vel.length_squared() > 0.1:
            cd = 0.8; area = max(1.0, self.local_bounds.width); total_cd = cd
            for chute in self.parachutes:
                 if chute.deployed and not chute.is_broken: total_cd += chute.part_data.get("deploy_drag",0); area += chute.part_data.get("width",5)*chute.part_data.get("deploy_area_factor",10)
            v_sq = self.vel.length_squared(); v_mag = math.sqrt(v_sq); drag_mag = 0.5*current_air_density*v_sq*total_cd*area
            if v_mag > 0: drag_force = -self.vel.normalize() * drag_mag
            if self.total_mass > 0.01: self.acc += drag_force / self.total_mass
            # drag_torque = (self.get_world_com() - self.get_world_com()).cross(drag_force); net_torque += drag_torque # Approx drag at CoM

        # --- Control Status & Torque ---
        root_ref = self.original_root_part_ref; self.has_active_control = (root_ref is not None) and (root_ref in self.parts) and (not root_ref.is_broken)
        ctrl_torque = 0
        if self.has_active_control:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]: ctrl_torque += REACTION_WHEEL_TORQUE # CCW Positive
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]: ctrl_torque -= REACTION_WHEEL_TORQUE # CW Negative
        net_torque += ctrl_torque

        # --- Integration ---
        self.vel += self.acc * dt; self.pos += self.vel * dt
        if self.moment_of_inertia > 0: self.angular_velocity += math.degrees(net_torque / self.moment_of_inertia) * dt
        self.angular_velocity *= (1.0 - ANGULAR_DAMPING * dt); self.angle = (self.angle + self.angular_velocity * dt) % 360

        # --- Landing State & Parachute Destruction ---
        currently_on_ground = self.get_lowest_point_world().y >= GROUND_Y
        just_landed = currently_on_ground and not self.was_landed_last_frame
        just_took_off = not currently_on_ground and self.was_landed_last_frame

        if just_landed: # Process landing impact
            impact_velocity = self.vel.copy(); impact_vel_mag = impact_velocity.length()
            should_apply_damage = (not self.thrusting and impact_vel_mag >= MIN_IMPACT_VEL_DAMAGE) or \
                                  (self.thrusting and impact_velocity.y > 1.0 and impact_vel_mag >= MIN_IMPACT_VEL_DAMAGE)
            if should_apply_damage: self.apply_collision_damage(impact_vel_mag, None)

        if currently_on_ground: # Apply ground constraints
            self.landed = True
            if self.is_active and self.parts:
                correction = self.get_lowest_point_world().y - GROUND_Y
                if correction > 0: self.pos.y -= correction
                self.vel.y = 0; self.vel.x *= 0.6; self.angular_velocity = 0
            else: self.vel = pygame.math.Vector2(0,0) # Stop wreckage
        else:
            self.landed = False

        if just_took_off: # Check parachutes on takeoff
            print(f"[{self.sim_instance_id}] Took off.")
            destroyed_chutes = []
            for chute in self.parachutes:
                if chute.deployed and not chute.is_broken:
                    print(f"  >> Parachute {chute.part_id} destroyed by takeoff after deployment! <<")
                    chute.is_broken = True # Mark as broken
                    chute.deployed = False # Can't be deployed if broken
                    chute.current_hp = 0
                    destroyed_chutes.append(chute)
                    # Optional: Add a small "shredding" effect
                    try: self.effects.append(ExplosionEffect(self.get_world_part_center(chute), num_particles=5, max_life=0.3, max_speed=50))
                    except NameError: pass
            # If chutes were destroyed, might need structure check? Unlikely for chutes.
            # if destroyed_chutes: self.handle_destroyed_parts(destroyed_chutes) # Might be overkill

        self.was_landed_last_frame = self.landed # Store state for next frame

        # --- Update Effects ---
        [e.update(dt) for e in self.effects]; self.effects = [e for e in self.effects if e.is_alive]

    def draw(self, surface, camera):
        num_broken_visually = 0
        if not self.is_active: return 0
        for part in self.parts:
            part_center_world = self.get_world_part_center(part); part_screen_pos = camera.apply(part_center_world); part_world_angle = self.angle
            indicator_color = None; activatable = part.part_data.get("activatable", False)
            is_chute = part.part_data.get("type") == "Parachute"
            show_deployed = part.deployed and not part.is_broken

            # *** HIDE DEPLOYED CHUTE VISUAL IF LANDED ***
            if is_chute and self.landed:
                show_deployed = False

            if part.is_broken: num_broken_visually += 1
            elif activatable: # Determine indicator color
                 if is_chute: indicator_color = COLOR_ACTIVATABLE_USED if part.deployed else COLOR_ACTIVATABLE_READY
                 elif part.part_data.get("type") == "Engine": indicator_color = COLOR_ENGINE_ENABLED if part.engine_enabled else COLOR_ENGINE_DISABLED
                 elif part.part_data.get("type") == "Separator": indicator_color = COLOR_ACTIVATABLE_USED if part.separated else COLOR_ACTIVATABLE_READY

            # Call draw_part_shape, passing the possibly modified show_deployed flag
            try: draw_part_shape(surface, part.part_data, part_screen_pos, part_world_angle, broken=part.is_broken, deployed=show_deployed)
            except NameError: pygame.draw.circle(surface, RED, part_screen_pos, 5)
            if indicator_color: pygame.draw.circle(surface, indicator_color, part_screen_pos, 5)

        if self.thrusting: # Draw flame
            flame_scale = 0.5 + 0.5 * self.throttle_level
            for engine in self.engines:
                if engine.engine_enabled and not engine.is_broken:
                    ecw=self.get_world_part_center(engine); ewa=self.angle; flbo=pygame.math.Vector2(0, engine.part_data["height"]/2).rotate(-ewa); fbw=ecw+flbo; fl=(15+random.uniform(-2,2))*flame_scale; fw=engine.part_data["width"]*0.8*flame_scale; fdw=pygame.math.Vector2(0,1).rotate(-ewa); fsw=pygame.math.Vector2(1,0).rotate(-ewa); ftw=fbw+fdw*fl; flw=fbw-fsw*fw/2; frw=fbw+fsw*fw/2; fps=[camera.apply(p) for p in [flw,frw,ftw]]
                    try: pygame.draw.polygon(surface, COLOR_FLAME, fps)
                    except NameError: pygame.draw.line(surface, RED, camera.apply(fbw), camera.apply(ftw), 3)

        [e.draw(surface, camera) for e in self.effects]
        return num_broken_visually

# --- Background/Terrain Functions ---
# Ensure these functions are defined or imported above run_simulation
def create_stars(count, bounds):
    stars = []; STAR_FIELD_DEPTH = bounds.height # Example depth
    for _ in range(count): x = random.uniform(bounds.left, bounds.right); y = random.uniform(bounds.top, bounds.bottom); z = random.uniform(1, max(2, STAR_FIELD_DEPTH)); stars.append((pygame.math.Vector2(x, y), z));
    return stars
def get_air_density(altitude_agl):
    ATMOSPHERE_HEIGHT = 10000 # Define if not global
    if altitude_agl < 0: return AIR_DENSITY_SEA_LEVEL
    if altitude_agl > ATMOSPHERE_HEIGHT: return AIR_DENSITY_VACUUM
    factor = 1.0 - (altitude_agl / ATMOSPHERE_HEIGHT); return AIR_DENSITY_SEA_LEVEL * factor

def draw_earth_background(surface, camera, stars):
    screen_rect = surface.get_rect()
    avg_world_y = camera.offset.y + camera.height / 2
    ground_screen_y = camera.apply(pygame.math.Vector2(0, GROUND_Y)).y

    # Define limits if not global
    try: _ = BLUE_SKY_Y_LIMIT; _ = SPACE_Y_LIMIT
    except NameError: BLUE_SKY_Y_LIMIT = -2000; SPACE_Y_LIMIT = -15000
    # Define colors if not global
    try: _ = COLOR_SKY_BLUE; _ = COLOR_SPACE_BLACK
    except NameError: COLOR_SKY_BLUE = (135, 206, 250); COLOR_SPACE_BLACK = (0,0,0)


    if avg_world_y > BLUE_SKY_Y_LIMIT: # Low altitude
        if ground_screen_y < screen_rect.bottom:
             pygame.draw.rect(surface, COLOR_SKY_BLUE, (0, 0, screen_rect.width, ground_screen_y))
        else: surface.fill(COLOR_SKY_BLUE)
        # No stars here

    elif avg_world_y < SPACE_Y_LIMIT: # High altitude / space
        surface.fill(COLOR_SPACE_BLACK)
        # *** Call draw_stars with full alpha ***
        draw_stars(surface, stars, camera, alpha=255)

    else: # Transition altitude
        interp_factor = max(0.0, min(1.0, (avg_world_y - BLUE_SKY_Y_LIMIT) / (SPACE_Y_LIMIT - BLUE_SKY_Y_LIMIT)))
        bg_color = pygame.Color(COLOR_SKY_BLUE).lerp(COLOR_SPACE_BLACK, interp_factor)
        surface.fill(bg_color)
        star_alpha = int(255 * interp_factor)
        if star_alpha > 10:
            # *** Call draw_stars with fading alpha ***
            draw_stars(surface, stars, camera, alpha=star_alpha)

def draw_stars(surface, stars, camera, alpha=255):
    if alpha <= 0: return
    screen_rect = surface.get_rect() # Get screen dimensions
    base_star_color = pygame.Color(200, 200, 200)
    try: _ = STAR_FIELD_DEPTH
    except NameError: STAR_FIELD_DEPTH = 10000

    for star_world_pos, z in stars:
        parallax_factor = 1.0 / (z / (STAR_FIELD_DEPTH / 20) + 1) # Adjust divisor for effect strength
        star_relative_to_cam = star_world_pos - camera.offset
        effective_offset = camera.offset * parallax_factor # Far stars use less offset
        star_screen_pos = star_world_pos - effective_offset
        sx = int(star_screen_pos.x)
        sy = int(star_screen_pos.y)

        if 0 <= sx < screen_rect.width and 0 <= sy < screen_rect.height:
            # Size based on depth (closer stars slightly larger)
            size = max(1, int(2.5 * (1.0 - z / max(1, STAR_FIELD_DEPTH))))
            # Final color with alpha
            alpha_factor = alpha / 255.0
            final_color_tuple = (int(base_star_color.r * alpha_factor),
                                 int(base_star_color.g * alpha_factor),
                                 int(base_star_color.b * alpha_factor))

            if final_color_tuple != (0,0,0): # Avoid drawing black stars
                 pygame.draw.circle(surface, final_color_tuple, (sx, sy), size)

def draw_terrain(surface, camera):
    WORLD_WIDTH = 5000; GROUND_Y = 1000; COLOR_GROUND = (0,150,0) # Define if not global
    ground_view_rect = pygame.Rect(camera.offset.x - WORLD_WIDTH, GROUND_Y, camera.width + WORLD_WIDTH * 2, SCREEN_HEIGHT * 2)
    ground_rect_screen = camera.apply_rect(ground_view_rect); pygame.draw.rect(surface, COLOR_GROUND, ground_rect_screen)


# --- Simulation Runner Function ---
def run_simulation(screen, clock, blueprint_file):
    print(f"--- Starting Simulation ---")
    print(f"Loading blueprint: {blueprint_file}")
    if not os.path.exists(blueprint_file): print(f"Error: Blueprint file not found: {blueprint_file}"); return
    initial_blueprint = RocketBlueprint.load_from_json(blueprint_file)
    if not initial_blueprint or not initial_blueprint.parts: print("Blueprint load failed or empty."); return

    all_rockets: list[FlyingRocket] = []; controlled_rocket: FlyingRocket | None = None; next_sim_id = 0
    original_root_part_instance = None
    if initial_blueprint.parts:
        for part in initial_blueprint.parts:
            if part.part_data and part.part_data.get("type") == "CommandPod": original_root_part_instance = part; print(f"DEBUG: Identified CommandPod '{part.part_id}' as root ref."); break
        if not original_root_part_instance: print("Warning: No CommandPod found in blueprint.")

    initial_subassemblies = initial_blueprint.find_connected_subassemblies()
    if not initial_subassemblies: print("Error: No parts found after connectivity check."); return

    for i, assembly_parts in enumerate(initial_subassemblies):
        if not assembly_parts: continue
        temp_bp = RocketBlueprint(); temp_bp.parts = assembly_parts
        lowest_offset_y = temp_bp.get_lowest_point_offset_y(); start_x = i * 50; start_y = GROUND_Y - lowest_offset_y + 500
        target_initial_com_pos = pygame.math.Vector2(start_x, start_y)
        contains_original_root = original_root_part_instance and (original_root_part_instance in assembly_parts)
        is_primary = (controlled_rocket is None and contains_original_root) or (controlled_rocket is None and original_root_part_instance is None and i == 0)
        try:
            rocket_instance = FlyingRocket(list(assembly_parts), target_initial_com_pos, 0, pygame.math.Vector2(0,0), next_sim_id, is_primary, original_root_part_instance)
            all_rockets.append(rocket_instance)
            if is_primary:
                 controlled_rocket = rocket_instance
                 root_ref_in_instance = controlled_rocket.original_root_part_ref
                 controlled_rocket.has_active_control = (root_ref_in_instance is not None) and (root_ref_in_instance in controlled_rocket.parts) and (not root_ref_in_instance.is_broken)
                 control_status_msg = 'CONTROLLED' if controlled_rocket.has_active_control else 'CONTROLLED (No Pod/Broken)'
                 print(f"Created initial rocket {next_sim_id} ({control_status_msg}) with {len(assembly_parts)} parts.")
            else: rocket_instance.has_active_control = False; print(f"Created initial rocket {next_sim_id} (DEBRIS) with {len(assembly_parts)} parts.")
            next_sim_id += 1
        except Exception as e: print(f"Error initializing rocket instance {next_sim_id}: {e}")

    if controlled_rocket is None and all_rockets:
         print("Warning: Fallback control assignment to first assembly."); controlled_rocket = all_rockets[0]
         root_ref_in_fallback = controlled_rocket.original_root_part_ref
         controlled_rocket.has_active_control = (root_ref_in_fallback is not None) and (root_ref_in_fallback in controlled_rocket.parts) and (not root_ref_in_fallback.is_broken)

    camera = Camera(SCREEN_WIDTH, SCREEN_HEIGHT)
    if controlled_rocket: camera.update(controlled_rocket.get_world_com())
    elif all_rockets: camera.update(all_rockets[0].get_world_com())
    else: camera.update(pygame.math.Vector2(0, GROUND_Y))
    try: star_area_bounds = pygame.Rect(-WORLD_WIDTH * 2, SPACE_Y_LIMIT - STAR_FIELD_DEPTH, WORLD_WIDTH * 4, abs(SPACE_Y_LIMIT) + GROUND_Y + STAR_FIELD_DEPTH*1.5); stars = create_stars(STAR_COUNT * 2, star_area_bounds)
    except NameError: stars = []
    ui_font = pygame.font.SysFont(None, 20); ui_font_large = pygame.font.SysFont(None, 36)

    sim_running = True; last_respawn_time = time.time()
    while sim_running:
        dt = clock.tick(60) / 1000.0; dt = min(dt, 0.05)
        newly_created_rockets_this_frame: list[FlyingRocket] = []; rockets_to_remove_this_frame: list[FlyingRocket] = []

        for event in pygame.event.get(): # --- Event Handling ---
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: sim_running = False
                if controlled_rocket and controlled_rocket.has_active_control:
                    if event.key == pygame.K_SPACE: controlled_rocket.master_thrust_enabled = not controlled_rocket.master_thrust_enabled; print(f"[{controlled_rocket.sim_instance_id}] Master Thrust: {controlled_rocket.master_thrust_enabled}")
                    if event.key == pygame.K_p:
                         for chute in controlled_rocket.parachutes:
                             if not chute.deployed and not chute.is_broken: chute.deployed = True; print(f"Deployed {chute.part_id} via key.")
                current_time = time.time() # --- Respawn ---
                if event.key == pygame.K_r and (current_time - last_respawn_time > 1.0):
                    print("--- RESPAWNING ROCKET ---"); last_respawn_time = current_time; all_rockets.clear(); controlled_rocket = None; newly_created_rockets_this_frame.clear(); rockets_to_remove_this_frame.clear()
                    reloaded_blueprint = RocketBlueprint.load_from_json(blueprint_file)
                    if reloaded_blueprint and reloaded_blueprint.parts:
                        original_root_part_instance = None
                        for part in reloaded_blueprint.parts:
                            if part.part_data and part.part_data.get("type") == "CommandPod": original_root_part_instance = part; break
                        initial_subassemblies = reloaded_blueprint.find_connected_subassemblies()
                        for i, assembly_parts in enumerate(initial_subassemblies):
                            if not assembly_parts: continue
                            temp_bp = RocketBlueprint(); temp_bp.parts = assembly_parts; lowest_offset_y = temp_bp.get_lowest_point_offset_y(); start_x = i * 50; start_y = GROUND_Y - lowest_offset_y - 0.1
                            target_initial_com_pos = pygame.math.Vector2(start_x, start_y); contains_original_root = original_root_part_instance and (original_root_part_instance in assembly_parts)
                            is_primary = (controlled_rocket is None and contains_original_root) or (controlled_rocket is None and original_root_part_instance is None and i == 0)
                            try:
                                rocket_instance = FlyingRocket(list(assembly_parts), target_initial_com_pos, 0, pygame.math.Vector2(0,0), next_sim_id, is_primary, original_root_part_instance)
                                newly_created_rockets_this_frame.append(rocket_instance)
                                if is_primary: controlled_rocket = rocket_instance; root_ref = controlled_rocket.original_root_part_ref; controlled_rocket.has_active_control = (root_ref is not None) and (root_ref in controlled_rocket.parts) and (not root_ref.is_broken)
                                else: rocket_instance.has_active_control = False
                                next_sim_id += 1
                            except Exception as e: print(f"Respawn Error creating instance: {e}")
                        if controlled_rocket is None and newly_created_rockets_this_frame: controlled_rocket = newly_created_rockets_this_frame[0]; root_ref = controlled_rocket.original_root_part_ref; controlled_rocket.has_active_control = (root_ref is not None) and (root_ref in controlled_rocket.parts) and (not root_ref.is_broken)
                        print("Respawn Complete.")
                    else: print("Respawn Failed: Cannot reload blueprint.")
                    all_rockets.extend(newly_created_rockets_this_frame); newly_created_rockets_this_frame.clear() # Process respawn immediately
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: # --- Click Activation ---
                 if controlled_rocket: click_screen_pos = pygame.math.Vector2(event.pos); click_world_pos = click_screen_pos + camera.offset; controlled_rocket.activate_part_at_pos(click_world_pos)

        if controlled_rocket and controlled_rocket.has_active_control: # --- Throttle Input ---
            keys = pygame.key.get_pressed(); throttle_change = 0
            if keys[pygame.K_w] or keys[pygame.K_UP]: throttle_change += THROTTLE_CHANGE_RATE * dt
            if keys[pygame.K_s] or keys[pygame.K_DOWN]: throttle_change -= THROTTLE_CHANGE_RATE * dt
            if throttle_change != 0: controlled_rocket.throttle_level = max(0.0, min(1.0, controlled_rocket.throttle_level + throttle_change))

        for rocket in all_rockets: # --- Stage 1: Physics Update ---
            if not rocket.is_active: continue
            try: current_air_density = get_air_density(GROUND_Y - rocket.get_world_com().y)
            except NameError: current_air_density = 0.0
            rocket.update(dt, current_air_density)
            if not rocket.is_active and rocket not in rockets_to_remove_this_frame: rockets_to_remove_this_frame.append(rocket)

        collision_pairs_processed = set() # --- Stage 1.5: Inter-Rocket Collision ---
        for i, r1 in enumerate(all_rockets):
            # Skip inactive/removed rockets or rockets with no parts
            if r1 in rockets_to_remove_this_frame or not r1.is_active or not r1.parts:
                continue

            for j in range(i + 1, len(all_rockets)):
                r2 = all_rockets[j]
                # Skip inactive/removed rockets or rockets with no parts
                if r2 in rockets_to_remove_this_frame or not r2.is_active or not r2.parts:
                    continue

                # Optional Broad Phase Check
                # ...

                # Narrow Phase
                collision_found_between_r1_r2 = False
                # Outer loop for parts in the first rocket (r1)
                for p1 in r1.parts:
                    if p1.is_broken: continue
                    # Define rect1 *inside* the p1 loop
                    rect1 = r1.get_world_part_aabb(p1)

                    # Inner loop for parts in the second rocket (r2)
                    # This loop MUST be indented relative to the p1 loop
                    for p2 in r2.parts:
                        if p2.is_broken: continue
                        # Define rect2 *inside* the p2 loop
                        rect2 = r2.get_world_part_aabb(p2)

                        # Perform the collision check *inside* the p2 loop
                        if rect1.colliderect(rect2):
                            pair_key = tuple(sorted((r1.sim_instance_id, r2.sim_instance_id)))
                            if pair_key in collision_pairs_processed: continue  # Skip if pair already handled

                            print(
                                f"COLLISION: R{r1.sim_instance_id}({p1.part_id}) & R{r2.sim_instance_id}({p2.part_id})")
                            collision_found_between_r1_r2 = True
                            collision_pairs_processed.add(pair_key)

                            # --- Collision Response ---
                            relative_velocity = r1.vel - r2.vel;
                            impact_speed = relative_velocity.length()
                            r1.apply_collision_damage(impact_speed, specific_part_to_damage=p1)
                            r2.apply_collision_damage(impact_speed, specific_part_to_damage=p2)
                            # --- Basic Push Apart ---
                            overlap_vec = pygame.math.Vector2(rect1.center) - pygame.math.Vector2(rect2.center)
                            if overlap_vec.length_squared() > 0:  # Avoid division by zero if centers overlap exactly
                                dx = (rect1.width + rect2.width) / 2 - abs(overlap_vec.x)
                                dy = (rect1.height + rect2.height) / 2 - abs(overlap_vec.y)
                                if dx > 0 and dy > 0:
                                    push = 0.5;
                                    sep_vec = pygame.math.Vector2()
                                    total_m = r1.total_mass + r2.total_mass
                                    if dx < dy:
                                        sep_vec.x = dx * (-1 if overlap_vec.x < 0 else 1)
                                    else:
                                        sep_vec.y = dy * (-1 if overlap_vec.y < 0 else 1)
                                    if total_m > 0.01:
                                        r1.pos += sep_vec * (r2.total_mass / total_m) * push
                                        r2.pos -= sep_vec * (r1.total_mass / total_m) * push
                            # --- End Push Apart ---

                            # Break inner loops after handling collision for this p1
                            break  # Stop checking other p2 parts against this p1
                    # End of p2 loop

                    # If a collision was found involving p1, stop checking other p1 parts against r2
                    if collision_found_between_r1_r2:
                        break
                # End of p1 loop
            # End of r2 loop
        # End of r1 loop

        rockets_to_process = list(all_rockets) # --- Stage 2: State Changes ---
        for rocket in rockets_to_process:
            if rocket in rockets_to_remove_this_frame or not rocket.is_active: continue
            processed_split = False
            if rocket.needs_connectivity_check: # --- Destruction Split ---
                rocket.needs_connectivity_check = False; print(f"[{rocket.sim_instance_id}] Checking connectivity (Destruction)..."); temp_bp = RocketBlueprint(); temp_bp.parts = rocket.parts; subassemblies = temp_bp.find_connected_subassemblies()
                if len(subassemblies) > 1:
                    processed_split = True; print(f"[{rocket.sim_instance_id}] SPLIT (Destruction) into {len(subassemblies)}!");
                    if rocket not in rockets_to_remove_this_frame: rockets_to_remove_this_frame.append(rocket)
                    for assembly in subassemblies:
                         if not assembly: continue
                         try:
                             sub_com = rocket.calculate_subassembly_world_com(assembly); contains_root = rocket.original_root_part_ref in assembly; is_primary = rocket.has_active_control and contains_root
                             nr = FlyingRocket(list(assembly), sub_com, rocket.angle, rocket.vel, next_sim_id, is_primary, rocket.original_root_part_ref); nr.angular_velocity = rocket.angular_velocity; newly_created_rockets_this_frame.append(nr)
                             print(f"  > New {'controlled' if is_primary else 'debris'} R{next_sim_id} (dest. split)."); next_sim_id += 1
                         except Exception as e: print(f"Error creating from dest. split: {e}")
            if rocket.pending_separation and not processed_split: # --- Separator Split ---
                 print(f"[{rocket.sim_instance_id}] Processing {len(rocket.pending_separation)} separations..."); seps_activated = list(rocket.pending_separation); rocket.pending_separation.clear(); parts_start = list(rocket.parts); parts_remain = list(rocket.parts); split_by_sep = False
                 for sep_part in seps_activated:
                     if sep_part not in parts_remain: continue; print(f"  Processing sep: {sep_part.part_id}"); sep_pos = rocket.get_world_part_center(sep_part); sep_force = sep_part.part_data.get("separation_force", 1000)
                     check_list = [p for p in parts_remain if p != sep_part]; temp_bp = RocketBlueprint(); temp_bp.parts = check_list; subassemblies = temp_bp.find_connected_subassemblies()
                     if len(subassemblies) > 1:
                         split_by_sep = True; print(f"  > SPLIT by {sep_part.part_id} into {len(subassemblies)}!");
                         if rocket not in rockets_to_remove_this_frame: rockets_to_remove_this_frame.append(rocket)
                         for assembly in subassemblies:
                              if not assembly: continue
                              try:
                                  sub_com = rocket.calculate_subassembly_world_com(assembly); contains_root = rocket.original_root_part_ref in assembly; is_primary = rocket.has_active_control and contains_root
                                  nr = FlyingRocket(list(assembly), sub_com, rocket.angle, rocket.vel, next_sim_id, is_primary, rocket.original_root_part_ref); nr.angular_velocity = rocket.angular_velocity
                                  sep_vec = nr.get_world_com() - sep_pos; sep_dir = sep_vec.normalize() if sep_vec.length() > 0 else pygame.math.Vector2(0,-1).rotate(-rocket.angle)
                                  impulse = (sep_force / max(0.1, nr.total_mass)) * 0.05; delta_v = sep_dir * impulse; nr.vel += delta_v; print(f"    Applying impulse {delta_v.length():.1f} to new R{next_sim_id}")
                                  newly_created_rockets_this_frame.append(nr); print(f"    > New {'controlled' if is_primary else 'debris'} R{next_sim_id} (separation)."); next_sim_id += 1
                              except Exception as e: print(f"Error creating from sep. split: {e}")
                         break # Stop processing more seps for this original rocket
                     else: print(f"  > Sep {sep_part.part_id} caused no split."); parts_remain = check_list
                 if not split_by_sep and len(parts_remain) < len(parts_start):
                      print(f"[{rocket.sim_instance_id}] Updating parts after non-splitting sep(s)."); rocket.parts = parts_remain
                      rocket.engines = [e for e in rocket.engines if e in rocket.parts]; rocket.fuel_tanks = [t for t in rocket.fuel_tanks if t in rocket.parts]; rocket.parachutes = [pc for pc in rocket.parachutes if pc in rocket.parts]; rocket.separators = [s for s in rocket.separators if s in rocket.parts]
                      if not rocket.parts: rocket.is_active = False; rockets_to_remove_this_frame.append(rocket)
                      else: rocket.calculate_physics_properties(); rocket.calculate_bounds()

        if newly_created_rockets_this_frame: # --- Stage 3: Apply Adds/Removes ---
            new_ctrl = None
            for nr in newly_created_rockets_this_frame: all_rockets.append(nr);
            if nr.has_active_control:
                 if controlled_rocket and controlled_rocket not in rockets_to_remove_this_frame: print(f"WARN: Multiple control! Old={controlled_rocket.sim_instance_id}, New={nr.sim_instance_id}"); controlled_rocket.has_active_control = False
                 new_ctrl = nr
            if new_ctrl: controlled_rocket = new_ctrl
        if rockets_to_remove_this_frame:  # --- Stage 3: Apply Adds/Removes ---
            was_controlled_rocket_removed = controlled_rocket in rockets_to_remove_this_frame
            all_rockets = [r for r in all_rockets if r not in rockets_to_remove_this_frame]  # Filter list
            if was_controlled_rocket_removed:
                print("Controlled rocket instance removed/replaced.")
                controlled_rocket = None  # Clear current control
                # Attempt to find a new primary controlled rocket marked during split logic
                for rkt in all_rockets:
                    if rkt.has_active_control:
                        controlled_rocket = rkt
                        break  # Found one marked, stop searching
                # If none were marked (e.g., root part destruction didn't mark a new primary)
                if not controlled_rocket:
                    # Find the first remaining rocket that has the original root part and isn't broken
                    for rkt in all_rockets:
                        if rkt.original_root_part_ref and \
                                rkt.original_root_part_ref in rkt.parts and \
                                not rkt.original_root_part_ref.is_broken:
                            controlled_rocket = rkt
                            controlled_rocket.has_active_control = True  # Grant control
                            # *** FIX: Move break to its own line ***
                            break
                # Report outcome of control transfer search
                if controlled_rocket:
                    print(f"Control transferred to rocket instance {controlled_rocket.sim_instance_id}.")
                else:
                    print("No controllable rocket found after removals.")

        if controlled_rocket: camera.update(controlled_rocket.get_world_com()) # --- Stage 4: Update Camera ---
        elif all_rockets: camera.update(all_rockets[0].get_world_com())

        screen.fill(BLACK) # --- Stage 5: Drawing ---
        try: draw_earth_background(screen, camera, stars)
        except NameError: pass
        try: draw_terrain(screen, camera)
        except NameError: pass
        total_parts = 0; total_broken = 0
        for rocket in all_rockets:
            if rocket.is_active: broken = rocket.draw(screen, camera); total_parts += len(rocket.parts); total_broken += broken

        if controlled_rocket: # --- UI Overlay ---
            bar_width=20; bar_height=100; bar_x=15; bar_y=SCREEN_HEIGHT-bar_height-40; pygame.draw.rect(screen,(50,50,50),(bar_x,bar_y,bar_width,bar_height)); fill_h=bar_height*controlled_rocket.throttle_level; pygame.draw.rect(screen,(0,200,0),(bar_x,bar_y+bar_height-fill_h,bar_width,fill_h)); pygame.draw.rect(screen,WHITE,(bar_x,bar_y,bar_width,bar_height),1); th_txt=ui_font.render("Thr",True,WHITE); screen.blit(th_txt,(bar_x,bar_y+bar_height+5)); th_val=ui_font.render(f"{controlled_rocket.throttle_level*100:.0f}%",True,WHITE); screen.blit(th_val,(bar_x,bar_y-18))
            alt_agl=max(0,GROUND_Y-controlled_rocket.get_lowest_point_world().y); alt_msl=GROUND_Y-controlled_rocket.get_world_com().y; cs="OK" if controlled_rocket.has_active_control else "NO CTRL"; mts="ON" if controlled_rocket.master_thrust_enabled else "OFF"; ls="LANDED" if controlled_rocket.landed else "FLYING"
            st=[f"Alt(AGL): {alt_agl:.1f}m",f"Alt(MSL): {alt_msl:.1f}m",f"Vvel: {controlled_rocket.vel.y:.1f}",f"Hvel: {controlled_rocket.vel.x:.1f}",f"Speed: {controlled_rocket.vel.length():.1f}",f"Angle: {controlled_rocket.angle:.1f}",f"AngVel: {controlled_rocket.angular_velocity:.1f}",f"Thr: {controlled_rocket.throttle_level*100:.0f}% [{mts}]",f"Fuel: {controlled_rocket.current_fuel:.1f}",f"Mass: {controlled_rocket.total_mass:.1f}kg",f"Ctrl: {cs}",f"Status: {ls}",f"Inst: {controlled_rocket.sim_instance_id}"]
            t_y=10; tc=WHITE if controlled_rocket.has_active_control else RED;
            for i,t in enumerate(st): screen.blit(ui_font.render(t,True,tc),(bar_x+bar_width+10,t_y+i*18))
        elif not all_rockets:
            dt_txt=ui_font_large.render("ALL ROCKETS DESTROYED",True,RED); tr=dt_txt.get_rect(center=(SCREEN_WIDTH//2,SCREEN_HEIGHT//2)); screen.blit(dt_txt,tr); rt_txt=ui_font.render("Press 'R' to Respawn", True, WHITE); rtr=rt_txt.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2+40)); screen.blit(rt_txt, rtr)
        fps=clock.get_fps(); screen.blit(ui_font.render(f"FPS: {fps:.1f}",True,WHITE),(SCREEN_WIDTH-80,10)); screen.blit(ui_font.render(f"Objs: {len(all_rockets)}",True,WHITE),(SCREEN_WIDTH-80,30))

        pygame.display.flip()

    print("--- Exiting Simulation ---")