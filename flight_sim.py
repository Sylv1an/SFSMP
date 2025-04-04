# flight_sim.py
import pygame
import math
import sys
import random
from rocket_data import RocketBlueprint, PlacedPart # To use the loaded blueprint
from parts import draw_part_shape # To draw individual parts

# --- Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 150, 0)
GRAY = (100, 100, 100)

# Physics Constants
GRAVITY = 9.81 * 8
ROTATION_SPEED = 200 # Degrees per second

# World Constants
GROUND_Y = 1000
WORLD_WIDTH = 5000
STAR_COUNT = 200
STAR_FIELD_DEPTH = 10000

# --- Camera Class (same as before) ---
class Camera:
    def __init__(self, width, height):
        self.camera_rect = pygame.Rect(0, 0, width, height)
        self.width = width
        self.height = height
        self.offset = pygame.math.Vector2(0, 0)

    def apply(self, target_pos):
        return target_pos - self.offset

    def apply_rect(self, target_rect):
        return target_rect.move(-self.offset.x, -self.offset.y)

    def update(self, target_pos): # Update based on a position vector now
        x = target_pos.x - self.width // 2
        y = target_pos.y - self.height // 2
        self.offset = pygame.math.Vector2(x, y)

# --- FlyingRocket Class ---
# Represents the rocket *during flight*, using a blueprint
class FlyingRocket:
    def __init__(self, blueprint: RocketBlueprint, initial_pos, initial_angle=0):
        self.blueprint = blueprint
        self.pos = pygame.math.Vector2(initial_pos) # World position of the ROOT part (usually pod)
        self.vel = pygame.math.Vector2(0, 0)
        self.acc = pygame.math.Vector2(0, 0)
        self.angle = initial_angle # Overall rocket angle (degrees, 0=Up)
        self.angular_velocity = 0 # Degrees per second (for later use)

        self.thrusting = False
        self.landed = False

        # --- Calculate initial properties from blueprint ---
        self.engines = self.blueprint.get_engines()
        self.fuel_tanks = self.blueprint.get_fuel_tanks()
        self.current_fuel = self.blueprint.get_total_fuel_capacity()
        self.dry_mass = self.blueprint.get_total_mass()
        # Assume fuel mass is proportional to capacity for now (e.g., 1 unit capacity = 0.1 mass unit)
        self.fuel_mass_per_unit = 0.1
        self.update_total_mass()

        # Center of Mass calculation is complex - start simply
        # For now, assume CoM is the root part's position relative to itself (i.e., 0,0)
        self.center_of_mass_offset = pygame.math.Vector2(0, 0) # Relative to root part pos

        # Bounding box for collision (simplistic) - find min/max extents
        self.calculate_bounds()


    def calculate_bounds(self):
        # Find the rough extents of the rocket based on part positions and sizes
        # This is a simple AABB (Axis-Aligned Bounding Box) relative to the root part's origin
        min_x, max_x, min_y, max_y = 0, 0, 0, 0
        for part in self.blueprint.parts:
            half_w = part.part_data['width'] / 2
            half_h = part.part_data['height'] / 2
            px, py = part.relative_pos.x, part.relative_pos.y
            # Check corners relative to part center
            for dx in [-half_w, half_w]:
                for dy in [-half_h, half_h]:
                    # For now, ignore relative angle of parts for simplicity
                    check_x = px + dx
                    check_y = py + dy
                    min_x = min(min_x, check_x)
                    max_x = max(max_x, check_x)
                    min_y = min(min_y, check_y)
                    max_y = max(max_y, check_y)
        # Store dimensions relative to the root part's position/angle
        self.local_bounds = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)


    def get_world_bounds(self):
         # Get the world-coordinate bounding box (approximate, ignores rotation for now)
         # TODO: Rotate the bounding box properly
         world_rect = self.local_bounds.copy()
         world_rect.center = self.pos # Center the bounding box on the rocket's root position
         return world_rect

    def get_lowest_point_world(self):
        # Find the lowest point on the rocket in world coordinates, considering rotation
        lowest_y = -float('inf')
        lowest_point_world = self.pos # Default to root position

        for part in self.blueprint.parts:
            w = part.part_data['width']
            h = part.part_data['height']
            # Part center relative to root part, rotated by main rocket angle
            rotated_rel_pos = part.relative_pos.rotate(-self.angle)
            part_center_world = self.pos + rotated_rel_pos

            # Consider part's own angle relative to rocket body (if implemented)
            part_angle_total = self.angle + part.relative_angle

            # Check corners of the part in world space
            corners_rel_part = [
                pygame.math.Vector2(-w/2, h/2), # Bottom-left
                pygame.math.Vector2( w/2, h/2), # Bottom-right
            ]
            for corner in corners_rel_part:
                 # Rotate corner relative to part center, then add world center pos
                 rotated_corner_offset = corner.rotate(-part_angle_total)
                 corner_world = part_center_world + rotated_corner_offset
                 if corner_world.y > lowest_y:
                     lowest_y = corner_world.y
                     lowest_point_world = corner_world # Store the actual point vector

        # This calculation is complex, simplify if needed!
        # A simpler way: just check the bottom of the rotated local_bounds
        rotated_bottom_offset = pygame.math.Vector2(0, self.local_bounds.bottom).rotate(-self.angle)
        approx_bottom_world = self.pos + rotated_bottom_offset
        # return approx_bottom_world.y # Return just the Y coordinate

        # Use the more complex calculation for now
        if lowest_y == -float('inf'): # Should not happen if there are parts
             rotated_bottom_offset = pygame.math.Vector2(0, self.blueprint.parts[0].part_data['height']/2).rotate(-self.angle)
             return self.pos + rotated_bottom_offset # Fallback to root part bottom

        return lowest_point_world # Return the Vector2 of the lowest point


    def update_total_mass(self):
        self.total_mass = self.dry_mass + self.current_fuel * self.fuel_mass_per_unit
        if self.total_mass <= 0: self.total_mass = 0.01 # Avoid division by zero

    def apply_force(self, force_vector, point_of_application_offset=None):
        # F = ma -> a = F / m
        self.acc += force_vector / self.total_mass

        # Add torque later if force is not applied at center of mass
        # if point_of_application_offset:
        #    torque = point_of_application_offset.cross(force_vector)
        #    self.apply_torque(torque)

    # def apply_torque(self, torque):
    #     # Torque = I * alpha -> alpha = Torque / I
    #     # I = moment of inertia (complex to calculate for composite body)
    #     # Assume constant moment of inertia for now
    #     moment_of_inertia = self.total_mass * 100 # Placeholder! Needs calculation
    #     angular_acceleration = torque / moment_of_inertia
    #     self.angular_velocity += angular_acceleration * dt # dt needs to be passed in or accessible

    def consume_fuel(self, amount):
        consumed = min(self.current_fuel, amount)
        self.current_fuel -= consumed
        self.update_total_mass()
        return consumed > 0 # Return true if fuel was available

    def thrust(self, dt):
        total_thrust_force = 0
        total_consumption = 0
        fuel_available = self.current_fuel > 0

        if not fuel_available:
            self.thrusting = False
            return

        # Calculate total potential thrust from all active engines
        for engine_part in self.engines:
            # TODO: Check if engine is enabled/activated
            total_thrust_force += engine_part.part_data.get("thrust", 0)
            total_consumption += engine_part.part_data.get("fuel_consumption", 0)

        if total_thrust_force > 0:
             # Consume fuel based on potential thrust time
             actual_consumption = total_consumption * dt
             if self.consume_fuel(actual_consumption):
                 self.thrusting = True

                 # Apply thrust force along the rocket's main axis
                 # Use the CORRECTED direction calculation from the bugfix
                 rad_angle = math.radians(self.angle)
                 thrust_direction = pygame.math.Vector2(-math.sin(rad_angle), -math.cos(rad_angle))
                 thrust_vector = thrust_direction * total_thrust_force # This is force, not acceleration yet

                 # Apply force (will be divided by mass inside apply_force)
                 # TODO: Apply thrust at the engine's location for torque calculation later
                 self.apply_force(thrust_vector)
             else:
                 # Ran out of fuel mid-frame
                 self.thrusting = False
        else:
            self.thrusting = False


    def rotate(self, direction, dt):
        # Simple direct angle change for now, ignores torque/inertia
        self.angle = (self.angle + direction * ROTATION_SPEED * dt) % 360
        # Later: apply torque to change self.angular_velocity

    def update(self, dt):
        self.thrusting = False # Reset each frame
        self.landed = False
        # Store previous velocity for collision response? (optional)
        # prev_vel = self.vel.copy()

        # --- Handle Input ---
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.thrust(dt) # Consumes fuel, applies force if fuel available
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.rotate(-1, dt) # CCW rotation
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.rotate(1, dt)  # CW rotation

        # --- Apply Physics ---
        # 1. Apply gravity (acts on center of mass, approx root pos for now)
        gravity_force = pygame.math.Vector2(0, GRAVITY * self.total_mass) # Force = mass * g
        self.apply_force(gravity_force)

        # 2. Update velocity (v = v0 + a * dt)
        self.vel += self.acc * dt

        # 3. Update position (p = p0 + v * dt)
        self.pos += self.vel * dt

        # --- Collision Detection & Response ---
        # Use the calculated lowest point
        lowest_point = self.get_lowest_point_world()
        if lowest_point.y >= GROUND_Y:
            self.landed = True
            # Correct position: Move the entire rocket up so the lowest point is on the ground
            correction = lowest_point.y - GROUND_Y
            self.pos.y -= correction

            # Simple collision response: Dampen velocity significantly
            self.vel.y = 0
            self.vel.x *= 0.8 # Friction
            # Stop rotation on landing (can be more complex)
            self.angular_velocity = 0

            # Ensure we don't accumulate gravity force while landed
            # One way: if landed, counteract gravity in apply_force or here
            if self.acc.y > 0: # If accelerating downwards due to gravity
                 # Apply an upward force equal to gravity to stay put (simplistic ground reaction)
                 # This isn't quite right physically but prevents sinking
                 # A better way involves constraint solvers or checking 'grounded' state before applying gravity.
                 # For now, simply zeroing velocity is the main effect.
                 pass # Just zeroing velocity above is often enough visually

        # --- Reset Acceleration for next frame ---
        self.acc = pygame.math.Vector2(0, 0)

        # Update angular position based on angular velocity (if implemented)
        # self.angle = (self.angle + self.angular_velocity * dt) % 360

    def draw(self, surface, camera):
        # Draw each part relative to the rocket's main position and angle
        for part in self.blueprint.parts:
            # 1. Rotate the part's relative position around the root origin (0,0)
            rotated_rel_pos = part.relative_pos.rotate(-self.angle)
            # 2. Calculate the part's absolute world position
            part_world_pos = self.pos + rotated_rel_pos
            # 3. Calculate the part's final screen position
            part_screen_pos = camera.apply(part_world_pos)
            # 4. Calculate the part's total angle (rocket angle + part's relative angle)
            part_total_angle = self.angle + part.relative_angle

            # 5. Draw the part using its drawing function
            # Pass the calculated screen center position and total angle
            draw_part_shape(surface, part.part_data, part_screen_pos, part_total_angle)

            # Optional: Draw attachment points for debugging
            # for name, ap_rel in part.part_data["attachment_points"].items():
            #      # Rotate ap relative to part center, then add part center world pos
            #      abs_ap_offset = ap_rel.rotate(-part_total_angle)
            #      ap_world_pos = part_world_pos + abs_ap_offset
            #      ap_screen_pos = camera.apply(ap_world_pos)
            #      pygame.draw.circle(surface, RED, ap_screen_pos, 3)


        # Draw flame if thrusting (position relative to engines)
        if self.thrusting:
            for engine_part in self.engines:
                 # Calculate world position of the engine's bottom center
                 rotated_rel_pos = engine_part.relative_pos.rotate(-self.angle)
                 engine_center_world = self.pos + rotated_rel_pos
                 engine_total_angle = self.angle + engine_part.relative_angle

                 # Assume flame comes from bottom center of engine part data
                 flame_base_offset_local = pygame.math.Vector2(0, engine_part.part_data["height"] / 2)
                 rotated_flame_base_offset = flame_base_offset_local.rotate(-engine_total_angle)
                 flame_base_world = engine_center_world + rotated_flame_base_offset

                 # Flame dimensions (relative to direction)
                 flame_length = 25
                 flame_width = engine_part.part_data["width"] * 0.9

                 # Calculate flame points in world space
                 flame_dir_world = pygame.math.Vector2(0, 1).rotate(-engine_total_angle) # Vector pointing "down" from engine
                 flame_side_world = pygame.math.Vector2(1, 0).rotate(-engine_total_angle) # Vector pointing "right" from engine

                 flame_tip_world = flame_base_world + flame_dir_world * flame_length
                 flame_left_world = flame_base_world - flame_side_world * flame_width / 2
                 flame_right_world = flame_base_world + flame_side_world * flame_width / 2

                 # Convert to screen points
                 flame_points_screen = [
                     camera.apply(flame_left_world),
                     camera.apply(flame_right_world),
                     camera.apply(flame_tip_world)
                 ]
                 pygame.draw.polygon(surface, RED, flame_points_screen)


# --- Background/Terrain Functions (mostly same as before) ---
def create_stars(count, world_bounds_rect):
    # ... (same as before) ...
    stars = []
    for _ in range(count):
        x = random.uniform(world_bounds_rect.left, world_bounds_rect.right)
        y = random.uniform(world_bounds_rect.top, world_bounds_rect.bottom)
        z = random.uniform(1, STAR_FIELD_DEPTH) # Depth
        stars.append((pygame.math.Vector2(x, y), z))
    return stars


def draw_stars(surface, stars, camera):
    # ... (same as before, maybe adjust brightness/size calculation) ...
    center_world = camera.offset + pygame.math.Vector2(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
    for star_pos_world, z in stars:
        parallax_factor = 1.0 / (z / 100 + 1)
        star_view_pos = center_world + (star_pos_world - center_world) * parallax_factor
        star_screen_pos = camera.apply(star_view_pos)
        if 0 <= star_screen_pos.x < SCREEN_WIDTH and 0 <= star_screen_pos.y < SCREEN_HEIGHT:
            brightness = max(50, 200 * (1.0 - z / STAR_FIELD_DEPTH))
            color = (int(brightness), int(brightness), int(brightness))
            size = max(1, int(2 * (1.0 - z / STAR_FIELD_DEPTH)))
            pygame.draw.circle(surface, color, (int(star_screen_pos.x), int(star_screen_pos.y)), size)


def draw_terrain(surface, camera):
    # ... (same as before) ...
    ground_view_rect = pygame.Rect(
        camera.offset.x - 50,
        GROUND_Y,
        camera.width + 100,
        SCREEN_HEIGHT # Depth
    )
    ground_rect_screen = camera.apply_rect(ground_view_rect)
    pygame.draw.rect(surface, GREEN, ground_rect_screen)

# --- Simulation Runner ---
def run_simulation(screen, clock, blueprint_file):
    blueprint = RocketBlueprint.load_from_json(blueprint_file)
    if not blueprint:
        print("Failed to load blueprint for simulation.")
        return # Go back to menu or handle error

    # --- Setup Simulation Specifics ---
    # Start rocket above ground, centered horizontally in the world view
    start_x = 0 # Use world coordinates
    start_y = GROUND_Y - 300 # Start well above ground
    player_rocket = FlyingRocket(blueprint, (start_x, start_y))

    camera = Camera(SCREEN_WIDTH, SCREEN_HEIGHT)
    camera.update(player_rocket.pos) # Initial camera position

    star_area = pygame.Rect(-WORLD_WIDTH, -SCREEN_HEIGHT * 2, WORLD_WIDTH * 2, GROUND_Y + SCREEN_HEIGHT * 3)
    stars = create_stars(STAR_COUNT, star_area)

    # --- Simulation Loop ---
    sim_running = True
    while sim_running:
        dt = clock.tick(60) / 1000.0
        dt = min(dt, 0.1) # Clamp dt

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit() # Quit entire application
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    sim_running = False # Exit simulation, back to main menu

        # --- Logic Update ---
        player_rocket.update(dt)
        camera.update(player_rocket.pos) # Camera follows the root part's position

        # --- Drawing ---
        screen.fill(BLACK)
        draw_stars(screen, stars, camera)
        draw_terrain(screen, camera)
        player_rocket.draw(screen, camera)

        # --- UI Overlay (Fuel, Velocity, etc.) ---
        font = pygame.font.SysFont(None, 24)
        fuel_text = font.render(f"Fuel: {player_rocket.current_fuel:.1f}", True, WHITE)
        vel_text = font.render(f"Vel: ({player_rocket.vel.x:.1f}, {player_rocket.vel.y:.1f})", True, WHITE)
        alt_text = font.render(f"Alt (AGL): {GROUND_Y - player_rocket.get_lowest_point_world().y:.1f}", True, WHITE) # Altitude above ground level
        screen.blit(fuel_text, (10, 10))
        screen.blit(vel_text, (10, 30))
        screen.blit(alt_text, (10, 50))

        # --- Display Update ---
        pygame.display.flip()

    # Simulation loop ended (e.g., ESC pressed), return to caller (main menu)