# builder.py
import pygame
import sys
import math
import os
# Use the modified parts functions
from parts import PARTS_CATALOG, get_part_data, draw_part_shape
from rocket_data import RocketBlueprint, PlacedPart, AMBIENT_TEMPERATURE
# Use new state constants if needed, Button already imported
from ui_elements import (Button, STATE_MAIN_MENU, STATE_MULTIPLAYER_LOBBY,
                         SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, BLACK, GRAY, LIGHT_GRAY, BLUE, RED, GREEN)

# --- Constants ---
BUILD_BG_COLOR = (40, 40, 60)
PLACEMENT_OK_COLOR = GREEN
PLACEMENT_BLOCKED_COLOR = RED
GRID_COLOR = (60, 60, 80)
GRID_SIZE = 10 # Pixels per grid cell
BUILD_AREA_BORDER = 5
ATTACHMENT_SNAP_DISTANCE_SQ = (GRID_SIZE * 1.7)**2 # How close mouse needs to be (squared)

# --- Builder Layout ---
PARTS_LIST_WIDTH = 180
PARTS_LIST_RECT = pygame.Rect(0, 0, PARTS_LIST_WIDTH, SCREEN_HEIGHT)
BUILD_AREA_RECT = pygame.Rect(PARTS_LIST_WIDTH + BUILD_AREA_BORDER, 0, SCREEN_WIDTH - (PARTS_LIST_WIDTH + BUILD_AREA_BORDER), SCREEN_HEIGHT)
BUTTON_AREA_HEIGHT = 50
BUTTON_AREA_RECT = pygame.Rect(BUILD_AREA_RECT.left, SCREEN_HEIGHT - BUTTON_AREA_HEIGHT, BUILD_AREA_RECT.width, BUTTON_AREA_HEIGHT)

# --- Builder State Variables ---
selected_part_id = None
current_blueprint = RocketBlueprint("Unnamed Build")
build_offset = pygame.math.Vector2(BUILD_AREA_RECT.centerx, BUILD_AREA_RECT.centery - 100)
build_zoom = 1.0
builder_buttons = {}
part_list_buttons = {}
TEMP_BLUEPRINT_FILE = os.path.join("assets", "current_build.json")

# --- Helper Functions (Unchanged) ---
# screen_to_world, world_to_screen, snap_to_grid, draw_grid, draw_blueprint
# get_part_world_rect, check_placement_collision, find_potential_attachment
# --- All helpers remain the same ---
def screen_to_world(screen_pos: pygame.math.Vector2, offset: pygame.math.Vector2, zoom: float) -> pygame.math.Vector2:
    return screen_pos - offset # Basic version without zoom

def world_to_screen(world_pos: pygame.math.Vector2, offset: pygame.math.Vector2, zoom: float) -> pygame.math.Vector2:
    return world_pos + offset # Basic version without zoom

def snap_to_grid(world_pos: pygame.math.Vector2) -> pygame.math.Vector2:
    grid_x = round(world_pos.x / GRID_SIZE)
    grid_y = round(world_pos.y / GRID_SIZE)
    return pygame.math.Vector2(grid_x * GRID_SIZE, grid_y * GRID_SIZE)

def draw_grid(surface, offset, zoom, view_rect, grid_size):
    start_col = math.floor((view_rect.left - offset.x) / grid_size)
    end_col = math.ceil((view_rect.right - offset.x) / grid_size)
    start_row = math.floor((view_rect.top - offset.y) / grid_size)
    end_row = math.ceil((view_rect.bottom - offset.y) / grid_size)

    for col in range(start_col, end_col + 1):
        x = offset.x + col * grid_size
        if view_rect.left <= x <= view_rect.right:
             pygame.draw.line(surface, GRID_COLOR, (x, view_rect.top), (x, view_rect.bottom))
    for row in range(start_row, end_row + 1):
        y = offset.y + row * grid_size
        if view_rect.top <= y <= view_rect.bottom:
            pygame.draw.line(surface, GRID_COLOR, (view_rect.left, y), (view_rect.right, y))

def draw_blueprint(surface, blueprint: RocketBlueprint, offset, zoom):
    if not blueprint.parts: return
    for part in blueprint.parts:
        part_data = part.part_data
        part_world_center = part.relative_pos
        part_screen_center = world_to_screen(part_world_center, offset, zoom)
        # Basic Culling (Only draw if roughly on screen)
        draw_radius = max(part_data.get('width', 10), part_data.get('height', 10)) # Approximate radius
        if BUILD_AREA_RECT.inflate(draw_radius*2, draw_radius*2).collidepoint(part_screen_center):
             # Use existing draw_part_shape which handles state (though not relevant in builder)
             draw_part_shape(surface, part_data, part_screen_center, part.relative_angle,
                             broken=part.is_broken, deployed=part.deployed, heat_factor=0.0)


def get_part_world_rect(part: PlacedPart):
    part_data = part.part_data; w=part_data['width']; h=part_data['height']
    world_rect = pygame.Rect(0, 0, w, h); world_rect.center = part.relative_pos
    return world_rect

def check_placement_collision(blueprint: RocketBlueprint, new_part_id: str, new_part_world_center: pygame.math.Vector2, attaching_to_part: PlacedPart = None):
    if not blueprint.parts: return False
    try: temp_new_part = PlacedPart(new_part_id, new_part_world_center)
    except ValueError: return True # Invalid part ID means blocked
    new_part_rect = get_part_world_rect(temp_new_part)
    # Check collision against existing parts
    for existing_part in blueprint.parts:
        if existing_part is attaching_to_part: continue # Don't collide with the part we are attaching to
        existing_part_rect = get_part_world_rect(existing_part)
        # Use inflate slightly to prevent parts visually touching but logically not colliding
        if new_part_rect.inflate(1, 1).colliderect(existing_part_rect):
            return True # Collision detected
    return False # No collision

def find_potential_attachment(blueprint: RocketBlueprint, part_to_place_data: dict, target_screen_pos: pygame.math.Vector2, offset: pygame.math.Vector2, zoom: float) -> tuple[PlacedPart | None, str | None, str | None, pygame.math.Vector2 | None]:
    target_world_pos = screen_to_world(target_screen_pos, offset, zoom)
    closest_dist_sq = ATTACHMENT_SNAP_DISTANCE_SQ
    best_match = (None, None, None, None) # (existing_part, rule_on_existing, rule_on_new, connection_world_pos)
    if not blueprint.parts: return best_match # Cannot attach if no parts exist

    part_to_place_type = part_to_place_data.get("type")
    part_to_place_rules = part_to_place_data.get("attachment_rules", {})
    part_to_place_points = part_to_place_data.get("logical_points", {}) # Need points of new part

    for existing_part in blueprint.parts:
        existing_part_data = existing_part.part_data
        existing_part_rules = existing_part_data.get("attachment_rules", {})
        existing_part_points = existing_part_data.get("logical_points", {})

        for point_name, local_pos in existing_part_points.items():
            # Calculate world position of the attachment point on the existing part
            point_world_pos = existing_part.relative_pos + local_pos.rotate(-existing_part.relative_angle)
            dist_sq = (point_world_pos - target_world_pos).length_squared()

            if dist_sq < closest_dist_sq:
                 # Map existing part's point name to its corresponding rule name
                 rule_name_existing = blueprint._get_rule_name_for_point(point_name, existing_part_rules)
                 if not rule_name_existing: continue # Skip if point doesn't map to a rule

                 # Check if the new part type is allowed by the existing part's rule
                 allowed_types_on_existing = existing_part_rules[rule_name_existing].get("allowed_types", [])
                 if part_to_place_type not in allowed_types_on_existing:
                      continue # New part type not allowed here

                 # Find the compatible rule name on the *new* part
                 compatible_new_rule_name = None
                 if "bottom" in rule_name_existing and "top_center" in part_to_place_rules: compatible_new_rule_name = "top_center"
                 elif "top" in rule_name_existing and "bottom_center" in part_to_place_rules: compatible_new_rule_name = "bottom_center"
                 elif "left" in rule_name_existing and "right_center" in part_to_place_rules: compatible_new_rule_name = "right_center"
                 elif "right" in rule_name_existing and "left_center" in part_to_place_rules: compatible_new_rule_name = "left_center"
                 # Add more compatibility checks if needed (front/back etc.)

                 if compatible_new_rule_name:
                      # Final check: Does the *new* part's compatible rule allow the *existing* part's type?
                      allowed_types_on_new = part_to_place_rules.get(compatible_new_rule_name, {}).get("allowed_types", [])
                      if existing_part_data.get("type") in allowed_types_on_new:
                          # Found a valid attachment!
                          closest_dist_sq = dist_sq
                          best_match = (existing_part, rule_name_existing, compatible_new_rule_name, point_world_pos)

    return best_match


def setup_builder_ui(is_multiplayer=False): # Added parameter
    global builder_buttons, part_list_buttons
    builder_buttons = {}; part_list_buttons = {}
    button_width = 120; button_height = 35
    start_x = BUTTON_AREA_RECT.left + 20
    y_pos = BUTTON_AREA_RECT.top + (BUTTON_AREA_RECT.height - button_height) // 2

    # Change Launch button text in MP
    launch_text = "Launch Ready" if is_multiplayer else "Launch"
    builder_buttons["launch"] = Button(launch_text, (start_x, y_pos, button_width + (20 if is_multiplayer else 0), button_height))

    builder_buttons["clear"] = Button("Clear", (start_x + button_width + (30 if is_multiplayer else 10), y_pos, button_width, button_height))
    builder_buttons["back"] = Button("Back", (BUTTON_AREA_RECT.right - button_width - 20, y_pos, button_width, button_height)) # Simpler Back button

    # Part list buttons (unchanged)
    part_y = 20; part_button_height = 40; part_font_size = 18
    for part_id, part_data in PARTS_CATALOG.items():
        rect = pygame.Rect(10, part_y, PARTS_LIST_WIDTH - 20, part_button_height)
        part_list_buttons[part_id] = Button(part_data["name"], rect, font_size=part_font_size, color=(80,80,90), hover_color=(110,110,120))
        part_y += part_button_height + 5


# --- Main Builder Loop ---
# Added is_multiplayer parameter
def run_builder(screen, clock, is_multiplayer=False):
    global selected_part_id, current_blueprint, build_offset
    # Load or create blueprint
    if os.path.exists(TEMP_BLUEPRINT_FILE):
        loaded_bp = RocketBlueprint.load_from_json(TEMP_BLUEPRINT_FILE)
        # Ensure runtime state is reset when loading for builder
        if loaded_bp:
             for part in loaded_bp.parts:
                 # Reset state that might persist from flight sim (though blueprint shouldn't save it)
                 part.current_hp = part.part_data.get("max_hp", 100)
                 part.is_broken = False
                 part.engine_enabled = True
                 part.deployed = False
                 part.separated = False
                 part.current_temp = AMBIENT_TEMPERATURE
                 part.is_overheating = False
                 # Reset fuel to capacity in builder
                 if part.part_data.get("type") == "FuelTank":
                     part.current_fuel = part.fuel_capacity
             current_blueprint = loaded_bp
        else:
             current_blueprint = RocketBlueprint("My Grid Rocket") # Fallback if load fails
    else: current_blueprint = RocketBlueprint("My Grid Rocket")

    # Save initial state (or freshly loaded state)
    current_blueprint.save_to_json(TEMP_BLUEPRINT_FILE)

    selected_part_id = None
    setup_builder_ui(is_multiplayer) # Pass MP flag to UI setup
    building = True
    placement_blocked = False
    ghost_draw_pos_screen = None
    potential_relative_pos = None
    is_attaching = False
    attaching_to_part = None
    dragging_view = False
    drag_start_mouse = None
    drag_start_offset = None

    while building:
        dt = clock.tick(60) / 1000.0
        mouse_pos_tuple = pygame.mouse.get_pos()
        mouse_pos = pygame.math.Vector2(mouse_pos_tuple)

        # Reset state
        placement_blocked = False; ghost_draw_pos_screen = None
        potential_relative_pos = None; is_attaching = False; attaching_to_part = None

        # --- Calculate potential placement ---
        if selected_part_id and BUILD_AREA_RECT.collidepoint(mouse_pos):
            selected_part_data = get_part_data(selected_part_id)
            if not selected_part_data: # Handle case where selected_part_id becomes invalid
                 selected_part_id = None
                 continue

            if current_blueprint.parts:
                attach_result = find_potential_attachment(current_blueprint, selected_part_data, mouse_pos, build_offset, build_zoom)
                target_part, target_area_rule, new_part_area_rule, connection_world_pos = attach_result

                if target_part and connection_world_pos: # --- ATTACHED PLACEMENT PREVIEW ---
                    is_attaching = True; attaching_to_part = target_part
                    new_part_points = selected_part_data.get("logical_points", {})

                    # Map the *compatible rule name* on the new part back to its *point name*
                    new_part_point_name = None
                    if new_part_area_rule == "top_center": new_part_point_name = "top"
                    elif new_part_area_rule == "bottom_center": new_part_point_name = "bottom"
                    elif new_part_area_rule == "left_center": new_part_point_name = "left"
                    elif new_part_area_rule == "right_center": new_part_point_name = "right"
                    # Add other mappings if needed

                    if new_part_point_name and new_part_point_name in new_part_points:
                        # Get the local offset of the connection point on the new part (relative to its center)
                        new_part_conn_local_offset = new_part_points[new_part_point_name]

                        # Calculate where the new part's *center* needs to be in world space
                        # Assume angle 0 for placement preview (rotation handled later if implemented)
                        # WorldCenter = WorldConnectionPoint - LocalOffsetOfConnectionPointOnNewPart
                        potential_center_world = connection_world_pos - new_part_conn_local_offset.rotate(0)

                        # Store the calculated relative position (relative to blueprint 0,0)
                        potential_relative_pos = potential_center_world
                        ghost_draw_pos_screen = world_to_screen(potential_relative_pos, build_offset, build_zoom)
                        # Check collision against other parts (excluding the one we're attaching to)
                        placement_blocked = check_placement_collision(current_blueprint, selected_part_id, potential_relative_pos, attaching_to_part=attaching_to_part)
                    else:
                        # Cannot map compatible area rule back to a logical point on the new part, block placement
                        placement_blocked = True
                        # Show ghost snapped to grid as fallback preview
                        mouse_world_pos = screen_to_world(mouse_pos, build_offset, build_zoom)
                        potential_relative_pos = snap_to_grid(mouse_world_pos)
                        ghost_draw_pos_screen = world_to_screen(potential_relative_pos, build_offset, build_zoom)

                else: # --- FREE PLACEMENT PREVIEW ---
                    is_attaching = False; attaching_to_part = None
                    mouse_world_pos = screen_to_world(mouse_pos, build_offset, build_zoom)
                    potential_relative_pos = snap_to_grid(mouse_world_pos) # Snap to grid
                    ghost_draw_pos_screen = world_to_screen(potential_relative_pos, build_offset, build_zoom)
                    # Check collision against all existing parts
                    placement_blocked = check_placement_collision(current_blueprint, selected_part_id, potential_relative_pos)

            else: # --- ROOT PART PLACEMENT PREVIEW ---
                is_attaching = False; attaching_to_part = None
                # Snap root part's center to the blueprint origin (0,0)
                potential_relative_pos = pygame.math.Vector2(0, 0)
                ghost_draw_pos_screen = world_to_screen(potential_relative_pos, build_offset, build_zoom)
                placement_blocked = False # Cannot collide when placing the first part

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if selected_part_id:
                        selected_part_id = None; print("Deselected part via Escape.")
                    else:
                        building = False # Exit builder if Esc pressed with nothing selected

            if event.type == pygame.MOUSEBUTTONDOWN:
                click_pos = pygame.math.Vector2(event.pos)

                if event.button == 1: # Left Click
                    clicked_action = None; clicked_part_list_item = False
                    # Check UI buttons first (bottom area)
                    for name, button in builder_buttons.items():
                        if button.is_clicked(event): clicked_action = name; break
                    # Check part list buttons only if no action button was clicked
                    if not clicked_action:
                        for part_id, button in part_list_buttons.items():
                            if button.is_clicked(event):
                                selected_part_id = part_id; clicked_part_list_item = True; break

                    # Process Action Button Clicks
                    if clicked_action == "launch":
                         if current_blueprint.parts:
                            # Save blueprint before returning launch signal
                            current_blueprint.save_to_json(TEMP_BLUEPRINT_FILE)
                            # Return tuple indicating launch and the blueprint file used
                            return ("LAUNCH", TEMP_BLUEPRINT_FILE)
                         else: print("Cannot launch empty rocket!")
                    elif clicked_action == "clear":
                         current_blueprint = RocketBlueprint("Cleared Rocket") # Create new empty BP
                         current_blueprint.save_to_json(TEMP_BLUEPRINT_FILE) # Save the empty state
                         selected_part_id = None; print("Blueprint cleared.")
                    elif clicked_action == "back":
                        building = False # Exit builder loop

                    # Process Part Placement Click (if no UI button was clicked)
                    elif not clicked_part_list_item and BUILD_AREA_RECT.collidepoint(click_pos) and selected_part_id:
                        if potential_relative_pos is not None and not placement_blocked:
                            # Add the part to the blueprint
                            current_blueprint.add_part(selected_part_id, potential_relative_pos, angle=0) # Assuming angle 0 for now
                            print(f"Placed {selected_part_id} at rel {potential_relative_pos} {'(attached)' if is_attaching else '(free)'}")
                            # Autosave after placing
                            current_blueprint.save_to_json(TEMP_BLUEPRINT_FILE)
                            # Deselect part after placing
                            selected_part_id = None
                        else:
                             # Optional: Provide feedback why placement failed
                             print(f"Placement blocked for {selected_part_id}.")

                    # Process View Drag Start (if no part selected and clicking in build area)
                    elif not clicked_part_list_item and BUILD_AREA_RECT.collidepoint(click_pos) and not selected_part_id:
                         dragging_view = True; drag_start_mouse = mouse_pos; drag_start_offset = build_offset.copy()

                elif event.button == 3: # Right Click
                    # Deselect currently held part
                    if selected_part_id: selected_part_id = None; print("Deselected part.")
                    # Remove part if clicking in build area with nothing selected
                    elif BUILD_AREA_RECT.collidepoint(event.pos):
                         click_world_pos = screen_to_world(pygame.math.Vector2(event.pos), build_offset, build_zoom)
                         part_to_remove = current_blueprint.get_part_at_world_pos(click_world_pos)
                         if part_to_remove:
                             current_blueprint.remove_part(part_to_remove)
                             print(f"Removed part: {part_to_remove.part_id}")
                             # Autosave after removing
                             current_blueprint.save_to_json(TEMP_BLUEPRINT_FILE)

                elif event.button == 2: # Middle Mouse for drag start
                     if BUILD_AREA_RECT.collidepoint(event.pos):
                         dragging_view = True; drag_start_mouse = mouse_pos; drag_start_offset = build_offset.copy()

            if event.type == pygame.MOUSEBUTTONUP:
                 if event.button == 1 or event.button == 2: # Release drag (Left or Middle)
                     dragging_view = False; drag_start_mouse = None; drag_start_offset = None

        # --- Update ---
        if dragging_view and drag_start_mouse: # Handle view drag
            delta = mouse_pos - drag_start_mouse; build_offset = drag_start_offset + delta

        # Update button hover states
        for button in builder_buttons.values(): button.check_hover(mouse_pos_tuple)
        for pid, button in part_list_buttons.items():
             is_selected = (selected_part_id == pid)
             # Hover if mouse is over OR if the part is selected
             button.is_hovered = button.rect.collidepoint(mouse_pos_tuple) or is_selected

        # --- Drawing ---
        screen.fill(BUILD_BG_COLOR)
        # Build area background
        pygame.draw.rect(screen, (25, 25, 35), BUILD_AREA_RECT)
        # Draw grid (clipped to build area)
        draw_grid(screen, build_offset, build_zoom, BUILD_AREA_RECT, GRID_SIZE)
        # Draw placed parts
        draw_blueprint(screen, current_blueprint, build_offset, build_zoom)

        # Draw ghost preview if a part is selected and mouse is in build area
        if selected_part_id and ghost_draw_pos_screen:
            ghost_part_data = get_part_data(selected_part_id)
            if ghost_part_data: # Check if part data is valid
                 w, h = ghost_part_data['width'], ghost_part_data['height']
                 # Create a surface for the ghost with alpha
                 ghost_surf = pygame.Surface((w, h), pygame.SRCALPHA)
                 # Draw the part shape onto the ghost surface
                 # Pass dummy state values (broken=False, etc.) to draw_part_shape
                 draw_part_shape(ghost_surf, ghost_part_data, pygame.math.Vector2(w/2, h/2), 0, False, False, 0.0)
                 # Set base transparency
                 ghost_surf.set_alpha(150)
                 # Create color overlay based on placement validity
                 color_overlay = pygame.Surface((w, h), pygame.SRCALPHA)
                 ghost_color = PLACEMENT_OK_COLOR if not placement_blocked else PLACEMENT_BLOCKED_COLOR
                 # Fill overlay with semi-transparent color
                 color_overlay.fill((*ghost_color, 100))
                 # Blend the color overlay onto the ghost surface
                 ghost_surf.blit(color_overlay, (0,0), special_flags=pygame.BLEND_RGBA_MULT)
                 # Blit the final ghost surface onto the screen, centered at the calculated screen position
                 screen.blit(ghost_surf, ghost_draw_pos_screen - pygame.math.Vector2(w/2, h/2))

                 # Highlight potential attachment point
                 if is_attaching and attaching_to_part:
                      # Recalculate for drawing just to be sure? Or reuse calculated pos? Reuse is fine.
                      # attach_res_draw = find_potential_attachment(current_blueprint, selected_part_data, mouse_pos, build_offset, build_zoom)
                      # if attach_res_draw[0] and attach_res_draw[3]: # Check if still valid attachment found and has world pos
                      if potential_relative_pos: # Check if we have a potential position (means attachment calc worked)
                          connection_world_pos = potential_relative_pos + new_part_conn_local_offset.rotate(0) # Recalc world connection point
                          conn_screen_pos_draw = world_to_screen(connection_world_pos, build_offset, build_zoom)
                          highlight_color = WHITE if not placement_blocked else RED
                          pygame.draw.circle(screen, highlight_color, conn_screen_pos_draw, 5, 1) # Draw circle outline
            else:
                # Handle case where selected_part_id might be invalid somehow
                selected_part_id = None


        # Draw UI Areas (Parts List and Bottom Buttons)
        pygame.draw.rect(screen, (30, 30, 40), PARTS_LIST_RECT) # Parts list bg
        pygame.draw.line(screen, WHITE, (PARTS_LIST_WIDTH - 1, 0), (PARTS_LIST_WIDTH - 1, SCREEN_HEIGHT), 1) # Divider line
        # Parts list title
        list_font = pygame.font.SysFont(None, 24)
        list_title = list_font.render("Available Parts", True, WHITE)
        screen.blit(list_title, (10, 5))
        # Draw part buttons
        for button in part_list_buttons.values(): button.draw(screen)

        # Bottom button area background
        pygame.draw.rect(screen, (20, 20, 30), BUTTON_AREA_RECT)
        pygame.draw.line(screen, WHITE, (BUTTON_AREA_RECT.left, BUTTON_AREA_RECT.top), (BUTTON_AREA_RECT.right, BUTTON_AREA_RECT.top), 1) # Top line
        # Draw action buttons
        for button in builder_buttons.values(): button.draw(screen)

        pygame.display.flip()

    # --- Builder Exit ---
    # Save the final blueprint when exiting the builder
    current_blueprint.save_to_json(TEMP_BLUEPRINT_FILE)
    print("Exiting builder, blueprint saved.")
    # Return the appropriate state to go back to (Main Menu or MP Lobby)
    return STATE_MULTIPLAYER_LOBBY if is_multiplayer else STATE_MAIN_MENU