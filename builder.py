# builder.py
import pygame
import sys
import math
import os
# Use the modified parts functions
from parts import PARTS_CATALOG, get_part_data, draw_part_shape, can_attach # Removed get_part_attachment_areas as less needed now
from rocket_data import RocketBlueprint, PlacedPart
from ui_elements import Button, STATE_MAIN_MENU
from ui_elements import SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, BLACK, GRAY, LIGHT_GRAY, BLUE, RED, GREEN

# --- Constants ---
BUILD_BG_COLOR = (40, 40, 60)
PLACEMENT_OK_COLOR = GREEN
PLACEMENT_BLOCKED_COLOR = RED
GRID_COLOR = (60, 60, 80)
GRID_SIZE = 10 # Pixels per grid cell
BUILD_AREA_BORDER = 5 # Prevent parts list overlap visually
ATTACHMENT_SNAP_DISTANCE_SQ = (GRID_SIZE * 1.7)**2 # How close mouse needs to be to an attachment point (squared)

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
build_zoom = 1.0 # Keep for potential future use
builder_buttons = {}
part_list_buttons = {}
TEMP_BLUEPRINT_FILE = os.path.join("assets", "current_build.json")

# --- Helper Functions ---

def screen_to_world(screen_pos: pygame.math.Vector2, offset: pygame.math.Vector2, zoom: float) -> pygame.math.Vector2:
    return screen_pos - offset # Basic version without zoom

def world_to_screen(world_pos: pygame.math.Vector2, offset: pygame.math.Vector2, zoom: float) -> pygame.math.Vector2:
    return world_pos + offset # Basic version without zoom

def snap_to_grid(world_pos: pygame.math.Vector2) -> pygame.math.Vector2:
    """ Snaps world coordinates to the center of the nearest grid cell. """
    grid_x = round(world_pos.x / GRID_SIZE)
    grid_y = round(world_pos.y / GRID_SIZE)
    return pygame.math.Vector2(grid_x * GRID_SIZE, grid_y * GRID_SIZE)

def draw_grid(surface, offset, zoom, view_rect, grid_size):
    """ Draws the grid lines within the viewable area """
    start_col = math.floor((view_rect.left - offset.x) / grid_size)
    end_col = math.ceil((view_rect.right - offset.x) / grid_size)
    start_row = math.floor((view_rect.top - offset.y) / grid_size)
    end_row = math.ceil((view_rect.bottom - offset.y) / grid_size)

    for col in range(start_col, end_col + 1): # Ensure lines reach edge
        x = offset.x + col * grid_size
        if view_rect.left <= x <= view_rect.right:
             pygame.draw.line(surface, GRID_COLOR, (x, view_rect.top), (x, view_rect.bottom))

    for row in range(start_row, end_row + 1): # Ensure lines reach edge
        y = offset.y + row * grid_size
        if view_rect.top <= y <= view_rect.bottom:
            pygame.draw.line(surface, GRID_COLOR, (view_rect.left, y), (view_rect.right, y))

def draw_blueprint(surface, blueprint: RocketBlueprint, offset, zoom):
    if not blueprint.parts: return
    for part in blueprint.parts:
        part_data = part.part_data
        part_world_center = part.relative_pos # relative_pos IS the world center in blueprint coords
        part_screen_center = world_to_screen(part_world_center, offset, zoom)

        if BUILD_AREA_RECT.collidepoint(part_screen_center): # Basic check
             # Draw part (assuming relative_angle is 0 for grid build)
             draw_part_shape(surface, part_data, part_screen_center, part.relative_angle)

             # --- Optional: Draw logical points for debugging attachments ---
             # points = part_data.get("logical_points", {})
             # for name, local_pos in points.items():
             #     # Calculate world position of logical point
             #     rotated_offset = local_pos.rotate(-part.relative_angle)
             #     world_pos = part_world_center + rotated_offset
             #     screen_pos = world_to_screen(world_pos, offset, zoom)
             #     pygame.draw.circle(surface, BLUE, screen_pos, 3)
             #     font = pygame.font.SysFont(None, 15)
             #     text = font.render(name, True, WHITE)
             #     surface.blit(text, screen_pos + pygame.math.Vector2(5, -5))


# --- Collision Checking (remains the same, uses world coordinates) ---
def get_part_world_rect(part: PlacedPart):
    """ Calculates the world AABB based on part's relative_pos (world center). Ignores rotation. """
    part_data = part.part_data
    w = part_data['width']
    h = part_data['height']
    world_rect = pygame.Rect(0, 0, w, h)
    world_rect.center = part.relative_pos # Use the part's stored world center
    return world_rect

def check_placement_collision(blueprint: RocketBlueprint, new_part_id: str, new_part_world_center: pygame.math.Vector2, attaching_to_part: PlacedPart = None):
    """ Checks if placing a new part at a specific world center collides with others. """
    if not blueprint.parts: return False

    try:
        # Create a temporary PlacedPart at the target world location (angle 0)
        temp_new_part = PlacedPart(new_part_id, new_part_world_center)
    except ValueError: return True # Invalid part ID

    new_part_rect = get_part_world_rect(temp_new_part)

    for existing_part in blueprint.parts:
        # Skip check if attaching TO this part (allows touching at connection)
        if existing_part is attaching_to_part:
            continue

        existing_part_rect = get_part_world_rect(existing_part)
        # Use a small inflation to catch near-misses due to rounding/floating point
        if new_part_rect.inflate(1, 1).colliderect(existing_part_rect):
            # print(f"Collision: New {new_part_id} overlaps {existing_part.part_id}") # Debug
            return True # Collision found
    return False # No collision detected

# --- NEW HELPER: Find potential attachment based on proximity to logical points ---
def find_potential_attachment(blueprint: RocketBlueprint, part_to_place_data: dict, target_screen_pos: pygame.math.Vector2, offset: pygame.math.Vector2, zoom: float) -> tuple[PlacedPart | None, str | None, str | None, pygame.math.Vector2 | None]:
    """ Finds the best existing part and attachment pair near the target screen position.
        Returns: (target_part, target_area_name, new_part_area_name, connection_world_pos)
    """
    target_world_pos = screen_to_world(target_screen_pos, offset, zoom)
    closest_dist_sq = ATTACHMENT_SNAP_DISTANCE_SQ # Use the constant defined earlier
    best_match = (None, None, None, None)

    if not blueprint.parts: return best_match

    part_to_place_type = part_to_place_data.get("type")
    part_to_place_rules = part_to_place_data.get("attachment_rules", {})

    for existing_part in blueprint.parts:
        existing_part_rules = existing_part.part_data.get("attachment_rules", {})
        existing_part_points = existing_part.part_data.get("logical_points", {})

        # Iterate through the *logical points* of the existing part
        for point_name, local_pos in existing_part_points.items():
            # Calculate world position of this logical point (assume angle 0 for existing)
            # TODO: Support rotated existing parts if needed later
            point_world_pos = existing_part.relative_pos + local_pos.rotate(-existing_part.relative_angle)

            # Check distance from mouse (in world space) to this point
            dist_sq = (point_world_pos - target_world_pos).length_squared()

            if dist_sq < closest_dist_sq:
                 # Now, check if the attachment RULES allow this connection
                 # Find the rule name associated with this logical point (e.g., "bottom" point -> "bottom_center" rule)
                 rule_name = None
                 if "bottom" in point_name and "bottom_center" in existing_part_rules: rule_name = "bottom_center"
                 elif "top" in point_name and "top_center" in existing_part_rules: rule_name = "top_center"
                 elif "left" in point_name and "left_center" in existing_part_rules: rule_name = "left_center"
                 elif "right" in point_name and "right_center" in existing_part_rules: rule_name = "right_center"
                 # Add more mappings if needed

                 if rule_name:
                     # Check if existing part's rule allows the new part type
                     allowed_types = existing_part_rules[rule_name].get("allowed_types", [])
                     if part_to_place_type in allowed_types:
                          # Find the corresponding point/rule on the new part
                          compatible_new_area = None
                          if "bottom" in rule_name and "top_center" in part_to_place_rules: compatible_new_area = "top_center"
                          elif "top" in rule_name and "bottom_center" in part_to_place_rules: compatible_new_area = "bottom_center"
                          # Add side/other rules

                          if compatible_new_area:
                              # Found a valid match!
                              closest_dist_sq = dist_sq
                              best_match = (existing_part, rule_name, compatible_new_area, point_world_pos)
                              # Continue checking other points/parts for potentially closer matches


    return best_match


def setup_builder_ui():
    # ... (Setup remains the same as before) ...
    global builder_buttons, part_list_buttons
    builder_buttons = {}
    part_list_buttons = {}
    button_width = 120
    button_height = 35
    start_x = BUTTON_AREA_RECT.left + 20 # Position relative to build area left
    y_pos = BUTTON_AREA_RECT.top + (BUTTON_AREA_RECT.height - button_height) // 2
    builder_buttons["launch"] = Button("Launch", (start_x, y_pos, button_width, button_height))
    builder_buttons["clear"] = Button("Clear", (start_x + button_width + 10, y_pos, button_width, button_height))
    builder_buttons["back"] = Button("Back to Menu", (BUTTON_AREA_RECT.right - (button_width + 40) - 20, y_pos, button_width + 40, button_height))

    part_y = 20
    part_button_height = 40
    part_font_size = 18
    for part_id, part_data in PARTS_CATALOG.items():
        rect = pygame.Rect(10, part_y, PARTS_LIST_WIDTH - 20, part_button_height)
        part_list_buttons[part_id] = Button(part_data["name"], rect, font_size=part_font_size, color=(80,80,90), hover_color=(110,110,120))
        part_y += part_button_height + 5


# --- Main Builder Loop ---
def run_builder(screen, clock):
    global selected_part_id, current_blueprint, build_offset
    # Load or create blueprint
    if os.path.exists(TEMP_BLUEPRINT_FILE):
        loaded_bp = RocketBlueprint.load_from_json(TEMP_BLUEPRINT_FILE)
        current_blueprint = loaded_bp if loaded_bp else RocketBlueprint("My Grid Rocket")
    else:
        current_blueprint = RocketBlueprint("My Grid Rocket")
    # Save initial state (or loaded state)
    current_blueprint.save_to_json(TEMP_BLUEPRINT_FILE)

    selected_part_id = None
    setup_builder_ui()

    building = True
    placement_blocked = False
    ghost_draw_pos_screen = None # Screen position to draw the ghost center
    potential_relative_pos = None # Calculated RELATIVE position for placing (world coords relative to blueprint 0,0)
    is_attaching = False # Flag to indicate if preview/placement is based on attachment
    attaching_to_part = None # Which part instance are we trying to attach to?

    dragging_view = False
    drag_start_mouse = None
    drag_start_offset = None

    while building:
        dt = clock.tick(60) / 1000.0
        mouse_pos_tuple = pygame.mouse.get_pos()
        mouse_pos = pygame.math.Vector2(mouse_pos_tuple)

        # Reset state for this frame
        placement_blocked = False
        ghost_draw_pos_screen = None
        potential_relative_pos = None
        is_attaching = False
        attaching_to_part = None

        # --- Calculate potential placement and check validity ---
        if selected_part_id and BUILD_AREA_RECT.collidepoint(mouse_pos):
            selected_part_data = get_part_data(selected_part_id)

            # --- Attachment Logic ---
            if current_blueprint.parts:
                # Find if mouse is close enough to a valid attachment point
                attach_result = find_potential_attachment(current_blueprint, selected_part_data, mouse_pos, build_offset, build_zoom)
                target_part, target_area, new_part_area, connection_world_pos = attach_result

                if target_part:
                    # --- ATTACHED PLACEMENT PREVIEW ---
                    is_attaching = True
                    attaching_to_part = target_part # Store which part we are targeting
                    # Calculate where the new part's center should be to make the connection
                    new_part_logical_points = selected_part_data.get("logical_points", {})
                    new_part_connection_point_name = None
                    if new_part_area == "top_center": new_part_connection_point_name = "top"
                    elif new_part_area == "bottom_center": new_part_connection_point_name = "bottom"
                    # Add side mappings etc.

                    if new_part_connection_point_name and new_part_connection_point_name in new_part_logical_points:
                        new_part_connection_local_offset = new_part_logical_points[new_part_connection_point_name]
                        # Calculate precise world center for the new part (angle 0 assumed for new part preview)
                        # potential_center_world = connection_point_world - offset_on_new_part_rotated
                        potential_center_world = connection_world_pos - new_part_connection_local_offset.rotate(0)

                        # *** USE PRECISE POSITION, DO NOT SNAP TO GRID ***
                        potential_relative_pos = potential_center_world # This is the target relative position

                        # Set ghost position based on this precise calculation
                        ghost_draw_pos_screen = world_to_screen(potential_relative_pos, build_offset, build_zoom)

                        # Check collision EXCLUDING the part we attach to
                        placement_blocked = check_placement_collision(current_blueprint, selected_part_id, potential_relative_pos, attaching_to_part)
                        if placement_blocked: print(f"Collision detected for attachment.")

                    else: # Cannot determine placement from attachment points
                        placement_blocked = True
                        # Fallback: show ghost snapped to grid near mouse
                        mouse_world_pos = screen_to_world(mouse_pos, build_offset, build_zoom)
                        potential_relative_pos = snap_to_grid(mouse_world_pos)
                        ghost_draw_pos_screen = world_to_screen(potential_relative_pos, build_offset, build_zoom)

                else:
                    # --- FREE PLACEMENT PREVIEW (Not attaching) ---
                    is_attaching = False
                    attaching_to_part = None
                    mouse_world_pos = screen_to_world(mouse_pos, build_offset, build_zoom)
                    # *** SNAP TO GRID for free placement ***
                    potential_relative_pos = snap_to_grid(mouse_world_pos)
                    ghost_draw_pos_screen = world_to_screen(potential_relative_pos, build_offset, build_zoom)
                    # Check collision against ALL parts
                    placement_blocked = check_placement_collision(current_blueprint, selected_part_id, potential_relative_pos)
                    # if placement_blocked: print(f"Collision detected for free placement.")


            else: # --- ROOT PART PLACEMENT PREVIEW ---
                is_attaching = False
                attaching_to_part = None
                # Place root at blueprint origin (0, 0), snapped implicitly
                potential_relative_pos = pygame.math.Vector2(0, 0)
                ghost_draw_pos_screen = world_to_screen(potential_relative_pos, build_offset, build_zoom)
                placement_blocked = False # No collision possible


        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if selected_part_id:
                        selected_part_id = None  # Deselect part if one is selected
                        print("Deselected part via Escape.")
                    else:
                        building = False  # Exit builder if no part selected

            if event.type == pygame.MOUSEBUTTONDOWN:
                click_pos = pygame.math.Vector2(event.pos)

                if event.button == 1: # Left Click
                    clicked_action = None; clicked_part_list_item = False
                    # Check UI Buttons
                    for name, button in builder_buttons.items():
                        if button.is_clicked(event): clicked_action = name; break
                    if not clicked_action:
                        for part_id, button in part_list_buttons.items():
                            if button.is_clicked(event):
                                selected_part_id = part_id; clicked_part_list_item = True; break

                    # Handle Actions
                    if clicked_action == "launch":
                         if current_blueprint.parts:
                            current_blueprint.save_to_json(TEMP_BLUEPRINT_FILE)
                            print(f"Saved blueprint for launch: {TEMP_BLUEPRINT_FILE}")
                            return ("LAUNCH", TEMP_BLUEPRINT_FILE)
                         else: print("Cannot launch empty rocket!")
                    elif clicked_action == "clear":
                         current_blueprint = RocketBlueprint("Cleared Rocket")
                         current_blueprint.save_to_json(TEMP_BLUEPRINT_FILE)
                         selected_part_id = None; print("Blueprint cleared.")
                    elif clicked_action == "back": building = False

                    # --- Try Placing Part in Build Area ---
                    elif not clicked_part_list_item and BUILD_AREA_RECT.collidepoint(click_pos) and selected_part_id:
                        # Use the potential_relative_pos calculated during preview
                        if potential_relative_pos is not None and not placement_blocked:
                            # Add part with angle 0 for grid build
                            current_blueprint.add_part(selected_part_id, potential_relative_pos, angle=0)
                            print(f"Placed {selected_part_id} at rel {potential_relative_pos} {'(attached)' if is_attaching else '(free)'}")
                            current_blueprint.save_to_json(TEMP_BLUEPRINT_FILE) # Autosave
                            selected_part_id = None # Deselect after placing
                        else:
                            block_reason = "collision" if placement_blocked else "invalid position"
                            print(f"Placement blocked for {selected_part_id} ({block_reason}).")

                    # Initiate view drag if clicking empty space without selection
                    elif not clicked_part_list_item and BUILD_AREA_RECT.collidepoint(click_pos) and not selected_part_id:
                         dragging_view = True
                         drag_start_mouse = mouse_pos
                         drag_start_offset = build_offset.copy()


                elif event.button == 3: # Right Click
                    if selected_part_id:
                        selected_part_id = None; print("Deselected part.")
                    elif BUILD_AREA_RECT.collidepoint(event.pos):
                         # Remove part under cursor
                         click_world_pos = screen_to_world(pygame.math.Vector2(event.pos), build_offset, build_zoom)
                         part_to_remove = current_blueprint.get_part_at_world_pos(click_world_pos)
                         if part_to_remove:
                             # Basic removal - doesn't handle removing parts that support others yet
                             current_blueprint.remove_part(part_to_remove)
                             print(f"Removed part: {part_to_remove.part_id}")
                             current_blueprint.save_to_json(TEMP_BLUEPRINT_FILE) # Autosave
                         # else: print("Right-click on empty space.")


                elif event.button == 2: # Middle Mouse Button Down for drag
                     if BUILD_AREA_RECT.collidepoint(event.pos):
                         dragging_view = True
                         drag_start_mouse = mouse_pos
                         drag_start_offset = build_offset.copy()


            if event.type == pygame.MOUSEBUTTONUP:
                 if event.button == 1 or event.button == 2: # Left or Middle release stops drag
                     dragging_view = False
                     drag_start_mouse = None
                     drag_start_offset = None

            # Handle Mouse Wheel for Zoom (Optional)
            # ... (zoom logic) ...

        # --- Update ---
        # View Dragging
        if dragging_view and drag_start_mouse:
            delta = mouse_pos - drag_start_mouse
            build_offset = drag_start_offset + delta

        # Update button hover states
        for button in builder_buttons.values(): button.check_hover(mouse_pos_tuple)
        for pid, button in part_list_buttons.items():
             button.check_hover(mouse_pos_tuple)
             # Highlight button if hovered OR if it's the selected part
             is_selected = (selected_part_id == pid)
             button.is_hovered = button.rect.collidepoint(mouse_pos_tuple) or is_selected


        # --- Drawing ---
        screen.fill(BUILD_BG_COLOR)

        # Build Area Background and Grid
        pygame.draw.rect(screen, (25, 25, 35), BUILD_AREA_RECT)
        draw_grid(screen, build_offset, build_zoom, BUILD_AREA_RECT, GRID_SIZE)

        # Draw the rocket blueprint
        draw_blueprint(screen, current_blueprint, build_offset, build_zoom)

        # Ghost Part Preview
        if selected_part_id and ghost_draw_pos_screen:
            ghost_part_data = get_part_data(selected_part_id)
            w, h = ghost_part_data['width'], ghost_part_data['height']
            ghost_surf = pygame.Surface((w, h), pygame.SRCALPHA)
            # Draw shape centered on the surface
            draw_part_shape(ghost_surf, ghost_part_data, pygame.math.Vector2(w/2, h/2))
            # Apply color tint based on validity
            ghost_surf.set_alpha(150)
            color_overlay = pygame.Surface((w, h), pygame.SRCALPHA)
            ghost_color = PLACEMENT_OK_COLOR if not placement_blocked else PLACEMENT_BLOCKED_COLOR
            color_overlay.fill((*ghost_color, 100))
            ghost_surf.blit(color_overlay, (0,0), special_flags=pygame.BLEND_RGBA_MULT)

            # Draw ghost centered at the calculated screen position
            screen.blit(ghost_surf, ghost_draw_pos_screen - pygame.math.Vector2(w/2, h/2))

            # Highlight attachment point if attaching
            if is_attaching and attaching_to_part:
                 # Find the world position of the connection point again for drawing
                 attach_res_draw = find_potential_attachment(current_blueprint, selected_part_data, mouse_pos, build_offset, build_zoom)
                 if attach_res_draw[0]: # Check if still valid
                      conn_world_pos_draw = attach_res_draw[3]
                      conn_screen_pos_draw = world_to_screen(conn_world_pos_draw, build_offset, build_zoom)
                      highlight_color = WHITE if not placement_blocked else RED
                      pygame.draw.circle(screen, highlight_color, conn_screen_pos_draw, 5, 1)


        # Parts List Area
        pygame.draw.rect(screen, (30, 30, 40), PARTS_LIST_RECT)
        pygame.draw.line(screen, WHITE, (PARTS_LIST_WIDTH - 1, 0), (PARTS_LIST_WIDTH - 1, SCREEN_HEIGHT), 1)
        list_font = pygame.font.SysFont(None, 24); list_title = list_font.render("Available Parts", True, WHITE); screen.blit(list_title, (10, 5))
        for button in part_list_buttons.values(): button.draw(screen)

        # Button Area
        pygame.draw.rect(screen, (20, 20, 30), BUTTON_AREA_RECT)
        pygame.draw.line(screen, WHITE, (BUTTON_AREA_RECT.left, BUTTON_AREA_RECT.top), (BUTTON_AREA_RECT.right, BUTTON_AREA_RECT.top), 1)
        for button in builder_buttons.values(): button.draw(screen)

        pygame.display.flip()

    # Autosave on exit
    current_blueprint.save_to_json(TEMP_BLUEPRINT_FILE)
    print("Exiting builder, blueprint saved.")
    return STATE_MAIN_MENU