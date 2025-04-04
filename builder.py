# builder.py
import pygame
import sys
import math
import os
from parts import PARTS_CATALOG, get_part_data, draw_part_shape
from rocket_data import RocketBlueprint, PlacedPart
from ui_elements import Button, STATE_MAIN_MENU
from ui_elements import SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, BLACK, GRAY, LIGHT_GRAY, BLUE, RED, GREEN

# --- Constants ---
BUILD_BG_COLOR = (40, 40, 60)
PLACEMENT_OK_COLOR = GREEN
PLACEMENT_BLOCKED_COLOR = RED # Color for ghost when blocked

# --- Builder Layout --- (same as before)
PARTS_LIST_WIDTH = 180
PARTS_LIST_RECT = pygame.Rect(0, 0, PARTS_LIST_WIDTH, SCREEN_HEIGHT)
BUILD_AREA_RECT = pygame.Rect(PARTS_LIST_WIDTH, 0, SCREEN_WIDTH - PARTS_LIST_WIDTH, SCREEN_HEIGHT)
BUTTON_AREA_HEIGHT = 50
BUTTON_AREA_RECT = pygame.Rect(PARTS_LIST_WIDTH, SCREEN_HEIGHT - BUTTON_AREA_HEIGHT, BUILD_AREA_RECT.width, BUTTON_AREA_HEIGHT)

# --- Builder State Variables --- (same as before)
selected_part_id = None
current_blueprint = RocketBlueprint("Unnamed Build")
build_area_center = BUILD_AREA_RECT.center
build_offset = pygame.math.Vector2(build_area_center[0], build_area_center[1] - 100)
build_zoom = 1.0 # Not implemented yet
builder_buttons = {}
part_list_buttons = {}
ATTACH_POINT_RADIUS = 5
ATTACH_SNAP_DISTANCE = 15
TEMP_BLUEPRINT_FILE = os.path.join("assets", "current_build.json")

# --- Helper Functions ---

def draw_blueprint(surface, blueprint, offset, zoom):
    # ... (same drawing code as before, including drawing attachment points) ...
    if not blueprint.parts: return
    root_part = blueprint.parts[0]
    root_screen_pos = offset + root_part.relative_pos
    for part_idx, part in enumerate(blueprint.parts):
        part_data = part.part_data
        part_screen_pos = offset + part.relative_pos
        draw_part_shape(surface, part_data, part_screen_pos, part.relative_angle)
        if "attachment_points" in part_data:
            for name, ap_local_pos in part_data["attachment_points"].items():
                 rotated_ap_offset = ap_local_pos.rotate(-part.relative_angle)
                 ap_screen_pos = part_screen_pos + rotated_ap_offset
                 ap_color = GREEN # TODO: Check occupancy later
                 pygame.draw.circle(surface, ap_color, ap_screen_pos, ATTACH_POINT_RADIUS)


def get_closest_attachment_point(blueprint, target_screen_pos, offset):
    # ... (same code as before) ...
    closest_ap = None; min_dist_sq = (ATTACH_SNAP_DISTANCE * build_zoom) ** 2
    closest_part = None; closest_ap_name = None; closest_ap_screen_pos = None
    if not blueprint.parts: return None
    for part_idx, part in enumerate(blueprint.parts):
        part_data = part.part_data
        part_screen_pos = offset + part.relative_pos
        if "attachment_points" in part_data:
            for name, ap_local_pos in part_data["attachment_points"].items():
                 # TODO: Check occupancy
                 rotated_ap_offset = ap_local_pos.rotate(-part.relative_angle)
                 ap_screen_pos = part_screen_pos + rotated_ap_offset
                 dist_sq = (target_screen_pos - ap_screen_pos).length_squared()
                 if dist_sq < min_dist_sq:
                     min_dist_sq = dist_sq
                     closest_ap = ap_local_pos; closest_ap_name = name
                     closest_part = part; closest_ap_screen_pos = ap_screen_pos
    if closest_part:
        return {"part": closest_part, "ap_name": closest_ap_name, "ap_local_pos": closest_ap, "ap_screen_pos": closest_ap_screen_pos}
    else:
        return None

# --- NEW HELPER: Calculate Approximate World Rect ---
def get_part_world_rect(part: PlacedPart, offset: pygame.math.Vector2):
    """Calculates an approximate world AABB for collision checking. Ignores rotation for simplicity."""
    part_data = part.part_data
    w = part_data['width']
    h = part_data['height']
    # Calculate world center position
    part_world_center = offset + part.relative_pos
    # Create AABB centered on world position
    # For rotated parts, this is just an approximation!
    world_rect = pygame.Rect(0, 0, w, h)
    world_rect.center = part_world_center
    return world_rect

# --- NEW HELPER: Check for Collision ---
def check_placement_collision(blueprint: RocketBlueprint, new_part_id: str, new_part_rel_pos: pygame.math.Vector2, offset: pygame.math.Vector2, attaching_to_part: PlacedPart = None):
    """Checks if placing a new part collides with existing parts (excluding the direct attachment)."""
    if not blueprint.parts: # No collision if it's the first part
        return False

    # Create a temporary PlacedPart object for the new part
    try:
        temp_new_part = PlacedPart(new_part_id, new_part_rel_pos) # Assumes angle 0 for now
    except ValueError:
        return True # Invalid part ID, treat as collision

    new_part_rect = get_part_world_rect(temp_new_part, offset)

    for existing_part in blueprint.parts:
        # *** IMPORTANT: Skip checking collision with the exact part we are attaching to ***
        # This allows parts to touch at the attachment point without failing the check.
        # If attaching_to_part is None (placing root), this check is skipped anyway.
        if existing_part is attaching_to_part:
            continue

        existing_part_rect = get_part_world_rect(existing_part, offset)
        if new_part_rect.colliderect(existing_part_rect):
            print(f"Collision detected between potential {new_part_id} and existing {existing_part.part_id}") # DEBUG
            return True # Collision found

    return False # No collision detected


def setup_builder_ui():
     # ... (same code as before) ...
    global builder_buttons, part_list_buttons
    builder_buttons = {}; part_list_buttons = {}
    button_width = 120; button_height = 35
    start_x = BUILD_AREA_RECT.left + 20
    y_pos = BUTTON_AREA_RECT.top + (BUTTON_AREA_RECT.height - button_height) // 2
    builder_buttons["launch"] = Button("Launch", (start_x, y_pos, button_width, button_height))
    builder_buttons["clear"] = Button("Clear", (start_x + button_width + 10, y_pos, button_width, button_height))
    builder_buttons["back"] = Button("Back to Menu", (BUILD_AREA_RECT.right - button_width - 60, y_pos, button_width + 40, button_height))
    part_y = 20; part_button_height = 40; part_font_size = 18
    for part_id, part_data in PARTS_CATALOG.items():
        rect = pygame.Rect(10, part_y, PARTS_LIST_WIDTH - 20, part_button_height)
        part_list_buttons[part_id] = Button(part_data["name"], rect, font_size=part_font_size, color=(80,80,90), hover_color=(110,110,120))
        part_y += part_button_height + 5


# --- Main Builder Loop ---
def run_builder(screen, clock):
    global selected_part_id, current_blueprint, build_offset
    current_blueprint = RocketBlueprint("My New Rocket")
    selected_part_id = None
    setup_builder_ui()

    building = True
    placement_blocked = False # Flag to color the ghost part red if placement invalid

    while building:
        dt = clock.tick(60) / 1000.0
        mouse_pos_tuple = pygame.mouse.get_pos()
        mouse_pos = pygame.math.Vector2(mouse_pos_tuple)
        placement_blocked = False # Reset placement block flag each frame

        # --- Calculate closest AP and potential placement for visual feedback ---
        closest_ap_info = None
        potential_placement_pos = None
        attaching_to_part_preview = None # Store which part we're previewing attachment to

        if selected_part_id and BUILD_AREA_RECT.collidepoint(mouse_pos):
            closest_ap_info = get_closest_attachment_point(current_blueprint, mouse_pos, build_offset)

            # Check if placement is possible (either root or valid snap)
            if not current_blueprint.parts: # Placing root
                 potential_placement_pos = pygame.math.Vector2(0, 0) # Root always at relative (0,0)
                 placement_blocked = check_placement_collision(current_blueprint, selected_part_id, potential_placement_pos, build_offset) # Should always be false for root
            elif closest_ap_info: # Snapping to existing part
                attaching_to_part_preview = closest_ap_info["part"]
                target_ap_local_pos = closest_ap_info["ap_local_pos"]
                new_part_data = get_part_data(selected_part_id)
                new_part_aps = new_part_data.get("attachment_points", {})

                # --- Find matching attachment point (including sides) ---
                new_part_ap_name = None
                target_ap_name = closest_ap_info["ap_name"]
                if "bottom" in target_ap_name and "top" in new_part_aps: new_part_ap_name = "top"
                elif "top" in target_ap_name and "bottom" in new_part_aps: new_part_ap_name = "bottom"
                elif "right" in target_ap_name and "left" in new_part_aps: new_part_ap_name = "left"
                elif "left" in target_ap_name and "right" in new_part_aps: new_part_ap_name = "right"
                # Add more complex rules? (e.g., side-to-top?)

                if new_part_ap_name:
                    new_part_ap_local_pos = new_part_aps[new_part_ap_name]
                    potential_placement_pos = attaching_to_part_preview.relative_pos + target_ap_local_pos - new_part_ap_local_pos
                    # Check for collision *before* confirming placement visually
                    placement_blocked = check_placement_collision(current_blueprint, selected_part_id, potential_placement_pos, build_offset, attaching_to_part_preview)
                else:
                    # No matching attachment point found on the selected part for this snap
                    potential_placement_pos = None
                    placement_blocked = True # Cannot place if no matching AP
            else:
                 # Not snapping and not root - placement blocked
                 placement_blocked = True


        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: building = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                click_pos = pygame.math.Vector2(event.pos)

                if event.button == 1: # Left Click
                    # --- Check UI Buttons First ---
                    clicked_action = None; clicked_part = False
                    for name, button in builder_buttons.items():
                        if button.is_clicked(event): clicked_action = name; break
                    if not clicked_action:
                        for part_id, button in part_list_buttons.items():
                            if button.is_clicked(event):
                                selected_part_id = part_id; clicked_part = True; break

                    # --- Handle Actions ---
                    if clicked_action == "launch":
                         if current_blueprint.parts and not placement_blocked: # Check if last placement was valid? Maybe not needed here.
                            try:
                                current_blueprint.save_to_json(TEMP_BLUEPRINT_FILE)
                                return ("LAUNCH", TEMP_BLUEPRINT_FILE)
                            except Exception as e: print(f"Error saving blueprint: {e}")
                         else: print("Cannot launch empty or invalid rocket!")
                    elif clicked_action == "clear": current_blueprint = RocketBlueprint("Cleared"); selected_part_id = None
                    elif clicked_action == "back": building = False

                    # --- Try Placing Part (if no UI button was clicked) ---
                    elif not clicked_part and BUILD_AREA_RECT.collidepoint(click_pos):
                        if selected_part_id:
                            # Use the potential placement calculated for preview, but re-check block
                            # Recalculate closest AP based on exact click pos might be slightly more robust,
                            # but using the frame's calculation is usually fine.
                            if potential_placement_pos is not None and not placement_blocked:
                                current_blueprint.add_part(selected_part_id, potential_placement_pos)
                                print(f"Placed part {selected_part_id} at {potential_placement_pos}")
                                selected_part_id = None # Deselect after placing
                            elif not current_blueprint.parts and not placement_blocked: # Specific check for root placement
                                current_blueprint.add_part(selected_part_id, pygame.math.Vector2(0, 0))
                                print(f"Placed root part {selected_part_id}")
                                selected_part_id = None # Deselect after placing
                            else:
                                print("Placement blocked (collision or no snap point).")
                        else:
                             print("Click in build area, no part selected.")

                elif event.button == 3: # Right Click
                    selected_part_id = None; print("Deselected part.")

        # --- Update ---
        for button in builder_buttons.values(): button.check_hover(mouse_pos_tuple)
        for pid, button in part_list_buttons.items():
             button.check_hover(mouse_pos_tuple)
             if selected_part_id == pid: button.is_hovered = True

        # --- Drawing ---
        screen.fill(BUILD_BG_COLOR)
        # Parts List Area... (same as before)
        pygame.draw.rect(screen, (30, 30, 40), PARTS_LIST_RECT); pygame.draw.line(screen, WHITE, (PARTS_LIST_WIDTH - 1, 0), (PARTS_LIST_WIDTH - 1, SCREEN_HEIGHT), 1)
        list_font = pygame.font.SysFont(None, 24); list_title = list_font.render("Available Parts", True, WHITE); screen.blit(list_title, (10, 5))
        for button in part_list_buttons.values(): button.draw(screen)

        # Build Area...
        draw_blueprint(screen, current_blueprint, build_offset, build_zoom)

        # Ghost Part Preview...
        if selected_part_id and BUILD_AREA_RECT.collidepoint(mouse_pos_tuple):
            ghost_part_data = get_part_data(selected_part_id)
            draw_pos = mouse_pos # Default to mouse pos
            ghost_color = PLACEMENT_OK_COLOR if not placement_blocked else PLACEMENT_BLOCKED_COLOR

            # Snap preview if possible
            if potential_placement_pos is not None:
                # Calculate screen position from relative potential position
                 draw_pos = build_offset + potential_placement_pos
            # If snapping but blocked, still show at snapped pos but in red
            elif closest_ap_info: # Means we found an AP but matching/collision failed
                 # Calculate snapped pos for visual feedback even if blocked
                 target_ap_screen_pos = closest_ap_info["ap_screen_pos"]
                 new_part_aps = ghost_part_data.get("attachment_points", {})
                 new_part_ap_name = None # Find matching name again
                 target_ap_name = closest_ap_info["ap_name"]
                 if "bottom" in target_ap_name and "top" in new_part_aps: new_part_ap_name = "top"
                 elif "top" in target_ap_name and "bottom" in new_part_aps: new_part_ap_name = "bottom"
                 elif "right" in target_ap_name and "left" in new_part_aps: new_part_ap_name = "left"
                 elif "left" in target_ap_name and "right" in new_part_aps: new_part_ap_name = "right"
                 if new_part_ap_name:
                     ghost_ap_local_pos = new_part_aps[new_part_ap_name]
                     draw_pos = target_ap_screen_pos - ghost_ap_local_pos


            # Draw ghost using alpha and color tint
            w, h = ghost_part_data['width'], ghost_part_data['height']
            center_offset = pygame.math.Vector2(w/2, h/2)
            ghost_surf = pygame.Surface((w, h), pygame.SRCALPHA)
            # Draw the basic part shape onto the surface
            draw_part_shape(ghost_surf, ghost_part_data, center_offset)
            # Apply color tint and alpha
            ghost_surf.set_alpha(150)
            # Create a colored overlay with transparency
            color_overlay = pygame.Surface((w, h), pygame.SRCALPHA)
            color_overlay.fill((*ghost_color, 100)) # Use placement color with alpha
            ghost_surf.blit(color_overlay, (0,0), special_flags=pygame.BLEND_RGBA_MULT) # Blend color

            screen.blit(ghost_surf, draw_pos - center_offset)

            # Highlight snap point if applicable
            if closest_ap_info:
                 pygame.draw.circle(screen, WHITE if not placement_blocked else RED, closest_ap_info["ap_screen_pos"], ATTACH_POINT_RADIUS + 2, 1)


        # Button Area... (same as before)
        pygame.draw.rect(screen, (20, 20, 30), BUTTON_AREA_RECT); pygame.draw.line(screen, WHITE, (BUTTON_AREA_RECT.left, BUTTON_AREA_RECT.top), (BUTTON_AREA_RECT.right, BUTTON_AREA_RECT.top), 1)
        for button in builder_buttons.values(): button.draw(screen)

        pygame.display.flip()

    return STATE_MAIN_MENU