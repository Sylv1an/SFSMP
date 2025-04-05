# parts.py
import pygame

# Add new types
PART_TYPES = ["CommandPod", "FuelTank", "Engine", "Parachute", "Separator"]

PARTS_CATALOG = {
    "pod_mk1": {
        "name": "Mk1 Command Pod", "type": "CommandPod", "mass": 50,
        "width": 20, "height": 25, "color": (200, 200, 200),
        "max_hp": 200,
        # Define attachment areas/rules instead of points for grid builder
        "attachment_rules": {
            "top_center": {"allowed_types": ["Parachute"]}, # Can attach parachute here
            "bottom_center": {"allowed_types": ["FuelTank", "Engine", "Separator"]} # Can attach stack parts below
        },
        # Keep internal points for drawing or logical centers if needed, but not for general snapping
        "logical_points": {"bottom": pygame.math.Vector2(0, 12.5), "top": pygame.math.Vector2(0, -12.5)}
    },
    "tank_small": {
        "name": "Small Fuel Tank", "type": "FuelTank", "mass": 20,
        "fuel_capacity": 100, "width": 20, "height": 40, "color": (255, 255, 255),
        "max_hp": 100,
        "attachment_rules": {
             "top_center": {"allowed_types": ["CommandPod", "FuelTank", "Separator"]},
             "bottom_center": {"allowed_types": ["FuelTank", "Engine", "Separator"]},
             # Example side attachment possibility (if needed later)
             # "left_center": {"allowed_types": ["Separator", "Winglet"]},
             # "right_center": {"allowed_types": ["Separator", "Winglet"]},
        },
        "logical_points": {"top": pygame.math.Vector2(0, -20), "bottom": pygame.math.Vector2(0, 20)}
    },
    "engine_basic": {
        "name": "Basic Liquid Engine", "type": "Engine", "mass": 30,
        "thrust": 9.81 * 8 * 150, "fuel_consumption": 0.5,
        "width": 18, "height": 20, "color": (100, 100, 100),
        "max_hp": 80,
        "attachment_rules": {
             "top_center": {"allowed_types": ["CommandPod", "FuelTank", "Separator"]},
        },
         "logical_points": {"top": pygame.math.Vector2(0, -10)}
    },
    # --- NEW PARTS ---
    "parachute_mk1": {
        "name": "Parachute Mk1", "type": "Parachute", "mass": 10,
        "width": 15, "height": 10, "color": (200, 100, 0), # Orange when packed
        "max_hp": 50,
        "deploy_drag": 8.0, # Drag coefficient increase when deployed (adjust value)
        "deploy_area_factor": 10, # Factor to increase effective area when deployed
        "activatable": True, # Can be clicked in flight
        "attachment_rules": {
             # Only has a bottom connection point logically
             "bottom_center": {"allowed_types": ["CommandPod"]}, # Can ONLY attach below to a pod
        },
        "logical_points": {"bottom": pygame.math.Vector2(0, 5)}
    },
    "separator_tr_s1": {
        "name": "Transverse Separator S1", "type": "Separator", "mass": 15,
        "width": 20, "height": 8, "color": (200, 200, 0), # Yellow band
        "max_hp": 60,
        "separation_force": 8000, # Force applied on separation
        "separation_axis": "transverse", # Separates top from bottom
        "activatable": True,
        "attachment_rules": {
             "top_center": {"allowed_types": ["CommandPod", "FuelTank", "Separator"]},
             "bottom_center": {"allowed_types": ["FuelTank", "Engine", "Separator"]},
        },
         "logical_points": {"top": pygame.math.Vector2(0, -4), "bottom": pygame.math.Vector2(0, 4)}
    },
    # Example: Longitudinal Separator (more complex logic needed)
    # "separator_ln_s1": {
    #     "name": "Longitudinal Separator S1", "type": "Separator", "mass": 12,
    #     "width": 8, "height": 40, "color": (0, 100, 200), # Blue band
    #     "max_hp": 50,
    #     "separation_force": 6000,
    #     "separation_axis": "longitudinal", # Separates side-attached items
    #     "activatable": True,
    #     "attachment_rules": {
    #         # Needs logic for how it attaches TO a main stack part
    #         "attach_to_side": {"allowed_types": ["FuelTank"]}, # e.g., can attach TO the side of a tank
    #         # Needs logic for what can attach TO IT (boosters?)
    #         "side_mount": {"allowed_types": ["FuelTank", "Engine"]}
    #     },
    #     "logical_points": { ... }
    # }
}

def get_part_data(part_id): return PARTS_CATALOG.get(part_id)

# Keep draw_part_shape, but add visual states (e.g., deployed parachute)
def draw_part_shape(surface, part_data, center_pos, angle_deg=0, broken=False, deployed=False): # Add deployed flag
    w = part_data["width"]; h = part_data["height"]
    base_color = part_data["color"]
    color = (50, 0, 0) if broken else base_color # Dark red if broken

    # Basic rectangle shape
    points = [ pygame.math.Vector2(x, y) for x, y in [(-w/2,-h/2), (w/2,-h/2), (w/2,h/2), (-w/2,h/2)] ]
    rotated_points = [p.rotate(-angle_deg) for p in points]
    screen_points = [p + center_pos for p in rotated_points]
    pygame.draw.polygon(surface, color, screen_points)
    pygame.draw.polygon(surface, (50,50,50) if not broken else (100,50,50), screen_points, 1)

    # Draw deployed parachute visual
    if part_data["type"] == "Parachute" and deployed and not broken:
        # Simple orange arc above the part
        deploy_w = w * 3
        deploy_h = h * 2
        deploy_rect = pygame.Rect(0, 0, deploy_w, deploy_h * 2) # Rect for arc drawing
        # Calculate top center position after rotation
        top_offset = pygame.math.Vector2(0, -h/2 - 2) # Small gap above part
        rotated_top_offset = top_offset.rotate(-angle_deg)
        deploy_center_screen = center_pos + rotated_top_offset - pygame.math.Vector2(0, deploy_h) # Arc center is above the top

        deploy_rect.center = deploy_center_screen
        pygame.draw.arc(surface, (255, 165, 0), deploy_rect, math.radians(20), math.radians(160), 4) # Orange arc

    if broken:
        pygame.draw.line(surface, (255,0,0), screen_points[0], screen_points[2], 2)
        pygame.draw.line(surface, (255,0,0), screen_points[1], screen_points[3], 2)

# --- ADDED FOR GRID BUILDER ---
import math
def get_part_attachment_areas(part: 'PlacedPart', part_world_center: pygame.math.Vector2, part_world_angle: float) -> dict:
    """ Calculates the world positions of key attachment areas based on rules. """
    areas = {}
    part_data = part.part_data
    rules = part_data.get("attachment_rules", {})
    points = part_data.get("logical_points", {})

    for area_name, rule_data in rules.items():
        if area_name == "top_center" and "top" in points:
            local_pos = points["top"]
        elif area_name == "bottom_center" and "bottom" in points:
            local_pos = points["bottom"]
        # Add more complex rules for sides etc. if needed
        # elif area_name == "left_center": local_pos = pygame.math.Vector2(-part_data['width']/2, 0)
        else:
            continue # Skip if logical point not defined for this rule

        rotated_offset = local_pos.rotate(-part_world_angle)
        world_pos = part_world_center + rotated_offset
        areas[area_name] = {"world_pos": world_pos, "allowed_types": rule_data.get("allowed_types", [])}

    return areas

def can_attach(part_to_place_data: dict, existing_part: 'PlacedPart', existing_part_world_center: pygame.math.Vector2, existing_part_world_angle: float, target_world_pos: pygame.math.Vector2, snap_distance_sq: float) -> tuple[str | None, str | None, pygame.math.Vector2 | None]:
    """ Checks if part_to_place can attach to existing_part near target_world_pos.
        Returns: (attach_area_on_existing, attach_area_on_new, connection_world_pos) or (None, None, None)
    """
    part_to_place_type = part_to_place_data.get("type")
    part_to_place_rules = part_to_place_data.get("attachment_rules", {})
    existing_part_areas = get_part_attachment_areas(existing_part, existing_part_world_center, existing_part_world_angle)

    for existing_area_name, existing_area_info in existing_part_areas.items():
        # Check distance
        dist_sq = (existing_area_info["world_pos"] - target_world_pos).length_squared()
        if dist_sq < snap_distance_sq:
            # Check if the existing part allows the new part type here
            if part_to_place_type in existing_area_info["allowed_types"]:
                # Now check if the new part *has* a compatible attachment rule
                # (e.g., if existing is "bottom_center", new needs a "top_center")
                compatible_new_area = None
                if "bottom" in existing_area_name and "top_center" in part_to_place_rules:
                    compatible_new_area = "top_center"
                elif "top" in existing_area_name and "bottom_center" in part_to_place_rules:
                    compatible_new_area = "bottom_center"
                # Add side rules etc.

                if compatible_new_area:
                    # Check if the new part allows the existing part's type (optional, usually implied)
                    # new_rule = part_to_place_rules[compatible_new_area]
                    # if existing_part.part_data['type'] in new_rule.get('allowed_types', []):
                    return existing_area_name, compatible_new_area, existing_area_info["world_pos"]

    return None, None, None # No valid attachment found