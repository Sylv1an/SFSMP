# parts.py
import pygame
import math # Added for aero calculations if needed later

# Add new types
PART_TYPES = ["CommandPod", "FuelTank", "Engine", "Parachute", "Separator"]

# --- Default Physical Properties ---
DEFAULT_MAX_TEMP = 1600.0 # Kelvin
DEFAULT_THERMAL_MASS_FACTOR = 150.0 # Joules per Kelvin per kg (Higher resists temp change)
DEFAULT_DRAG_COEFF = 0.8 # Dimensionless drag coefficient

PARTS_CATALOG = {
    # ... (pod_mk1) ...
    "pod_mk1": {
        "name": "Mk1 Command Pod", "type": "CommandPod", "mass": 50,
        "width": 20, "height": 25, "color": (200, 200, 200),
        "max_hp": 200,
        # --- Thermal & Aero ---
        "max_temp": DEFAULT_MAX_TEMP + 200, # Pods might be slightly more resistant
        "thermal_mass": 50 * DEFAULT_THERMAL_MASS_FACTOR, # mass * factor
        "base_drag_coeff": 0.6, # More aerodynamic shape
        # --- Attachments ---
        "attachment_rules": {
            "top_center": {"allowed_types": ["Parachute"]},
            "bottom_center": {"allowed_types": ["FuelTank", "Engine", "Separator"]},
            # *** ADDED side attachments to pod ***
            "left_center": {"allowed_types": ["Separator"]}, # Allow radial separators
            "right_center": {"allowed_types": ["Separator"]},
        },
        "logical_points": {
            "bottom": pygame.math.Vector2(0, 12.5), "top": pygame.math.Vector2(0, -12.5),
            # *** ADDED side points to pod ***
            "left": pygame.math.Vector2(-10, 0), "right": pygame.math.Vector2(10, 0),
        }
    },
    # ... (tank_small - WITH SIDE RULES) ...
    "tank_small": {
        "name": "Small Fuel Tank", "type": "FuelTank", "mass": 20,
        "fuel_capacity": 100, "width": 19, "height": 40, "color": (255, 255, 255),
        "max_hp": 100,
        # --- Thermal & Aero ---
        "max_temp": DEFAULT_MAX_TEMP - 100, # Tanks might be less resistant
        "thermal_mass": 20 * DEFAULT_THERMAL_MASS_FACTOR,
        "base_drag_coeff": DEFAULT_DRAG_COEFF,
        # --- Attachments ---
        "attachment_rules": {
             "top_center": {"allowed_types": ["CommandPod", "FuelTank", "Separator"]},
             "bottom_center": {"allowed_types": ["FuelTank", "Engine", "Separator"]},
             # *** Side rules allow attaching radial separators OR other tanks/engines directly ***
             "left_center": {"allowed_types": ["FuelTank", "Separator", "Engine"]},
             "right_center": {"allowed_types": ["FuelTank", "Separator", "Engine"]},
        },
        "logical_points": {
            "top": pygame.math.Vector2(0, -20), "bottom": pygame.math.Vector2(0, 20),
            "left": pygame.math.Vector2(-9.5, 0), "right": pygame.math.Vector2(9.5, 0) # Adjusted slightly for width 19
        }
    },
    # ... (engine_basic) ...
    "engine_basic": {
        "name": "Basic Liquid Engine", "type": "Engine", "mass": 30,
        "thrust": 9.81 * 8 * 150, "fuel_consumption": 0.5,
        "width": 18, "height": 20, "color": (100, 100, 100),
        "max_hp": 80,
        # --- Thermal & Aero ---
        "max_temp": DEFAULT_MAX_TEMP + 400, # Engines are built for heat
        "thermal_mass": 30 * DEFAULT_THERMAL_MASS_FACTOR * 1.5, # Denser/more resistant
        "base_drag_coeff": 0.9, # Nozzle might create drag
        # --- Attachments ---
        "attachment_rules": {
             "top_center": {"allowed_types": ["CommandPod", "FuelTank", "Separator"]},
             # Allow engines to attach radially (e.g., to side of tank or radial separator)
             "left_center": {"allowed_types": ["FuelTank", "Separator"]},
             "right_center": {"allowed_types": ["FuelTank", "Separator"]},
        },
         "logical_points": {
             "top": pygame.math.Vector2(0, -10),
             # Add side points for radial attachment
             "left": pygame.math.Vector2(-9, 0),
             "right": pygame.math.Vector2(9, 0),
         }
    },
    # ... (parachute_mk1) ...
    "parachute_mk1": {
        "name": "Parachute Mk1", "type": "Parachute", "mass": 10,
        "width": 15, "height": 10, "color": (200, 100, 0), # Orange when packed
        "max_hp": 50,
        "deploy_drag": 8.0, # This drag is added when deployed
        "deploy_area_factor": 10,
        "activatable": True,
        # --- Thermal & Aero ---
        "max_temp": DEFAULT_MAX_TEMP - 300, # Fabric burns easier
        "thermal_mass": 10 * DEFAULT_THERMAL_MASS_FACTOR * 0.5, # Light material
        "base_drag_coeff": 0.7, # Packed shape
        # --- Attachments ---
        "attachment_rules": {
             "bottom_center": {"allowed_types": ["CommandPod"]},
        },
        "logical_points": {"bottom": pygame.math.Vector2(0, 5)}
    },
    # ... (separator_tr_s1 - Transverse) ...
    "separator_tr_s1": {
        "name": "Transverse Separator S1", "type": "Separator", "mass": 15,
        "width": 20, "height": 8, "color": (200, 200, 0), # Yellow band
        "max_hp": 60,
        "separation_force": 8000,
        "separation_axis": "transverse", # Informative tag
        "activatable": True,
        # --- Thermal & Aero ---
        "max_temp": DEFAULT_MAX_TEMP,
        "thermal_mass": 15 * DEFAULT_THERMAL_MASS_FACTOR,
        "base_drag_coeff": DEFAULT_DRAG_COEFF,
        # --- Attachments ---
        "attachment_rules": {
             "top_center": {"allowed_types": ["CommandPod", "FuelTank", "Separator"]},
             "bottom_center": {"allowed_types": ["FuelTank", "Engine", "Separator"]},
             # Maybe allow radial attachment *to* this?
             "left_center": {"allowed_types": ["Separator"]},
             "right_center": {"allowed_types": ["Separator"]},
        },
         "logical_points": {
             "top": pygame.math.Vector2(0, -4), "bottom": pygame.math.Vector2(0, 4),
             "left": pygame.math.Vector2(-10, 0), "right": pygame.math.Vector2(10, 0),
         }
    },

    # *** NEW RADIAL SEPARATOR ***
    "separator_rd_s1": {
        "name": "Radial Separator S1", "type": "Separator", "mass": 8, # Lighter than transverse?
        "width": 8, "height": 15, # Taller than wide
        "color": (150, 150, 255), # Light blue
        "max_hp": 50,
        "separation_force": 6000, # Maybe slightly less force?
        "separation_axis": "radial", # Informative tag
        "activatable": True,
        # --- Thermal & Aero ---
        "max_temp": DEFAULT_MAX_TEMP,
        "thermal_mass": 8 * DEFAULT_THERMAL_MASS_FACTOR,
        "base_drag_coeff": DEFAULT_DRAG_COEFF,
        # --- Attachments ---
        "attachment_rules": {
             # "left_center": Attaches TO the core stack (Tank, Pod, another Separator)
             "left_center": {"allowed_types": ["FuelTank", "CommandPod", "Separator"]},
             # "right_center": Parts attach TO THIS side (facing outwards radially)
             "right_center": {"allowed_types": ["FuelTank", "Engine", "Separator"]},
             # Allow stacking radial separators vertically?
             "top_center": {"allowed_types": ["Separator"]},
             "bottom_center": {"allowed_types": ["Separator"]},
        },
         "logical_points": {
             # Corresponds to attachment rules
             "left": pygame.math.Vector2(-4, 0), # "Back" side attaching to core
             "right": pygame.math.Vector2(4, 0), # "Front" side facing outwards
             "top": pygame.math.Vector2(0, -7.5),
             "bottom": pygame.math.Vector2(0, 7.5),
         }
    },
}

def get_part_data(part_id):
    data = PARTS_CATALOG.get(part_id)
    if data:
        # Ensure defaults are present if missing from definition
        data.setdefault("max_temp", DEFAULT_MAX_TEMP)
        data.setdefault("thermal_mass", data.get("mass", 10) * DEFAULT_THERMAL_MASS_FACTOR)
        data.setdefault("base_drag_coeff", DEFAULT_DRAG_COEFF)
    return data


# Keep draw_part_shape, but add visual states (e.g., deployed parachute, overheating)
def draw_part_shape(surface, part_data, center_pos, angle_deg=0, broken=False, deployed=False, heat_factor=0.0): # Add heat_factor (0 to 1)
    w = part_data["width"]; h = part_data["height"]
    base_color = part_data["color"]
    color = (50, 0, 0) if broken else base_color # Dark red if broken

    # --- Overheating Glow ---
    if not broken and heat_factor > 0.0:
        # Interpolate towards a hot color (e.g., white-yellow-red)
        # Simple approach: Lerp towards bright red/orange
        hot_color = pygame.Color(255, 100, 0) # Target hot color
        current_color = pygame.Color(color)
        glow_color = current_color.lerp(hot_color, heat_factor)
        color = glow_color

    # Basic rectangle shape
    points = [ pygame.math.Vector2(x, y) for x, y in [(-w/2,-h/2), (w/2,-h/2), (w/2,h/2), (-w/2,h/2)] ]
    rotated_points = [p.rotate(-angle_deg) for p in points]
    screen_points = [p + center_pos for p in rotated_points]
    pygame.draw.polygon(surface, color, screen_points)
    pygame.draw.polygon(surface, (50,50,50) if not broken else (100,50,50), screen_points, 1) # Border

    # Draw deployed parachute visual (unchanged, but uses base_color potentially)
    if part_data["type"] == "Parachute" and deployed and not broken:
        deploy_w = w * 3; deploy_h = h * 2
        deploy_rect = pygame.Rect(0, 0, deploy_w, deploy_h * 2)
        top_offset = pygame.math.Vector2(0, -h/2 - 2); rotated_top_offset = top_offset.rotate(-angle_deg)
        deploy_center_screen = center_pos + rotated_top_offset - pygame.math.Vector2(0, deploy_h)
        deploy_rect.center = deploy_center_screen
        # Use a fixed color for deployed chute, ignore heat glow for this part
        pygame.draw.arc(surface, (255, 165, 0), deploy_rect, math.radians(20), math.radians(160), 4)

    if broken:
        pygame.draw.line(surface, (255,0,0), screen_points[0], screen_points[2], 2)
        pygame.draw.line(surface, (255,0,0), screen_points[1], screen_points[3], 2)

# --- Functions below remain largely unchanged ---

# import math # Already imported
def get_part_attachment_areas(part: 'PlacedPart', part_world_center: pygame.math.Vector2, part_world_angle: float) -> dict:
    """ Calculates the world positions of key attachment areas based on rules. """
    areas = {}; part_data = part.part_data; rules = part_data.get("attachment_rules", {}); points = part_data.get("logical_points", {})
    for area_name, rule_data in rules.items():
        point_name = None
        # Map rule area name back to logical point name
        if area_name == "top_center": point_name = "top"
        elif area_name == "bottom_center": point_name = "bottom"
        elif area_name == "left_center": point_name = "left"
        elif area_name == "right_center": point_name = "right"
        # *** ADDED back/front mapping for clarity, though underlying logic uses left/right ***
        elif area_name == "back_center": point_name = "back" # If you define points named "back"
        elif area_name == "front_center": point_name = "front" # If you define points named "front"

        if point_name and point_name in points:
            local_pos = points[point_name]; rotated_offset = local_pos.rotate(-part_world_angle); world_pos = part_world_center + rotated_offset
            areas[area_name] = {"world_pos": world_pos, "allowed_types": rule_data.get("allowed_types", [])}
    return areas

def can_attach(part_to_place_data: dict, existing_part: 'PlacedPart', existing_part_world_center: pygame.math.Vector2, existing_part_world_angle: float, target_world_pos: pygame.math.Vector2, snap_distance_sq: float) -> tuple[str | None, str | None, pygame.math.Vector2 | None]:
    """ Checks if part_to_place can attach to existing_part near target_world_pos.
        Returns: (attach_area_on_existing, attach_area_on_new, connection_world_pos) or (None, None, None)
    """
    part_to_place_type = part_to_place_data.get("type"); part_to_place_rules = part_to_place_data.get("attachment_rules", {})
    existing_part_areas = get_part_attachment_areas(existing_part, existing_part_world_center, existing_part_world_angle)
    for existing_area_name, existing_area_info in existing_part_areas.items():
        dist_sq = (existing_area_info["world_pos"] - target_world_pos).length_squared()
        if dist_sq < snap_distance_sq:
            if part_to_place_type in existing_area_info["allowed_types"]:
                # Check compatibility based on rule names (top connects to bottom, left connects to right)
                compatible_new_area = None
                if "bottom" in existing_area_name and "top_center" in part_to_place_rules: compatible_new_area = "top_center"
                elif "top" in existing_area_name and "bottom_center" in part_to_place_rules: compatible_new_area = "bottom_center"
                elif "left" in existing_area_name and "right_center" in part_to_place_rules: compatible_new_area = "right_center"
                elif "right" in existing_area_name and "left_center" in part_to_place_rules: compatible_new_area = "left_center"
                # Add aliases if using front/back conceptually
                elif "front" in existing_area_name and "back_center" in part_to_place_rules: compatible_new_area = "back_center"
                elif "back" in existing_area_name and "front_center" in part_to_place_rules: compatible_new_area = "front_center"


                if compatible_new_area:
                    # Final check: Does the new part allow the existing part type on its compatible area?
                    new_part_allowed = part_to_place_rules.get(compatible_new_area, {}).get("allowed_types", [])
                    if existing_part.part_data.get("type") in new_part_allowed:
                        return existing_area_name, compatible_new_area, existing_area_info["world_pos"]

    return None, None, None