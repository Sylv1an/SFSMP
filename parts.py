# parts.py
import pygame

PART_TYPES = ["CommandPod", "FuelTank", "Engine"]

PARTS_CATALOG = {
    "pod_mk1": {
        "name": "Mk1 Command Pod", "type": "CommandPod", "mass": 50,
        "width": 20, "height": 25, "color": (200, 200, 200),
        "max_hp": 200,  # Stronger pod
        "attachment_points": { "bottom": pygame.math.Vector2(0, 12.5) }
    },
    "tank_small": {
        "name": "Small Fuel Tank", "type": "FuelTank", "mass": 20,
        "fuel_capacity": 100, "width": 20, "height": 40, "color": (255, 255, 255),
        "max_hp": 100,  # Standard tank HP
        "attachment_points": {
            "top": pygame.math.Vector2(0, -20), "bottom": pygame.math.Vector2(0, 20),
            "left": pygame.math.Vector2(-10, 0), "right": pygame.math.Vector2(10, 0)
        }
    },
    "engine_basic": {
        "name": "Basic Liquid Engine", "type": "Engine", "mass": 30,
        "thrust": 9.81 * 8 * 150, "fuel_consumption": 0.5,
        "width": 18, "height": 20, "color": (100, 100, 100),
        "max_hp": 80,   # Engines are a bit more fragile
        "attachment_points": { "top": pygame.math.Vector2(0, -10) }
    }
}

def get_part_data(part_id): return PARTS_CATALOG.get(part_id)

def draw_part_shape(surface, part_data, center_pos, angle_deg=0, broken=False): # Add broken flag
    w = part_data["width"]; h = part_data["height"]
    base_color = part_data["color"]
    # Change color if broken
    color = (50, 0, 0) if broken else base_color # Dark red if broken

    points = [ pygame.math.Vector2(x, y) for x, y in [(-w/2,-h/2), (w/2,-h/2), (w/2,h/2), (-w/2,h/2)] ]
    rotated_points = [p.rotate(-angle_deg) for p in points]
    screen_points = [p + center_pos for p in rotated_points]
    pygame.draw.polygon(surface, color, screen_points)
    pygame.draw.polygon(surface, (50,50,50) if not broken else (100,50,50), screen_points, 1) # Outline also changes

    # Optional: Draw cracks or 'X' if broken
    if broken:
        pygame.draw.line(surface, (255,0,0), screen_points[0], screen_points[2], 2)
        pygame.draw.line(surface, (255,0,0), screen_points[1], screen_points[3], 2)