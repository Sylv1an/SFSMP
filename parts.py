# parts.py
import pygame

# Part dimensions are relative to their center point for now
PART_TYPES = ["CommandPod", "FuelTank", "Engine"]

PARTS_CATALOG = {
    "pod_mk1": {
        "name": "Mk1 Command Pod",
        "type": "CommandPod",
        "mass": 50,
        "width": 20,
        "height": 25,
        "color": (200, 200, 200),
        "attachment_points": { # Relative positions from center
            # name: Vector2(x, y)
            "bottom": pygame.math.Vector2(0, 12.5) # Positive Y is down
        }
    },
    "tank_small": {
        "name": "Small Fuel Tank",
        "type": "FuelTank",
        "mass": 20,
        "fuel_capacity": 100,
        "width": 20,
        "height": 40,
        "color": (255, 255, 255),
        "attachment_points": {
            "top": pygame.math.Vector2(0, -20),   # Negative Y is up
            "bottom": pygame.math.Vector2(0, 20),    # Positive Y is down
            "left": pygame.math.Vector2(-10, 0),  # Negative X is left
            "right": pygame.math.Vector2(10, 0)   # Positive X is right
        }
    },
    "engine_basic": {
        "name": "Basic Liquid Engine",
        "type": "Engine",
        "mass": 30,
        "thrust": 9.81 * 8 * 150,
        "fuel_consumption": 0.5,
        "width": 18,
        "height": 20,
        "color": (100, 100, 100),
        "attachment_points": {
            "top": pygame.math.Vector2(0, -10)
            # Engines typically don't have side/bottom attachments
        }
    }
    # Add more parts later (e.g., radial decouplers, structural parts)
}

def get_part_data(part_id):
    return PARTS_CATALOG.get(part_id)

# draw_part_shape function remains the same
def draw_part_shape(surface, part_data, center_pos, angle_deg=0):
    w = part_data["width"]
    h = part_data["height"]
    color = part_data["color"]
    points = [
        pygame.math.Vector2(-w/2, -h/2), pygame.math.Vector2( w/2, -h/2),
        pygame.math.Vector2( w/2,  h/2), pygame.math.Vector2(-w/2,  h/2),
    ]
    rotated_points = [p.rotate(-angle_deg) for p in points]
    screen_points = [p + center_pos for p in rotated_points]
    pygame.draw.polygon(surface, color, screen_points)
    pygame.draw.polygon(surface, (50,50,50), screen_points, 1) # Outline