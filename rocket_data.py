# rocket_data.py
import json
import pygame
from parts import get_part_data  # Import the function to access part details

class PlacedPart:
    def __init__(self, part_id, position, angle=0, connections=None):
        self.part_id = part_id  # e.g., "pod_mk1"
        self.part_data = get_part_data(part_id)  # Get the specs from parts.py
        if not self.part_data:
            raise ValueError(f"Invalid part_id: {part_id}")

        # Position and angle *relative to the root part* (usually the first command pod)
        self.relative_pos = pygame.math.Vector2(position)
        self.relative_angle = angle  # Degrees

        # Connections could store which attachment point connects to which part/point
        # For now, we can infer connections based on proximity in the builder
        self.connections = connections if connections is not None else {}

    def to_dict(self):
        # Convert to a dictionary for saving to JSON
        return {
            "part_id": self.part_id,
            "pos_x": self.relative_pos.x,
            "pos_y": self.relative_pos.y,
            "angle": self.relative_angle,
            # Save connections later if needed
        }

    @classmethod
    def from_dict(cls, data):
        # Create a PlacedPart instance from a dictionary (loaded from JSON)
        pos = pygame.math.Vector2(data["pos_x"], data["pos_y"])
        return cls(data["part_id"], pos, data.get("angle", 0))


class RocketBlueprint:
    def __init__(self, name="My Rocket"):
        self.name = name
        self.parts = [] # List of PlacedPart objects

    def add_part(self, part_id, position, angle=0):
        placed_part = PlacedPart(part_id, position, angle)
        self.parts.append(placed_part)

    def get_total_mass(self):
        # Ignores fuel mass for now
        return sum(p.part_data["mass"] for p in self.parts)

    def get_engines(self):
        return [p for p in self.parts if p.part_data["type"] == "Engine"]

    def get_fuel_tanks(self):
        return [p for p in self.parts if p.part_data["type"] == "FuelTank"]

    def get_total_fuel_capacity(self):
        return sum(p.part_data.get("fuel_capacity", 0) for p in self.parts)

    # Add methods for center of mass, total thrust, drawing the blueprint etc. later
    def get_lowest_point_offset_y(self) -> float:
            """
            Calculates the Y-offset of the lowest point of the rocket assembly
            relative to the root part's origin (0,0).
            Assumes angle 0 for all parts for simplicity in this calculation.
            Returns 0 if no parts exist.
            """
            if not self.parts:
                return 0.0

            max_y_offset = -float('inf')  # Initialize to a very small number

            for part in self.parts:
                part_data = part.part_data
                part_relative_center_y = part.relative_pos.y
                part_half_height = part_data.get('height', 0) / 2.0

                # Calculate the Y-coordinate of the bottom edge of this part relative to the root origin
                part_bottom_y = part_relative_center_y + part_half_height

                # Keep track of the maximum Y value found so far
                max_y_offset = max(max_y_offset, part_bottom_y)

            # If somehow no parts had positive height, default to 0? Should use root bottom.
            if max_y_offset == -float('inf'):
                if self.parts:  # If parts exist but calculation failed, use root part's bottom
                    root_part = self.parts[0]
                    return root_part.relative_pos.y + root_part.part_data.get('height', 0) / 2.0
                else:
                    return 0.0  # No parts, offset is 0

            return max_y_offset

    def save_to_json(self, filename):
        data = {
            "name": self.name,
            "parts": [p.to_dict() for p in self.parts]
        }
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Blueprint saved to {filename}")
        except IOError as e:
            print(f"Error saving blueprint: {e}")

    @classmethod
    def load_from_json(cls, filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            blueprint = cls(data.get("name", "Unnamed Rocket"))
            blueprint.parts = [PlacedPart.from_dict(part_data) for part_data in data.get("parts", [])]
            print(f"Blueprint loaded from {filename}")
            # Add validation here - check if parts exist, etc.
            return blueprint
        except FileNotFoundError:
            print(f"Error: Blueprint file not found: {filename}")
            return None
        except (IOError, json.JSONDecodeError, ValueError) as e:
            print(f"Error loading blueprint: {e}")
            return None


# --- Example Usage (can be run directly for testing) ---
if __name__ == "__main__":
    # Create a default blueprint programmatically
    default_bp = RocketBlueprint("Default Stack")
    # Add parts - position is relative to the *first* part added (root)
    # NOTE: These positions assume parts stack vertically downwards (+Y)
    # and attachment points align. A real builder calculates these.
    pod_data = get_part_data("pod_mk1")
    tank_data = get_part_data("tank_small")
    engine_data = get_part_data("engine_basic")

    default_bp.add_part("pod_mk1", (0, 0))  # Root part at (0,0) relative coords

    # Tank attaches to bottom of pod
    pod_bottom_attach = pod_data["attachment_points"]["bottom"]  # e.g., (0, 12.5)
    tank_top_attach = tank_data["attachment_points"]["top"]  # e.g., (0, -20)
    # Relative position of tank center = pod_bottom_attach - tank_top_attach
    tank_pos = pod_bottom_attach - tank_top_attach  # Should be (0, 32.5)
    default_bp.add_part("tank_small", tank_pos)

    # Engine attaches to bottom of tank
    tank_bottom_attach = tank_data["attachment_points"]["bottom"]  # e.g., (0, 20)
    engine_top_attach = engine_data["attachment_points"]["top"]  # e.g., (0, -10)
    # Relative pos of engine center = tank_pos + (tank_bottom_attach - engine_top_attach)
    engine_pos = tank_pos + (tank_bottom_attach - engine_top_attach)  # Should be (0, 32.5) + (0, 30) = (0, 62.5)
    default_bp.add_part("engine_basic", engine_pos)

    # Save it
    default_bp.save_to_json("assets/default_rocket.json")

    # Load it back
    loaded_bp = RocketBlueprint.load_from_json("assets/default_rocket.json")
    if loaded_bp:
        print(f"Loaded blueprint: {loaded_bp.name}")
        print(f"Total Mass (dry): {loaded_bp.get_total_mass()}")
        print(f"Engines: {len(loaded_bp.get_engines())}")
        print(f"Fuel Capacity: {loaded_bp.get_total_fuel_capacity()}")
        print("Parts:")
        for part in loaded_bp.parts:
            print(f" - {part.part_data['name']} at {part.relative_pos}")