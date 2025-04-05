# rocket_data.py
import json
import pygame
import os
from parts import get_part_data, PARTS_CATALOG
from collections import deque

class PlacedPart:
    def __init__(self, part_id, relative_center_pos, angle=0, connections=None):
        self.part_id = part_id
        self.part_data = get_part_data(part_id)
        if not self.part_data:
            raise ValueError(f"Invalid part_id: {part_id}")

        # Position of the part's CENTER relative to the blueprint's origin (0,0)
        # This should be grid-aligned in the builder
        self.relative_pos = pygame.math.Vector2(relative_center_pos)
        # Angle is relative to the blueprint's orientation (usually 0 for grid)
        self.relative_angle = angle

        # Runtime flight state (initialized here, managed by FlyingRocket)
        self.current_hp = self.part_data.get("max_hp", 100)
        self.is_broken = False
        self.engine_enabled = True # For engines
        self.deployed = False      # For parachutes
        self.separated = False     # For separators
        self.part_index = -1       # Set by FlyingRocket during init

        # Connections might be needed for complex separation logic later
        self.connections = connections if connections is not None else {}

    def to_dict(self):
        # Convert to a dictionary for saving to JSON
        return {
            "part_id": self.part_id,
            # Save the center position
            "pos_x": self.relative_pos.x,
            "pos_y": self.relative_pos.y,
            "angle": self.relative_angle,
        }

    @classmethod
    def from_dict(cls, data):
        # Create a PlacedPart instance from a dictionary (loaded from JSON)
        pos = pygame.math.Vector2(data["pos_x"], data["pos_y"])
        # Pass the loaded position as the relative_center_pos
        return cls(data["part_id"], pos, data.get("angle", 0))

    def __hash__(self):
        # Hash based on something unique within the blueprint instance,
        # like its position or a unique ID if we assigned one.
        # Using position tuple is simple for grid/unique positions.
        return hash((self.part_id, self.relative_pos.x, self.relative_pos.y))

    def __eq__(self, other):
        if not isinstance(other, PlacedPart):
            return NotImplemented
        return (self.part_id == other.part_id and
                self.relative_pos == other.relative_pos)

class RocketBlueprint:
    def __init__(self, name="My Rocket"):
        self.name = name
        self.parts = [] # List of PlacedPart objects, order matters (root is parts[0])

    def add_part(self, part_id, relative_center_pos, angle=0):
        # Ensure position is Vector2
        pos = pygame.math.Vector2(relative_center_pos)
        # Check if this exact position is already occupied (optional strict check)
        # for p in self.parts:
        #    if p.relative_pos == pos:
        #        print(f"Warning: Part already exists at {pos}")
        #        return # Prevent stacking directly on top

        placed_part = PlacedPart(part_id, pos, angle)
        self.parts.append(placed_part)
        # Re-sort parts? Or assume builder manages order? For now, append.
        # If adding root, ensure it's first?
        # self.ensure_root_first() # Might be needed if adding order changes

    def remove_part(self, part_to_remove: PlacedPart):
         # Simple removal by object identity. Need checks for attached parts?
         if part_to_remove in self.parts:
             # TODO: Add logic here? If removing a part breaks the rocket in two?
             # For now, just remove the single part. The builder prevents orphans.
             # Need to handle removing the ROOT part carefully!
             if self.parts.index(part_to_remove) == 0 and len(self.parts) > 1:
                 print("Warning: Removing the root part! This might break the blueprint.")
                 # Optionally prevent root removal if children exist, or designate a new root.
                 # Simple approach: Allow removal, user must fix.

             self.parts.remove(part_to_remove)
             print(f"Removed {part_to_remove.part_id} from blueprint.")
             # Consider recursively removing parts attached ONLY to the removed part if needed.
         else:
             print("Attempted to remove a part not in the blueprint.")

    def get_part_at_world_pos(self, world_pos: pygame.math.Vector2, tolerance=5) -> PlacedPart | None:
         """ Finds a part whose bounding box contains the world position. """
         # Note: world_pos here is relative to the blueprint origin (0,0)
         for part in reversed(self.parts): # Check topmost parts first
             part_data = part.part_data
             w, h = part_data['width'], part_data['height']
             part_rect = pygame.Rect(0, 0, w, h)
             part_rect.center = part.relative_pos # Use stored relative pos
             # Add tolerance to rect
             if part_rect.inflate(tolerance*2, tolerance*2).collidepoint(world_pos):
                 return part
         return None


    def get_total_mass(self):
        # Ignores fuel mass for now
        if not self.parts: return 0.01
        return sum(p.part_data.get("mass", 0) for p in self.parts)

    # ... (get_engines, get_fuel_tanks, get_total_fuel_capacity remain similar) ...
    def get_engines(self):
        return [p for p in self.parts if p.part_data.get("type") == "Engine"]

    def get_fuel_tanks(self):
        return [p for p in self.parts if p.part_data.get("type") == "FuelTank"]

    def get_total_fuel_capacity(self):
        return sum(p.part_data.get("fuel_capacity", 0) for p in self.parts)

    def get_lowest_point_offset_y(self) -> float:
            """ Recalculated based on relative_pos being center """
            if not self.parts: return 0.0
            max_y_offset = -float('inf')
            for part in self.parts:
                part_data = part.part_data
                part_center_y = part.relative_pos.y
                part_half_height = part_data.get('height', 0) / 2.0
                part_bottom_y = part_center_y + part_half_height # Assumes angle 0
                max_y_offset = max(max_y_offset, part_bottom_y)
            return max_y_offset if max_y_offset > -float('inf') else 0.0

    def get_part_bounding_box(self, part: PlacedPart) -> pygame.Rect:
        """ Helper to get the Rect for a part based on its center and size. """
        part_data = part.part_data
        w = part_data.get('width', 1)
        h = part_data.get('height', 1)
        rect = pygame.Rect(0, 0, w, h)
        rect.center = part.relative_pos  # relative_pos is world center
        return rect

    def find_connected_subassemblies(self) -> list[list[PlacedPart]]:
        """
        Finds distinct groups of connected parts within the blueprint.
        Uses proximity check between compatible logical attachment points.
        """
        if not self.parts: return []

        all_parts_set = set(self.parts)
        visited = set()
        subassemblies = []
        # How close logical points need to be to be considered connected (squared)
        # Should be very small, allowing for slight float inaccuracies
        CONNECTION_TOLERANCE_SQ = (2.0)**2 # e.g., 2x2 pixels squared

        # Pre-calculate logical point world positions for efficiency
        part_world_points = {}
        for part in self.parts:
            points = part.part_data.get("logical_points", {})
            world_points = {}
            for name, local_pos in points.items():
                # Assume relative_angle is 0 for connectivity check simplicity
                world_points[name] = part.relative_pos + local_pos.rotate(-part.relative_angle)
            part_world_points[part] = world_points


        while len(visited) < len(self.parts):
            start_node = None
            for part in self.parts: # Iterate in defined order
                 if part not in visited: start_node = part; break
            if start_node is None: break # Should not happen

            current_assembly = []
            queue = deque([start_node])
            visited.add(start_node)

            while queue:
                current_part = queue.popleft()
                current_assembly.append(current_part)
                current_part_data = current_part.part_data
                current_rules = current_part_data.get("attachment_rules", {})
                current_points = part_world_points.get(current_part, {})

                # Find neighbors by checking compatible attachment points nearby
                for other_part in all_parts_set:
                    if other_part not in visited:
                        other_part_data = other_part.part_data
                        other_rules = other_part_data.get("attachment_rules", {})
                        other_points = part_world_points.get(other_part, {})

                        # Check all pairs of logical points between current_part and other_part
                        is_connected = False
                        for current_point_name, current_world_pos in current_points.items():
                            for other_point_name, other_world_pos in other_points.items():
                                # 1. Check proximity
                                if (current_world_pos - other_world_pos).length_squared() < CONNECTION_TOLERANCE_SQ:
                                    # 2. Check attachment rule compatibility
                                    # Map point names to rule names (can be helper function)
                                    current_rule_name = self._get_rule_name_for_point(current_point_name, current_rules)
                                    other_rule_name = self._get_rule_name_for_point(other_point_name, other_rules)

                                    if current_rule_name and other_rule_name:
                                        # Check if current allows other's type AND other allows current's type
                                        # AND if the rules are compatible (e.g., top connects to bottom)
                                        if self._check_rule_compatibility(current_part_data.get("type"), current_rule_name, current_rules,
                                                                          other_part_data.get("type"), other_rule_name, other_rules):
                                            is_connected = True
                                            break # Found a valid connection between these parts
                            if is_connected:
                                break # Stop checking points for this pair

                        if is_connected:
                            visited.add(other_part)
                            queue.append(other_part) # Add connected neighbor to queue

            if current_assembly:
                subassemblies.append(current_assembly)

        # --- Optional: Sort subassemblies (same as before) ---
        def sort_key(assembly):
            for part in assembly:
                if part.part_data.get("type") == "CommandPod":
                    try:
                        if self.parts and part == self.parts[0]: return 0 # Original first part (likely root)
                        else: return 1 # Other command pods
                    except IndexError: return 1
            return 2 # Assemblies without command pods
        subassemblies.sort(key=sort_key)

        print(f"Connectivity check found {len(subassemblies)} subassemblies.")
        # for i, asm in enumerate(subassemblies): print(f"  Assembly {i}: {[p.part_id for p in asm]}")
        return subassemblies

    # --- Helper methods for connectivity check ---
    def _get_rule_name_for_point(self, point_name: str, rules: dict) -> str | None:
            """ Maps a logical point name (e.g., 'bottom') to its corresponding rule name (e.g., 'bottom_center'). """
            if "bottom" in point_name and "bottom_center" in rules: return "bottom_center"
            if "top" in point_name and "top_center" in rules: return "top_center"
            if "left" in point_name and "left_center" in rules: return "left_center"
            if "right" in point_name and "right_center" in rules: return "right_center"
            # Add more mappings if needed
            return None

    def _check_rule_compatibility(self, type1, rule_name1, rules1, type2, rule_name2, rules2) -> bool:
            """ Checks if two attachment rules are compatible for connection. """
            # 1. Check if each rule allows the other part's type
            allowed1 = rules1.get(rule_name1, {}).get("allowed_types", [])
            allowed2 = rules2.get(rule_name2, {}).get("allowed_types", [])
            if type2 not in allowed1 or type1 not in allowed2:
                return False

            # 2. Check if the rule types themselves are compatible pairs
            if ("top" in rule_name1 and "bottom" in rule_name2) or \
                    ("bottom" in rule_name1 and "top" in rule_name2) or \
                    ("left" in rule_name1 and "right" in rule_name2) or \
                    ("right" in rule_name1 and "left" in rule_name2):
                # Add other compatible pairs if needed (e.g., side-to-top)
                return True

            return False

    def save_to_json(self, filename):
        # Ensure the assets directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        data = {
            "name": self.name,
            "parts": [p.to_dict() for p in self.parts]
        }
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
            # print(f"Blueprint saved to {filename}") # Less verbose saving
        except IOError as e:
            print(f"Error saving blueprint: {e}")

    @classmethod
    def load_from_json(cls, filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            blueprint = cls(data.get("name", "Unnamed Rocket"))
            # Load parts in the order they were saved
            blueprint.parts = [PlacedPart.from_dict(part_data) for part_data in data.get("parts", [])]

            if not blueprint.parts:
                 print(f"Warning: Loaded blueprint '{filename}' is empty.")
            elif not blueprint.parts[0].part_data.get("type") == "CommandPod":
                 print(f"Warning: Loaded blueprint '{filename}' root part is not a CommandPod.")

            print(f"Blueprint loaded from {filename}")
            return blueprint
        except FileNotFoundError:
            print(f"Error: Blueprint file not found: {filename}")
            return None
        except (IOError, json.JSONDecodeError, ValueError, KeyError) as e: # Added KeyError
            print(f"Error loading blueprint '{filename}': {e}")
            return None

# Example Usage (test loading/saving)
if __name__ == "__main__":
    # Create a blueprint using grid logic conceptually
    grid_bp = RocketBlueprint("Grid Test")
    grid_bp.add_part("pod_mk1", (0, 0)) # Root at 0,0
    # Tank center below pod center. Pod H=25, Tank H=40. Pod Bot = 12.5, Tank Top = -20.
    # Connection point is pod_bottom. Target pos for tank = conn_point - tank_top_attach
    # Target pos = (0, 12.5) - (0, -20) = (0, 32.5) -> Tank Center = (0, 32.5) relative to pod center.
    # Assume pod center is (0,0) in blueprint space.
    pod_data = get_part_data("pod_mk1")
    tank_data = get_part_data("tank_small")
    pod_bottom_local = pod_data['logical_points']['bottom'] # (0, 12.5)
    tank_top_local = tank_data['logical_points']['top']     # (0, -20)
    # tank_center_relative_to_pod_center = pod_bottom_local - tank_top_local
    # Since pod is at (0,0), this is the tank's relative pos
    tank_rel_pos = pod_bottom_local - tank_top_local # (0, 32.5)
    tank_rel_pos_snapped = pygame.math.Vector2(round(tank_rel_pos.x / 10) * 10, round(tank_rel_pos.y / 10) * 10) # Snap (0, 30)
    grid_bp.add_part("tank_small", tank_rel_pos_snapped)

    # Engine below tank. Tank bot = (0, 20), Eng top = (0, -10). Conn = tank_bottom
    # Eng center rel to tank center = tank_bottom_local - eng_top_local
    # Eng center rel to tank center = (0, 20) - (0, -10) = (0, 30)
    # Eng center rel to blueprint origin = tank_rel_pos + eng_center_rel_to_tank_center
    engine_data = get_part_data("engine_basic")
    tank_bottom_local = tank_data['logical_points']['bottom'] # (0, 20)
    engine_top_local = engine_data['logical_points']['top'] # (0, -10)
    engine_rel_to_tank = tank_bottom_local - engine_top_local # (0, 30)
    engine_rel_pos = tank_rel_pos_snapped + engine_rel_to_tank # (0, 30) + (0, 30) = (0, 60)
    engine_rel_pos_snapped = pygame.math.Vector2(round(engine_rel_pos.x / 10)*10, round(engine_rel_pos.y/10)*10) # Snap (0, 60)
    grid_bp.add_part("engine_basic", engine_rel_pos_snapped)

    # Add parachute on top of pod
    # Pod top = (0, -12.5), Para bot = (0, 5). Conn = pod_top
    # Para center rel to pod = pod_top_local - para_bot_local
    # Para center rel to pod = (0, -12.5) - (0, 5) = (0, -17.5)
    # Para center rel to blueprint = pod_rel_pos (0,0) + para_rel_to_pod
    parachute_data = get_part_data("parachute_mk1")
    pod_top_local = pod_data['logical_points']['top'] # (0, -12.5)
    para_bottom_local = parachute_data['logical_points']['bottom'] # (0, 5)
    para_rel_to_pod = pod_top_local - para_bottom_local # (0, -17.5)
    para_rel_pos = pygame.math.Vector2(0,0) + para_rel_to_pod # (0, -17.5)
    para_rel_pos_snapped = pygame.math.Vector2(round(para_rel_pos.x/10)*10, round(para_rel_pos.y/10)*10) # Snap (0, -20)
    grid_bp.add_part("parachute_mk1", para_rel_pos_snapped)


    grid_bp.save_to_json("assets/grid_rocket_test.json")

    loaded = RocketBlueprint.load_from_json("assets/grid_rocket_test.json")
    if loaded:
        print(f"\nLoaded Grid Blueprint: {loaded.name}")
        for p in loaded.parts:
            print(f" - {p.part_id} at {p.relative_pos}")