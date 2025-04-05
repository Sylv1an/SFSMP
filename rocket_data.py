# rocket_data.py
import json
import pygame
import os
from parts import get_part_data, PARTS_CATALOG # Keep PARTS_CATALOG import if needed by helpers
from collections import deque

# Define ambient temp here or import from flight_sim constants later
AMBIENT_TEMPERATURE = 293.0 # Kelvin (~20C)

class PlacedPart:
    def __init__(self, part_id, relative_center_pos, angle=0, connections=None):
        self.part_id = part_id
        self.part_data = get_part_data(part_id)
        if not self.part_data:
            raise ValueError(f"Invalid part_id: {part_id}")

        self.relative_pos = pygame.math.Vector2(relative_center_pos)
        self.relative_angle = angle

        # Runtime flight state (initialized here, managed by FlyingRocket)
        self.current_hp = self.part_data.get("max_hp", 100)
        self.is_broken = False
        self.engine_enabled = True
        self.deployed = False
        self.separated = False
        self.part_index = -1
        # --- NEW: Thermal State ---
        self.current_temp = AMBIENT_TEMPERATURE # Initialize to ambient K
        self.is_overheating = False # Flag for visual effects

        self.connections = connections if connections is not None else {}

    def to_dict(self):
        return {
            "part_id": self.part_id,
            "pos_x": self.relative_pos.x,
            "pos_y": self.relative_pos.y,
            "angle": self.relative_angle,
            # Note: Runtime state like temp, hp, deployed is NOT saved in blueprint
        }

    @classmethod
    def from_dict(cls, data):
        pos = pygame.math.Vector2(data["pos_x"], data["pos_y"])
        instance = cls(data["part_id"], pos, data.get("angle", 0))
        # Runtime state is reset when loaded into simulation
        return instance

    def __hash__(self):
        # Using position should be sufficient for builder uniqueness
        return hash((self.part_id, self.relative_pos.x, self.relative_pos.y))

    def __eq__(self, other):
        if not isinstance(other, PlacedPart):
            return NotImplemented
        # Equality check based on builder properties
        return (self.part_id == other.part_id and
                self.relative_pos == other.relative_pos and
                self.relative_angle == other.relative_angle)


class RocketBlueprint:
    def __init__(self, name="My Rocket"):
        self.name = name
        self.parts = [] # List of PlacedPart objects

    def add_part(self, part_id, relative_center_pos, angle=0):
        pos = pygame.math.Vector2(relative_center_pos)
        # Simple check to prevent exact overlaps during build
        for p in self.parts:
           if p.relative_pos == pos:
               print(f"Warning: Part already exists exactly at {pos}")
               # return # Or allow it, depending on desired build behavior

        placed_part = PlacedPart(part_id, pos, angle)
        self.parts.append(placed_part)

    def remove_part(self, part_to_remove: PlacedPart):
         try:
             # Need to handle potential structural breaks if not root
             # The builder's connectivity check relies on parts list, so remove first.
             original_index = self.parts.index(part_to_remove)
             self.parts.remove(part_to_remove)
             print(f"Removed {part_to_remove.part_id} from blueprint.")
             # Note: Connectivity check should happen *after* removal in builder
             # to see if the structure is still sound.
             if original_index == 0 and self.parts:
                 print("Warning: Root part removed. Connectivity might be broken.")

         except ValueError:
             print("Attempted to remove a part not in the blueprint.")


    def get_part_at_world_pos(self, world_pos: pygame.math.Vector2, tolerance=5) -> PlacedPart | None:
         """ Finds a part whose bounding box contains the world position (relative to blueprint origin). """
         for part in reversed(self.parts):
             part_data = part.part_data
             w, h = part_data['width'], part_data['height']
             # Assuming angle 0 in builder for simple AABB check
             part_rect = pygame.Rect(0, 0, w, h)
             part_rect.center = part.relative_pos
             if part_rect.inflate(tolerance*2, tolerance*2).collidepoint(world_pos):
                 return part
         return None

    # --- Methods below remain the same ---

    def get_total_mass(self):
        if not self.parts: return 0.01
        return sum(p.part_data.get("mass", 0) for p in self.parts)

    def get_engines(self):
        return [p for p in self.parts if p.part_data.get("type") == "Engine"]

    def get_fuel_tanks(self):
        return [p for p in self.parts if p.part_data.get("type") == "FuelTank"]

    def get_total_fuel_capacity(self):
        return sum(p.part_data.get("fuel_capacity", 0) for p in self.parts)

    def get_lowest_point_offset_y(self) -> float:
            if not self.parts: return 0.0
            max_y_offset = -float('inf')
            for part in self.parts:
                part_data = part.part_data; part_center_y = part.relative_pos.y
                part_half_height = part_data.get('height', 0) / 2.0
                part_bottom_y = part_center_y + part_half_height # Assumes angle 0
                max_y_offset = max(max_y_offset, part_bottom_y)
            return max_y_offset if max_y_offset > -float('inf') else 0.0

    def get_part_bounding_box(self, part: PlacedPart) -> pygame.Rect:
        part_data = part.part_data; w = part_data.get('width', 1); h = part_data.get('height', 1)
        rect = pygame.Rect(0, 0, w, h); rect.center = part.relative_pos
        return rect

    def find_connected_subassemblies(self) -> list[list[PlacedPart]]:
        if not self.parts: return []
        all_parts_set = set(self.parts); visited = set(); subassemblies = []
        CONNECTION_TOLERANCE_SQ = (2.0)**2
        part_world_points = {}
        for part in self.parts:
            points = part.part_data.get("logical_points", {}); world_points = {}
            for name, local_pos in points.items():
                world_points[name] = part.relative_pos + local_pos.rotate(-part.relative_angle)
            part_world_points[part] = world_points

        while len(visited) < len(self.parts):
            start_node = next((part for part in self.parts if part not in visited), None)
            if start_node is None: break
            current_assembly = []; queue = deque([start_node]); visited.add(start_node)
            while queue:
                current_part = queue.popleft(); current_assembly.append(current_part)
                current_part_data = current_part.part_data; current_rules = current_part_data.get("attachment_rules", {})
                current_points = part_world_points.get(current_part, {})
                for other_part in all_parts_set:
                    if other_part not in visited:
                        other_part_data = other_part.part_data; other_rules = other_part_data.get("attachment_rules", {})
                        other_points = part_world_points.get(other_part, {}); is_connected = False
                        for cpn, cwp in current_points.items():
                            for opn, owp in other_points.items():
                                if (cwp - owp).length_squared() < CONNECTION_TOLERANCE_SQ:
                                    crn = self._get_rule_name_for_point(cpn, current_rules)
                                    orn = self._get_rule_name_for_point(opn, other_rules)
                                    if crn and orn and self._check_rule_compatibility(
                                        current_part_data.get("type"), crn, current_rules,
                                        other_part_data.get("type"), orn, other_rules):
                                        is_connected = True; break
                            if is_connected: break
                        if is_connected: visited.add(other_part); queue.append(other_part)
            if current_assembly: subassemblies.append(current_assembly)

        def sort_key(assembly): # Sort key remains the same
            root_part_type = "CommandPod"
            has_root = any(p.part_data.get("type") == root_part_type for p in assembly)
            is_original_root_assembly = False
            if self.parts and assembly and self.parts[0] in assembly and self.parts[0].part_data.get("type") == root_part_type:
                is_original_root_assembly = True

            if is_original_root_assembly: return 0
            elif has_root: return 1
            else: return 2
        subassemblies.sort(key=sort_key)

        # print(f"Connectivity check found {len(subassemblies)} subassemblies.") # Less verbose
        return subassemblies

    def _get_rule_name_for_point(self, point_name: str, rules: dict) -> str | None:
            if "bottom" in point_name and "bottom_center" in rules: return "bottom_center"
            if "top" in point_name and "top_center" in rules: return "top_center"
            if "left" in point_name and "left_center" in rules: return "left_center"
            if "right" in point_name and "right_center" in rules: return "right_center"
            return None

    def _check_rule_compatibility(self, type1, rule_name1, rules1, type2, rule_name2, rules2) -> bool:
            allowed1 = rules1.get(rule_name1, {}).get("allowed_types", []); allowed2 = rules2.get(rule_name2, {}).get("allowed_types", [])
            if type2 not in allowed1 or type1 not in allowed2: return False
            if ("top" in rule_name1 and "bottom" in rule_name2) or \
               ("bottom" in rule_name1 and "top" in rule_name2) or \
               ("left" in rule_name1 and "right" in rule_name2) or \
               ("right" in rule_name1 and "left" in rule_name2):
                return True
            return False

    # --- calculate_subassembly_world_com Added (moved from FlyingRocket for use here) ---
    def calculate_subassembly_world_com(self, assembly_parts: list[PlacedPart]) -> pygame.math.Vector2:
        """ Calculates the approximate center of mass for a subset of parts based on blueprint positions. """
        if not assembly_parts: return pygame.math.Vector2(0, 0)
        com_num = pygame.math.Vector2(0, 0)
        total_m = 0
        for part in assembly_parts:
            mass = part.part_data.get("mass", 1) # Use blueprint mass
            com_num += part.relative_pos * mass
            total_m += mass
        if total_m <= 0: return assembly_parts[0].relative_pos # Fallback
        return com_num / total_m

    def save_to_json(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True); data = {"name": self.name, "parts": [p.to_dict() for p in self.parts]}
        try:
            with open(filename, 'w') as f: json.dump(data, f, indent=4)
        except IOError as e: print(f"Error saving blueprint: {e}")

    @classmethod
    def load_from_json(cls, filename):
        try:
            with open(filename, 'r') as f: data = json.load(f)
            blueprint = cls(data.get("name", "Unnamed Rocket"))
            blueprint.parts = [PlacedPart.from_dict(part_data) for part_data in data.get("parts", [])]
            if not blueprint.parts: print(f"Warning: Loaded blueprint '{filename}' is empty.")
            # Removed root part check here, less critical for loading itself
            # print(f"Blueprint loaded from {filename}") # Less verbose
            return blueprint
        except FileNotFoundError: print(f"Error: Blueprint file not found: {filename}"); return None
        except (IOError, json.JSONDecodeError, ValueError, KeyError) as e: print(f"Error loading blueprint '{filename}': {e}"); return None


# Example Usage (remains the same)
if __name__ == "__main__":
    grid_bp = RocketBlueprint("Grid Test Aero")
    # Calculations based on logical points remain valid concept
    pod_data = get_part_data("pod_mk1"); tank_data = get_part_data("tank_small"); engine_data = get_part_data("engine_basic"); parachute_data = get_part_data("parachute_mk1")
    pod_pos = pygame.math.Vector2(0,0); grid_bp.add_part("pod_mk1", pod_pos)
    tank_rel_pos = pod_data['logical_points']['bottom'] - tank_data['logical_points']['top']; grid_bp.add_part("tank_small", pod_pos + tank_rel_pos)
    engine_rel_pos = tank_data['logical_points']['bottom'] - engine_data['logical_points']['top']; grid_bp.add_part("engine_basic", pod_pos + tank_rel_pos + engine_rel_pos)
    para_rel_pos = pod_data['logical_points']['top'] - parachute_data['logical_points']['bottom']; grid_bp.add_part("parachute_mk1", pod_pos + para_rel_pos)

    grid_bp.save_to_json("assets/grid_rocket_aero_test.json")
    loaded = RocketBlueprint.load_from_json("assets/grid_rocket_aero_test.json")
    if loaded:
        print(f"\nLoaded Grid Blueprint: {loaded.name}")
        for p in loaded.parts: print(f" - {p.part_id} at {p.relative_pos}, Temp: {p.current_temp:.1f}K") # Show initial temp