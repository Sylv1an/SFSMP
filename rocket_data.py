# rocket_data.py
import json
import pygame
import os
from parts import get_part_data, PARTS_CATALOG # Keep PARTS_CATALOG import if needed by helpers
from collections import deque

# Define ambient temp here or import from flight_sim constants later
AMBIENT_TEMPERATURE = 293.0 # Kelvin (~20C)
# Define fuel mass per unit here (needed for physics calculations)
FUEL_MASS_PER_UNIT = 0.1 # kg per unit of fuel (adjust as needed)


class PlacedPart:
    def __init__(self, part_id, relative_center_pos, angle=0, connections=None):
        self.part_id = part_id
        self.part_data = get_part_data(part_id)
        if not self.part_data:
            raise ValueError(f"Invalid part_id: {part_id}")

        self.relative_pos = pygame.math.Vector2(relative_center_pos)
        self.relative_angle = angle

        # Runtime flight state (initialized here, managed by FlyingRocket)
        # These are the DEFAULTS. FlyingRocket.__init__ should preserve existing state if passed in.
        self.current_hp = self.part_data.get("max_hp", 100)
        self.is_broken = False
        self.engine_enabled = True # *** Engines default to ON ***
        self.deployed = False # Parachutes/etc start undeployed
        self.separated = False # Separators start unseparated
        self.part_index = -1 # Set by FlyingRocket during init/update
        # --- Thermal State ---
        self.current_temp = AMBIENT_TEMPERATURE # Initialize to ambient K
        self.is_overheating = False # Flag for visual effects

        # --- Fuel State (NEW) ---
        self.fuel_capacity = 0.0
        self.current_fuel = 0.0
        if self.part_data.get("type") == "FuelTank":
            self.fuel_capacity = self.part_data.get("fuel_capacity", 0)
            # Initialize current fuel to capacity by default (will be set/overridden by FlyingRocket)
            self.current_fuel = self.fuel_capacity

        # Builder-specific state, not used in flight sim directly?
        self.connections = connections if connections is not None else {}

    def to_dict(self):
        return {
            "part_id": self.part_id,
            "pos_x": self.relative_pos.x,
            "pos_y": self.relative_pos.y,
            "angle": self.relative_angle,
            # Runtime state (hp, temp, fuel, deployed etc.) is NOT saved in blueprint
        }

    @classmethod
    def from_dict(cls, data):
        pos = pygame.math.Vector2(data["pos_x"], data["pos_y"])
        # Create instance, which sets runtime defaults
        instance = cls(data["part_id"], pos, data.get("angle", 0))
        # Note: Fuel is initialized to capacity inside __init__ based on part_data
        return instance

    def __hash__(self):
        # Using position and ID should be sufficient for uniqueness within a blueprint/flight context
        return hash((self.part_id, self.relative_pos.x, self.relative_pos.y, self.relative_angle))

    def __eq__(self, other):
        if not isinstance(other, PlacedPart):
            return NotImplemented
        # Use hash for equality check might be faster if hash is good
        return hash(self) == hash(other)
        # Or explicit check:
        # return (self.part_id == other.part_id and
        #         self.relative_pos == other.relative_pos and
        #         self.relative_angle == other.relative_angle)


class RocketBlueprint:
    def __init__(self, name="My Rocket"):
        self.name = name
        self.parts: list[PlacedPart] = [] # List of PlacedPart objects
        # Add internal cache for connectivity lookups (blueprint context)
        self._part_connections_cache: dict[PlacedPart, list[PlacedPart]] | None = None

    def _build_connection_cache(self):
        """ Builds or rebuilds the cache mapping each part to its directly connected neighbors. """
        self._part_connections_cache = {}
        all_parts_set = set(self.parts)
        # Use a slightly larger tolerance to ensure connections are found reliably
        CONNECTION_TOLERANCE_SQ = (3.0)**2 # Increased tolerance slightly

        # Pre-calculate world points for all parts
        part_world_points = {}
        for part in self.parts:
            points = part.part_data.get("logical_points", {}); world_points = {}
            for name, local_pos in points.items():
                world_points[name] = part.relative_pos + local_pos.rotate(-part.relative_angle)
            part_world_points[part] = world_points

        for part in self.parts:
            self._part_connections_cache[part] = []
            current_part_data = part.part_data; current_rules = current_part_data.get("attachment_rules", {})
            current_points = part_world_points.get(part, {})

            for other_part in all_parts_set:
                if other_part == part: continue # Skip self

                other_part_data = other_part.part_data; other_rules = other_part_data.get("attachment_rules", {})
                other_points = part_world_points.get(other_part, {}); is_connected = False

                # Check connection between all points of current_part and other_part
                for cpn, cwp in current_points.items():
                    for opn, owp in other_points.items():
                        if (cwp - owp).length_squared() < CONNECTION_TOLERANCE_SQ:
                            # Points are close, check rule compatibility
                            crn = self._get_rule_name_for_point(cpn, current_rules)
                            orn = self._get_rule_name_for_point(opn, other_rules)
                            if crn and orn and self._check_rule_compatibility(
                                current_part_data.get("type"), crn, current_rules,
                                other_part_data.get("type"), orn, other_rules):
                                is_connected = True; break
                    if is_connected: break

                if is_connected:
                    self._part_connections_cache[part].append(other_part)

    def _invalidate_connection_cache(self):
        """ Call this whenever parts list changes. """
        self._part_connections_cache = None

    def get_connected_parts(self, part: PlacedPart) -> list[PlacedPart]:
        """ Returns a list of parts directly connected to the given part. Uses cache. """
        if self._part_connections_cache is None:
            self._build_connection_cache()
        # Ensure the part exists in the cache keys (might happen if called just after removal?)
        return self._part_connections_cache.get(part, [])


    def add_part(self, part_id, relative_center_pos, angle=0):
        pos = pygame.math.Vector2(relative_center_pos)
        # Simple check to prevent exact overlaps during build
        for p in self.parts:
           # Use a small tolerance for floating point comparisons
           if (p.relative_pos - pos).length_squared() < 1e-6:
               print(f"Warning: Part already exists very close to {pos}")
               # return # Or allow it, depending on desired build behavior

        placed_part = PlacedPart(part_id, pos, angle)
        self.parts.append(placed_part)
        self._invalidate_connection_cache() # Parts changed

    def remove_part(self, part_to_remove: PlacedPart):
         try:
             part_found = False
             for i, p in enumerate(self.parts):
                 if p == part_to_remove:
                     original_index = i
                     del self.parts[i]
                     part_found = True
                     print(f"Removed {part_to_remove.part_id} from blueprint.")
                     self._invalidate_connection_cache() # Parts changed
                     if original_index == 0 and self.parts:
                         print("Warning: Root part removed. Connectivity might be broken.")
                     break
             if not part_found:
                  print(f"Attempted to remove a part not found in the blueprint: {part_to_remove.part_id} at {part_to_remove.relative_pos}")

         except Exception as e: # Catch potential errors more broadly
             print(f"Error removing part: {e}")

    def get_part_at_world_pos(self, world_pos: pygame.math.Vector2, tolerance=5) -> PlacedPart | None:
         """ Finds a part whose bounding box contains the world position (relative to blueprint origin). """
         for part in reversed(self.parts): # Check topmost drawn parts first
             part_data = part.part_data
             w, h = part_data['width'], part_data['height']
             # Assuming angle 0 in builder for simple AABB check
             part_rect = pygame.Rect(0, 0, w, h)
             part_rect.center = part.relative_pos
             # Inflate rect slightly to make clicking easier
             if part_rect.inflate(tolerance*2, tolerance*2).collidepoint(world_pos):
                 return part
         return None

    # --- Methods below remain the same ---

    def get_total_mass(self):
        """ Calculates blueprint dry mass (ignores fuel). """
        if not self.parts: return 0.01
        # Use dry mass from part data for blueprint calculations
        return sum(p.part_data.get("mass", 0) for p in self.parts)

    def get_engines(self):
        return [p for p in self.parts if p.part_data.get("type") == "Engine"]

    def get_fuel_tanks(self):
        return [p for p in self.parts if p.part_data.get("type") == "FuelTank"]

    def get_total_fuel_capacity(self):
        """ Calculates total potential fuel capacity from all tanks in blueprint. """
        return sum(p.part_data.get("fuel_capacity", 0) for p in self.parts if p.part_data.get("type") == "FuelTank")

    def get_lowest_point_offset_y(self) -> float:
            if not self.parts: return 0.0
            max_y_offset = -float('inf')
            for part in self.parts:
                part_data = part.part_data; part_center_y = part.relative_pos.y
                # Approximation assuming angle 0 for simplicity in blueprint context
                part_half_height = part_data.get('height', 0) / 2.0
                part_bottom_y = part_center_y + part_half_height
                max_y_offset = max(max_y_offset, part_bottom_y)
            return max_y_offset if max_y_offset > -float('inf') else 0.0

    def get_part_bounding_box(self, part: PlacedPart) -> pygame.Rect:
        part_data = part.part_data; w = part_data.get('width', 1); h = part_data.get('height', 1)
        rect = pygame.Rect(0, 0, w, h); rect.center = part.relative_pos
        return rect

    def find_connected_subassemblies(self) -> list[list[PlacedPart]]:
        """ Finds structurally connected groups of parts using the connection cache. """
        if not self.parts: return []
        if self._part_connections_cache is None:
            self._build_connection_cache() # Ensure cache is built

        visited = set(); subassemblies = []
        all_parts_set = set(self.parts)

        while len(visited) < len(self.parts):
            start_node = next((part for part in self.parts if part not in visited), None)
            if start_node is None: break

            current_assembly = []; queue = deque([start_node]); visited.add(start_node)
            while queue:
                current_part = queue.popleft(); current_assembly.append(current_part)
                # Use the cached connections
                neighbors = self._part_connections_cache.get(current_part, [])
                for neighbor in neighbors:
                    if neighbor in all_parts_set and neighbor not in visited: # Check if neighbor still exists in blueprint
                        visited.add(neighbor)
                        queue.append(neighbor)

            if current_assembly:
                 subassemblies.append(current_assembly)


        # Sorting logic (unchanged)
        def sort_key(assembly):
            root_part_type = "CommandPod"
            has_cmd_pod = any(p.part_data.get("type") == root_part_type for p in assembly)
            is_original_root_assembly = bool(self.parts and assembly and self.parts[0] in assembly)

            if has_cmd_pod and is_original_root_assembly: return 0
            elif has_cmd_pod: return 1
            elif is_original_root_assembly: return 2
            else: return 3

        subassemblies.sort(key=sort_key)
        return subassemblies

    # --- Helper methods _get_rule_name_for_point and _check_rule_compatibility remain unchanged ---
    def _get_rule_name_for_point(self, point_name: str, rules: dict) -> str | None:
            if "bottom" in point_name and "bottom_center" in rules: return "bottom_center"
            if "top" in point_name and "top_center" in rules: return "top_center"
            if "left" in point_name and "left_center" in rules: return "left_center"
            if "right" in point_name and "right_center" in rules: return "right_center"
            return None

    def _check_rule_compatibility(self, type1, rule_name1, rules1, type2, rule_name2, rules2) -> bool:
            allowed_on_1 = rules1.get(rule_name1, {}).get("allowed_types", [])
            allowed_on_2 = rules2.get(rule_name2, {}).get("allowed_types", [])
            if type2 not in allowed_on_1 or type1 not in allowed_on_2: return False
            if ("top" in rule_name1 and "bottom" in rule_name2) or \
               ("bottom" in rule_name1 and "top" in rule_name2) or \
               ("left" in rule_name1 and "right" in rule_name2) or \
               ("right" in rule_name1 and "left" in rule_name2):
                return True
            return False

    # --- calculate_subassembly_world_com remains unchanged ---
    def calculate_subassembly_world_com(self, assembly_parts: list[PlacedPart]) -> pygame.math.Vector2:
        if not assembly_parts: return pygame.math.Vector2(0, 0)
        com_num = pygame.math.Vector2(0, 0); total_m = 0
        for part in assembly_parts:
            mass = part.part_data.get("mass", 1)
            com_num += part.relative_pos * mass
            total_m += mass
        if total_m <= 0:
             return assembly_parts[0].relative_pos if assembly_parts else pygame.math.Vector2(0,0)
        return com_num / total_m

    # --- save_to_json and load_from_json remain unchanged ---
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
            # Invalidate cache after loading new parts
            blueprint._invalidate_connection_cache()
            return blueprint
        except FileNotFoundError: print(f"Error: Blueprint file not found: {filename}"); return None
        except (IOError, json.JSONDecodeError, ValueError, KeyError) as e: print(f"Error loading blueprint '{filename}': {e}"); return None


# Example Usage (remains the same)
if __name__ == "__main__":
    # ... (example usage unchanged) ...
    grid_bp = RocketBlueprint("Grid Test Aero")
    pod_data = get_part_data("pod_mk1"); tank_data = get_part_data("tank_small"); engine_data = get_part_data("engine_basic"); parachute_data = get_part_data("parachute_mk1")
    pod_pos = pygame.math.Vector2(0,0); grid_bp.add_part("pod_mk1", pod_pos)
    tank_rel_pos = pod_data['logical_points']['bottom'] - tank_data['logical_points']['top']; grid_bp.add_part("tank_small", pod_pos + tank_rel_pos)
    engine_rel_pos = tank_data['logical_points']['bottom'] - engine_data['logical_points']['top']; grid_bp.add_part("engine_basic", pod_pos + tank_rel_pos + engine_rel_pos)
    para_rel_pos = pod_data['logical_points']['top'] - parachute_data['logical_points']['bottom']; grid_bp.add_part("parachute_mk1", pod_pos + para_rel_pos)

    # Test connection fetching
    print("Testing connections:")
    for part in grid_bp.parts:
        connections = grid_bp.get_connected_parts(part)
        print(f"  Part {part.part_id} at {part.relative_pos} connected to: {[p.part_id for p in connections]}")


    grid_bp.save_to_json("assets/grid_rocket_aero_test.json")
    loaded = RocketBlueprint.load_from_json("assets/grid_rocket_aero_test.json")
    if loaded:
        print(f"\nLoaded Grid Blueprint: {loaded.name}")
        for p in loaded.parts: print(f" - {p.part_id} at {p.relative_pos}, Fuel: {p.current_fuel}/{p.fuel_capacity}") # Show fuel state