import os
import json
from pathlib import Path
from utils import main_path
        
results_dir = os.path.join(main_path, "results", "final_world_states")
targets_dir = os.path.join(main_path, "data", "structures", "gold-processed")

def parse_structure(json_data):
    """Convert list-of-blocks JSON into dict with coordinate keys and color values."""
    return {(b["x"], b["y"], b["z"]): b["block_color"] for b in json_data}

def normalize_structure_coords(structure):
    """Shift all block coordinates so the structure starts at (0, 0, 0), to avoid penalties for global shifts."""
    if not structure:
        return {}

    min_x = min(coord[0] for coord in structure)
    min_y = min(coord[1] for coord in structure)
    min_z = min(coord[2] for coord in structure)

    normalized = {
        (x - min_x, y - min_y, z - min_z): color
        for (x, y, z), color in structure.items()
    }
    return normalized

def compute_custom_edit_distance(built_structure, target_structure, tol=1):
    """
    Compute the custom edit distance similarity score between the built and target structures
    according to following costs for various operations, with tolerance = 1 for blocks positions differences

    Costs:
    | Edit Operation     | Description                                 | Suggested Cost |
    | ------------------ | ------------------------------------------- | -------------- |
    | Move               | Block exists in both, but position differs  | 0.5            |
    | Recolor            | Block in same position, wrong color         | 0.3            |
    | Move + Recolor     | Both position and color are wrong           | 0.8            |
    | Insert             | Block exists in target but not in predicted | 1.0            |
    | Delete             | Extra block in predicted                    | 1.0            |

    Returns the similarity score based on the above operations.
    """
    total_cost = 0

    # Initialize the set of target coordinates for easy lookup
    target_coords = set(target_structure.keys())

    for b_coord, b_color in built_structure.items():
        if b_coord in target_coords:
            t_color = target_structure[b_coord]
            if b_color == t_color:
                continue  # No cost if position and color match

            # If only color is different, count a recolor
            total_cost += 0.3  # Recolor cost
        else:
            # If the position is wrong, apply move cost if above the tolerance threshold
            closest_target = min(target_structure.keys(),  # Compute the distance between each block in the target structure and the current one in the built structure, retain the smallest.
                                 key=lambda t_coord: abs(t_coord[0] - b_coord[0]) + abs(t_coord[1] - b_coord[1]) + abs(t_coord[2] - b_coord[2]))
            position_diff = abs(closest_target[0] - b_coord[0]) + abs(
                closest_target[1] - b_coord[1]) + abs(closest_target[2] - b_coord[2])  # Computes the actual distance between b_coord and closest_target

            if position_diff <= tol:
                # Within tolerance, check for color differences
                t_color = target_structure[closest_target]
                if b_color != t_color:
                    total_cost += 0.3  # Recolor cost applied

            else:
                total_cost += 0.5  # Move cost for out of tolerance shifts
                # Check if the color is also different
                t_color = target_structure[closest_target]
                if b_color != t_color:
                    total_cost += 0.3 # Recolor cost applied

    # Now check for insertions and deletions
    for t_coord in target_structure.keys():
        if t_coord not in built_structure:
            total_cost += 1.0  # Insert cost

    for b_coord in built_structure.keys():
        if b_coord not in target_structure:
            total_cost += 1.0  # Delete cost

    # Calculate the similarity as the inverse of the total cost
    max_cost = len(target_structure) + \
        len(built_structure)  # Maximum possible cost
    similarity = 1 - (total_cost / max_cost) if max_cost > 0 else 1.0

    return round(similarity, 2)

# Map file names to full paths
results_map = {
    f.strip().lower(): os.path.join(results_dir, f)
    for f in os.listdir(results_dir)
    if f.endswith(".json") and os.path.isfile(os.path.join(results_dir, f))
}

targets_map = {}
for root, _, files in os.walk(targets_dir):
    for f in files:
        if f.endswith(".json"):
            clean_name = f.strip().lower()
            full_path = os.path.join(root, f)
            targets_map[clean_name] = full_path

# Match files by name
matching_names = results_map.keys() & targets_map.keys()

# Loop over matching pairs
for name in sorted(matching_names):
    result_path = results_map[name]
    target_path = targets_map[name]

    with open(result_path, 'r', encoding='utf-8') as rf, open(target_path, 'r', encoding='utf-8') as tf:
        result_data = json.load(rf)
        target_data = json.load(tf)
    
    # Parse to dict keyed by coordinates
    result_struct = parse_structure(result_data)
    target_struct = parse_structure(target_data)

    # Normalize coordinates to ignore global shifts
    norm_result = normalize_structure_coords(result_struct)
    norm_target = normalize_structure_coords(target_struct)

    # Comput similarity score
    similarity_score = compute_custom_edit_distance(
        norm_result, norm_target, tol=1)
    print(f"File: {name}, Similarity Score: {similarity_score}")


   

