'''
This script is a variation of the actions_evaluation.py script in the main branch. 
It substitutes the use of accuracy, precision and iou as evaluation metrics to a custom edit distance metric.
It also normalizes the coordinates of both target and generated structures to start at (0, 0, 0), avoiding penalties for global shifts.
'''


import json
import os
from collections import defaultdict
from utils import main_path
from judge_utils import results_path, load_all_results, extract_conversation_data


def normalize_structure_coords(structure):
    """ Shift all block coordinates so the structure starts at (0, 0, 0), to avoid penalties for global shifts. """
    if not structure:
        return {}

    # Find the minimum x, y, z coordinates to understand how shifted the structure is w.r.t. the origin
    min_x = min(coord[0] for coord in structure)
    min_y = min(coord[1] for coord in structure)
    min_z = min(coord[2] for coord in structure)

    # Normalize by subtracting the minimum values so that the smallest coordinate becomes (0, 0, 0) and the other coordinates are modified accordingly
    normalized = {
        (x - min_x, y - min_y, z - min_z): color
        for (x, y, z), color in structure.items()
    }
    return normalized


def parse_actions(results_data, actions_file):
    """ Extract the Builder's actions and other relevant metadata from the results and organizes it into a JSON list of dictionaries. """
    actions_file = os.path.join(main_path, "analysis", "parsed_actions.json")

    all_actions = []

    for index, row in results_data.iterrows():

        # Open and load the JSON file
        json_path = os.path.join(results_path, row["json_file"])
        try:
            _, b_actions_list = extract_conversation_data(json_path)
            compound = {
                'index': index,
                'structure_id': row['structure_id'],
                'finished_by_architect': row['finished_by_architect'],
                'use_img': row['use_img'],
                'use_json': row['use_json'],
                'shot': row['shot'],
                'actions': b_actions_list,
            }
            all_actions.append(compound)
        except:
            continue

    # Update the JSON file with the new results
        with open(actions_file, "w") as f:
            json.dump(all_actions, f, indent=4)


def parse_coordinate(coord):
    """
    Parse a coordinate value to a float.

    Returns a tuple (value, is_approximated) where:
      - value: the float conversion of the coordinate.
      - is_approximated: True if the original type was not an int.
          (i.e., if coord is a float or string, it's marked as approximated)

    Raises ValueError if the coordinate cannot be converted.
    """
    if isinstance(coord, int):
        return float(coord), False
    elif isinstance(coord, float):
        # Even if the float is mathematically an integer (e.g., 98.0),
        # we mark it as approximated because its type is float.
        return coord, True
    elif isinstance(coord, str):
        try:
            value = float(coord)
        except ValueError:
            raise ValueError("Coordinate string cannot be converted to float")
        return value, True
    else:
        raise ValueError("Unsupported type for coordinate")


def process_actions(actions_list):
    """
    Process a list of action dictionaries to build the structure.

    For each "add" item (list of 4 elements) and "remove" item (list of 3 elements),
    the coordinates are parsed and converted to floats. This function also tracks:
      - valid_count: number of valid coordinate actions processed.
      - approximated_flag: True if any coordinate came in as float or string.

    Returns a tuple: (built_structure, valid_count, approximated_flag)
      - built_structure: dictionary mapping (x, y, z) to block color.
    """
    built_structure = {}
    valid_count = 0
    approximated_flag = False

    for action in actions_list:
        if not action:
            continue  # Skip empty actions

        # Process "add" items.
        for add_item in action.get("add", []):
            if not isinstance(add_item, list) or len(add_item) != 4:
                continue
            try:
                x_val, is_approx_x = parse_coordinate(add_item[0])
                y_val, is_approx_y = parse_coordinate(add_item[1])
                z_val, is_approx_z = parse_coordinate(add_item[2])
            except Exception:
                continue
            # Mark as approximated if any coordinate was not a pure int.
            if is_approx_x or is_approx_y or is_approx_z:
                approximated_flag = True
            valid_count += 1
            color = add_item[3]
            built_structure[(x_val, y_val, z_val)] = color

        # Process "remove" items.
        for remove_item in action.get("remove", []):
            if not isinstance(remove_item, list) or len(remove_item) != 3:
                continue
            try:
                x_val, is_approx_x = parse_coordinate(remove_item[0])
                y_val, is_approx_y = parse_coordinate(remove_item[1])
                z_val, is_approx_z = parse_coordinate(remove_item[2])
            except Exception:
                continue
            if is_approx_x or is_approx_y or is_approx_z:
                approximated_flag = True
            valid_count += 1
            built_structure.pop((x_val, y_val, z_val), None)

    return built_structure, valid_count, approximated_flag


def process_target_structure(target_file_path):
    """
    Load and process the target structure file.

    Converts the target block coordinates to floats and returns a dictionary
    mapping (x, y, z) to the block color.
    """
    with open(target_file_path, "r") as f:
        target_data = json.load(f)
    target_structure = {}
    for block in target_data:
        try:
            x = float(block["x"])
            y = float(block["y"])
            z = float(block["z"])
            color = block["block_color"]
        except (KeyError, ValueError):
            continue
        target_structure[(x, y, z)] = color
    return target_structure


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


def reorder_structure_keys(structure, metrics):
    """
    Reorder keys in the structure dictionary to include metrics and action format.

    Adds the similarity score and keeps the "actions" key at the end.
    """
    new_structure = {}
    for key, value in structure.items():
        if key != "actions":
            new_structure[key] = value

    # Add metrics and action format before actions
    new_structure.update(metrics)

    if "actions" in structure:
        new_structure["actions"] = structure["actions"]

    return new_structure


def evaluate_structure(structure_data, target_file_path):
    """
    Evaluate a structure's actions against the target structure.

    Processes the actions, computes the evaluation metrics, and determines the
    overall action format:
      - "incorrect" if no valid actions were found.
      - "approximated" if any coordinate was provided as a float or string.
      - "correct" if all coordinates were provided as Python integers.

    Returns a dictionary with metrics: similarity and action_format.
    """
    actions_list = structure_data.get("actions", [])
    built_structure, valid_count, approximated_flag = process_actions(
        actions_list)

    if valid_count == 0:
        action_format = "incorrect"
    elif approximated_flag:
        action_format = "approximated"
    else:
        action_format = "correct"

    try:
        target_structure = process_target_structure(target_file_path)
    except Exception:
        return None

    # Normalize both structures before computing similarity
    normalized_built = normalize_structure_coords(built_structure)
    normalized_target = normalize_structure_coords(target_structure)

    # Compute the custom similarity score
    similarity = compute_custom_edit_distance(
        normalized_built, normalized_target)

    # Prepare the metrics, including similarity
    metrics = {
        "similarity": similarity,
        "action_format": action_format
    }
    

    # Reorder the structure with the new metrics
    new_structure = reorder_structure_keys(structure_data, metrics)

    # Update the original structure data with the new structure
    structure_data.clear()
    structure_data.update(new_structure)

    return structure_data


def main():
    actions_file = os.path.join(main_path, "results", "new_results", "C100_wannabe_olympic_rings", "")
    if os.path.exists(actions_file):
        pass
    else:
        results_data = load_all_results(results_path)
        results_data = results_data.sort_values(
            'run_time').reset_index(drop=True)
        parse_actions(results_data, actions_file)

    with open(actions_file, "r") as f:
        actions_list = json.load(f)

    for structure in actions_list:
        structure_id = structure.get("structure_id")
        if not structure_id:
            continue

        # Construct the target file path; adjust the folder name as needed.
        gold_processed_path = os.path.join(
            main_path, "data", "structures", "gold-processed", "C100_wannabe_olympic_rings", "C100_wannabe_olympic_rings.json")
        target_file_path = os.path.join(
            gold_processed_path, structure_id, f"{structure_id}.json")
        if not os.path.exists(target_file_path):
            continue

        # Evaluate structure and update with similarity score
        metrics = evaluate_structure(structure, target_file_path)
        if metrics is None:
            continue

        new_structure = reorder_structure_keys(structure, metrics)
        structure.clear()
        structure.update(new_structure)

    actions_metrics_file = os.path.join(main_path, "analysis", "parsed_actions_with_metrics_new_results3.json")
    with open(actions_metrics_file, "w") as out_f:
        json.dump(actions_list, out_f, indent=4)



if __name__ == "__main__":
    main()
