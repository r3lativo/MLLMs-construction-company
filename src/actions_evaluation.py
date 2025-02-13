import json
import os
from utils import main_path
from judge_utils import results_path, load_all_results, extract_conversation_data


def parse_actions(results_data, actions_file):

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


def compute_metrics(built_structure, target_structure, tol=0.5):
    """
    Compute evaluation metrics using fuzzy matching with a tolerance.
    A built block is considered correct if for each coordinate the absolute difference 
    with the target coordinate is less than or equal to tol, and the colors match.
    """
    correct_count = 0
    matched_built = set()  # To avoid matching a built block more than once

    for t_coord, t_color in target_structure.items():
        for b_coord, b_color in built_structure.items():
            if b_coord in matched_built:
                continue
            # Check if each coordinate difference is within tolerance and colors match
            if all(abs(tc - bc) <= tol for tc, bc in zip(t_coord, b_coord)) and (t_color == b_color):
                correct_count += 1
                matched_built.add(b_coord)
                break

    total_target = len(target_structure)
    accuracy = correct_count / total_target if total_target > 0 else 0

    total_built = len(built_structure)
    precision = correct_count / total_built if total_built > 0 else 0

    # For IoU, we consider the union as the total unique blocks in both structures minus the correctly matched ones.
    union_count = len(target_structure) + len(built_structure) - correct_count
    iou = correct_count / union_count if union_count > 0 else 0

    return {
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "iou": round(iou, 2)
    }


def evaluate_structure(structure_data, target_file_path):
    """
    Evaluate a structure's actions against the target structure.
    
    Processes the actions, computes the evaluation metrics, and determines the
    overall action format:
      - "incorrect" if no valid actions were found.
      - "approximated" if any coordinate was provided as a float or string.
      - "correct" if all coordinates were provided as Python integers.
    
    Returns a dictionary with metrics: accuracy, precision, iou, and action_format.
    """
    actions_list = structure_data.get("actions", [])
    built_structure, valid_count, approximated_flag = process_actions(actions_list)

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

    metrics = compute_metrics(built_structure, target_structure)
    metrics["action_format"] = action_format
    return metrics


def reorder_structure_keys(structure, metrics):
    """
    Reorder keys in the structure dictionary so that:
      1. All original metadata keys (except "actions") come first.
      2. Then the evaluation metrics (including "action_format").
      3. Finally the "actions" key is added.
    
    Returns the reordered dictionary.
    """
    new_structure = {}
    for key, value in structure.items():
        if key != "actions":
            new_structure[key] = value
    # Add metrics before the "actions" key.
    new_structure.update(metrics)
    # Append the "actions" key at the end.
    if "actions" in structure:
        new_structure["actions"] = structure["actions"]
    return new_structure


def main():
    actions_file = os.path.join(main_path, "analysis", "parsed_actions.json")
    if os.path.exists(actions_file):
        pass
    else:
        results_data = load_all_results(results_path)
        results_data = results_data.sort_values('run_time').reset_index(drop=True)
        parse_actions(results_data, actions_file)

    
    with open(actions_file, "r") as f:
        actions_list = json.load(f)

    for structure in actions_list:
        structure_id = structure.get("structure_id")
        if not structure_id:
            continue

        # Construct the target file path; adjust the folder name as needed.
        gold_processed_path = os.path.join(main_path, "data", "structures", "gold-processed")
        target_file_path = os.path.join(gold_processed_path, structure_id, f"{structure_id}.json")
        if not os.path.exists(target_file_path):
            continue

        metrics = evaluate_structure(structure, target_file_path)
        if metrics is None:
            continue

        new_structure = reorder_structure_keys(structure, metrics)
        structure.clear()
        structure.update(new_structure)

    actions_metrics_file = os.path.join(main_path, "analysis", "parsed_actions_with_metrics.json")
    with open(actions_metrics_file, "w") as out_f:
        json.dump(actions_list, out_f, indent=4)


if __name__ == "__main__":
    main()
