import os
import json
from utils import main_path
import math # Needed for potential mirroring if you add it, but not strictly for rotations

# --- Core Structure Handling Functions ---
# --- Core Structure Handling Functions ---
def parse_structure(json_data):
    """
    Convert JSON data (either a direct list of block dictionaries
    or a dictionary containing a 'dialogue' key with block dictionaries)
    into a dictionary with (x, y, z) tuples as keys and block_color as values.
    """
    block_list_to_parse = []

    if isinstance(json_data, dict) and "dialogue" in json_data:
        # This is the new format for generated files (dict with 'dialogue' key)
        block_list_to_parse = json_data["dialogue"]
    elif isinstance(json_data, list):
        # This is the old format for generated files OR the target files (direct list of blocks)
        block_list_to_parse = json_data
    else:
        # Handle unexpected formats gracefully
        print(f"Warning: Unexpected JSON data format encountered in parse_structure. Type: {type(json_data)}. Expected list or dict with 'dialogue' key.")
        return {} # Return an empty structure to avoid further errors

    # Now, process the actual list of blocks
    return {(block["x"], block["y"], block["z"]): block["block_color"] for block in block_list_to_parse}

# (All other functions in your script remain the same)

def normalize_structure_coords(structure):
    """
    Shift all block coordinates so the structure starts at (0, 0, 0),
    to avoid penalties for global shifts.
    Input `structure` is a dictionary: (x, y, z) -> color.
    Returns a new dictionary with normalized coordinates.
    """
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

# --- Rotation Functions ---
def rotate_x_90(x, y, z):
    """Rotate +90 degrees around the X-axis."""
    return (x, -z, y)

def rotate_y_90(x, y, z):
    """Rotate +90 degrees around the Y-axis."""
    return (z, y, -x)

def rotate_z_90(x, y, z):
    """Rotate +90 degrees around the Z-axis."""
    return (-y, x, z)

def apply_rotation_to_structure(structure, rotation_function):
    """
    Applies a given rotation function to all blocks in a structure.
    Input `structure` is a dictionary: (x, y, z) -> color.
    Returns a new dictionary with rotated coordinates.
    """
    rotated_structure = {}
    for (x, y, z), color in structure.items():
        new_x, new_y, new_z = rotation_function(x, y, z)
        rotated_structure[(new_x, new_y, new_z)] = color
    return rotated_structure

# --- Canonical Orientation Function (This is the robust version) ---
def get_all_canonical_orientations(structure):
    """
    Generates all 24 unique 90-degree rotational orientations of a structure,
    each normalized to start at (0, 0, 0).
    Input `structure` is a dictionary: (x, y, z) -> color.
    Returns a set of frozensets, where each frozenset represents a unique
    canonical orientation of the structure (a set of (x, y, z, color) tuples).
    """
    if not structure:
        return {frozenset()} # Return a set containing one empty frozenset for empty structures

    canonical_orientations = set()
    
    # Start with the initial normalized structure
    initial_normalized_structure = normalize_structure_coords(structure)
    
    # Generate all 24 unique 90-degree rotations
    current_rot = initial_normalized_structure
    
    # Rotate around Y-axis (4 times for 0, 90, 180, 270 degrees)
    for _ in range(4):
        # Rotate around X-axis (4 times for 0, 90, 180, 270 degrees)
        # This will get all 6 faces pointing "up" (or "forward", etc.)
        current_rot_x = current_rot 
        for _ in range(4):
            # Normalize and add the current orientation to the set
            normalized_orientation = normalize_structure_coords(current_rot_x)
            canonical_orientations.add(frozenset(normalized_orientation.items()))
            
            # Apply Z-rotation (if desired, but typically covered by X/Y combinations for 24)
            # This part is key to ensuring all 24 are covered, but it's simpler
            # to think of it as 6 faces X 4 rotations.
            
            # For each X-Y combination, also consider Z rotations for completeness:
            current_rot_z = current_rot_x
            for _ in range(4):
                normalized_z = normalize_structure_coords(current_rot_z)
                canonical_orientations.add(frozenset(normalized_z.items()))
                current_rot_z = apply_rotation_to_structure(current_rot_z, rotate_z_90)


            current_rot_x = apply_rotation_to_structure(current_rot_x, rotate_x_90)
        current_rot = apply_rotation_to_structure(current_rot, rotate_y_90)

    # A more systematic way to generate exactly 24 unique rotations:
    generated_orientations_dicts = set()
    initial_norm = normalize_structure_coords(structure)

    # Start with base (identity) rotation
    generated_orientations_dicts.add(frozenset(initial_norm.items()))

    # Apply all combinations of rotations
    for rx_count in range(4): # 0, 90, 180, 270 deg around X
        current_x_rot = initial_norm
        for _ in range(rx_count):
            current_x_rot = apply_rotation_to_structure(current_x_rot, rotate_x_90)
        
        for ry_count in range(4): # 0, 90, 180, 270 deg around Y
            current_xy_rot = current_x_rot
            for _ in range(ry_count):
                current_xy_rot = apply_rotation_to_structure(current_xy_rot, rotate_y_90)
            
            for rz_count in range(4): # 0, 90, 180, 270 deg around Z
                current_xyz_rot = current_xy_rot
                for _ in range(rz_count):
                    current_xyz_rot = apply_rotation_to_structure(current_xyz_rot, rotate_z_90)
                
                # Normalize and add to the set
                normalized_form = normalize_structure_coords(current_xyz_rot)
                generated_orientations_dicts.add(frozenset(normalized_form.items()))
    
    return generated_orientations_dicts


# --- Jaccard Similarity Function ---
def compute_structure_similarity(structure1_dict, structure2_dict):
    """
    Computes a similarity score (Jaccard Index) between two structures,
    considering all their canonical orientations.
    A score of 1.0 means identical, 0.0 means no overlap.
    Input `structure1_dict` and `structure2_dict` are dictionaries: (x, y, z) -> color.
    """
    if not structure1_dict and not structure2_dict:
        return 1.0 # Both empty structures are perfectly similar
    if not structure1_dict or not structure2_dict:
        return 0.0 # One empty, one not, means no similarity

    orientations1 = get_all_canonical_orientations(structure1_dict)
    orientations2 = get_all_canonical_orientations(structure2_dict)

    max_similarity = 0.0

    # Iterate through all combinations of orientations to find the best match
    for o1 in orientations1:
        for o2 in orientations2:
            # o1 and o2 are frozensets of (coord, color) tuples
            intersection_size = len(o1.intersection(o2))
            union_size = len(o1.union(o2))
            
            if union_size == 0:
                # This case should ideally be handled by the initial empty structure checks,
                # but as a safeguard, if both frozensets are empty within the loop,
                # it means the original dicts were effectively empty.
                current_similarity = 1.0 
            else:
                current_similarity = intersection_size / union_size
            
            if current_similarity > max_similarity:
                max_similarity = current_similarity
                
    return max_similarity


# --- Main Evaluation Function ---
def evaluate_structures_and_metrics(results_dir, targets_dir, experimental_condition_name, main_path):
    """
    Evaluate the built structures against the target structures for a specific experimental condition,
    calculate classification metrics, and store detailed results.

    Args:
        results_dir (str): Path to the directory containing generated structure JSONs for this condition.
        targets_dir (str): Path to the directory containing target structure JSONs.
        experimental_condition_name (str): Name of the current experimental condition (e.g., "condition_A").
        main_path (str): The base path where the 'analysis' directory is located.
    """
    results_map = {
        f.strip().lower(): os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(".json") and os.path.isfile(os.path.join(results_dir, f))
    }

    targets_map = {}
    for root, _, files in os.walk(targets_dir): # os.walk allows targets_dir to have subdirectories
        for f in files:
            if f.endswith(".json"):
                clean_name = f.strip().lower()
                full_path = os.path.join(root, f)
                targets_map[clean_name] = full_path

    # Match files by name (structure_id is the filename without extension)
    matching_names = results_map.keys() & targets_map.keys()
    
    if not matching_names:
        print(f" No matching files found for {os.path.basename(os.path.normpath(results_dir))}")
        return {
            "true_positives": 0, "false_positives": 0, "false_negatives": 0,
            "precision": 0.0, "recall": 0.0, "f1_score": 0.0,
            "details": {}
        }

    all_results_with_scores = {}
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # --- Step 1: Process matched pairs for TP/FP ---
    for name in sorted(matching_names):
        result_path = results_map[name]
        target_path = targets_map[name]

        with open(result_path, 'r', encoding='utf-8') as rf, \
             open(target_path, 'r', encoding='utf-8') as tf:
            try: # Added try-except for robust JSON loading
                result_data_raw = json.load(rf)
                target_data_raw = json.load(tf)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for {name}: {e}. Skipping this pair.")
                continue # Skip to the next file if JSON is invalid

        # Parse to dict keyed by coordinates
        result_struct_dict = parse_structure(result_data_raw)
        target_struct_dict = parse_structure(target_data_raw)

        # --- Use compute_structure_similarity and threshold ---
        similarity_score = compute_structure_similarity(target_struct_dict, result_struct_dict)
        similarity_threshold = 0.80 # Adjust this value as needed!
        is_correct_match = (similarity_score >= similarity_threshold)
        
        if is_correct_match:
            true_positives += 1
            evaluation_status = f"MATCH (Score: {similarity_score:.2f})"
        else:
            false_positives += 1
            evaluation_status = f"MISMATCH (Score: {similarity_score:.2f})"
        
        all_results_with_scores[name] = {
            "status": evaluation_status,
            "similarity_score": similarity_score,
            "threshold_used": similarity_threshold,
            "generated_structure_path": result_path,
            "target_structure_path": target_path,
            # Only store a preview to avoid huge files if full data is large
            "generated_data_preview": result_data_raw[:5] if isinstance(result_data_raw, list) else "..."
        }
        print(f"File: {name}, Status: {evaluation_status}")

    # --- Step 2: Account for False Negatives (targets that were not generated) ---
    unmatched_target_ids = set(targets_map.keys()) - set(matching_names)
    false_negatives += len(unmatched_target_ids)

    for name in sorted(unmatched_target_ids):
        all_results_with_scores[name] = {
            "status": "NOT_GENERATED",
            "target_structure_path": targets_map[name],
            "generated_structure_path": None,
            "generated_data_preview": None
        }
        print(f"File: {name}, Status: NOT_GENERATED")

    # --- Step 3: Account for additional False Positives (generated structures without a target) ---
    unmatched_generated_ids = set(results_map.keys()) - set(targets_map.keys())
    false_positives += len(unmatched_generated_ids)
    
    for name in sorted(unmatched_generated_ids):
        # Need to re-load for preview if it's an unmatched generated file
        try:
            with open(results_map[name], 'r', encoding='utf-8') as f:
                generated_preview_data = json.load(f)
                generated_preview_data = generated_preview_data[:5] if isinstance(generated_preview_data, list) else "..."
        except json.JSONDecodeError:
            generated_preview_data = "Error loading JSON for preview"

        all_results_with_scores[name] = {
            "status": "UNEXPECTED_GENERATED",
            "generated_structure_path": results_map[name],
            "target_structure_path": None,
            "generated_data_preview": generated_preview_data
        }
        print(f"File: {name}, Status: UNEXPECTED_GENERATED")


    # Calculate overall metrics for this experimental condition
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }
    
    print(f"\nMetrics for {experimental_condition_name}:")
    print(f"  TP: {true_positives}, FP: {false_positives}, FN: {false_negatives}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1-Score: {f1_score:.2f}")

    # Save detailed results for this condition
    output_filename = f"evaluation_results_{experimental_condition_name}.json"
    analysis_output_path = os.path.join(main_path, "analysis", output_filename)
    os.makedirs(os.path.dirname(analysis_output_path), exist_ok=True) # Ensure 'analysis' dir exists
    
    with open(analysis_output_path, 'w', encoding='utf-8') as outfile:
        json.dump({"metrics": metrics, "details": all_results_with_scores}, outfile, indent=2)
    print(f"Detailed results for {experimental_condition_name} saved to {analysis_output_path}")

    return metrics # Return the metrics for aggregation if evaluating multiple conditions

# --- Main execution loop for experimental conditions ---
if __name__ == '__main__':
    # You no longer need to define base_project_path here if you're importing main_path
    # The 'main_path' variable is already available from your 'utils' import.

    targets_base_dir = os.path.join(main_path, "data", "structures", "20_new_project")
    results_base_dir = os.path.join(main_path, "results", "final_world_states") # Parent of condition_1, condition_2 etc.
    analysis_output_dir = os.path.join(main_path, "analysis") # Where to save results

    # Ensure analysis output directory exists
    os.makedirs(analysis_output_dir, exist_ok=True)

    # Define your experimental condition folder names
    experimental_conditions = ["img_only", "json_only", "img_json"] # Adjust to your actual folder names

    all_conditions_metrics = {}

    for condition_name in experimental_conditions:
        current_results_dir = os.path.join(results_base_dir, condition_name)
        
        # Check if the results directory for the condition exists
        if not os.path.isdir(current_results_dir):
            print(f"\n--- Skipping Condition: {condition_name} ---")
            print(f"Results directory not found: {current_results_dir}")
            continue # Skip to the next condition

        print(f"\n--- Evaluating Condition: {condition_name} ---")
        
        metrics = evaluate_structures_and_metrics(
            results_dir=current_results_dir,
            targets_dir=targets_base_dir,
            experimental_condition_name=condition_name,
            main_path=main_path # Pass the imported main_path
        )
        all_conditions_metrics[condition_name] = metrics

    print("\n--- Overall Metrics Across Conditions ---")
    for condition, metrics in all_conditions_metrics.items():
        print(f"\nCondition: {condition}")
        print(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
        print(f"  Precision: {metrics['precision']:.2f}")
        print(f"  Recall: {metrics['recall']:.2f}")
        print(f"  F1-Score: {metrics['f1_score']:.2f}")

    # Optionally, save aggregated metrics
    aggregated_metrics_path = os.path.join(analysis_output_dir, "aggregated_metrics.json")
    with open(aggregated_metrics_path, 'w', encoding='utf-8') as outfile:
        json.dump(all_conditions_metrics, outfile, indent=2)
    print(f"\nAggregated metrics saved to {aggregated_metrics_path}")