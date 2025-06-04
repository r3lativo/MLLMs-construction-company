import os
import json
from utils import main_path # Assuming utils.py is in the correct place and contains main_path
import math

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

# Define all 24 90-degree rotations
# This list is used by normalize_and_compare to explicitly check rotations
ROTATION_FUNCTIONS = [
    lambda x, y, z: (x, y, z),       # Identity
    rotate_x_90,                     # X+90
    lambda x, y, z: (x, -y, -z),     # X+180
    lambda x, y, z: (x, z, -y),      # X+270

    rotate_y_90,                     # Y+90
    lambda x, y, z: (-x, y, -z),     # Y+180
    lambda x, y, z: (-z, y, x),      # Y+270

    rotate_z_90,                     # Z+90
    lambda x, y, z: (-x, -y, z),     # Z+180
    lambda x, y, z: (y, -x, z),      # Z+270

    # Combinations for remaining 14 orientations
    lambda x, y, z: rotate_x_90(*rotate_y_90(x,y,z)), # Y+90, X+90
    lambda x, y, z: rotate_x_90(*rotate_y_90(*rotate_y_90(x,y,z))), # Y+180, X+90
    lambda x, y, z: rotate_x_90(*rotate_y_90(*rotate_y_90(*rotate_y_90(x,y,z)))), # Y+270, X+90
    
    lambda x, y, z: rotate_x_90(*rotate_z_90(x,y,z)), # Z+90, X+90
    lambda x, y, z: rotate_x_90(*rotate_z_90(*rotate_z_90(x,y,z))), # Z+180, X+90
    lambda x, y, z: rotate_x_90(*rotate_z_90(*rotate_z_90(*rotate_z_90(x,y,z)))), # Z+270, X+90

    lambda x, y, z: rotate_y_90(*rotate_x_90(x,y,z)), # X+90, Y+90
    lambda x, y, z: rotate_y_90(*rotate_x_90(*rotate_x_90(x,y,z))), # X+180, Y+90
    lambda x, y, z: rotate_y_90(*rotate_x_90(*rotate_x_90(*rotate_x_90(x,y,z)))), # X+270, Y+90

    lambda x, y, z: rotate_y_90(*rotate_z_90(x,y,z)), # Z+90, Y+90
    lambda x, y, z: rotate_y_90(*rotate_z_90(*rotate_z_90(x,y,z))), # Z+180, Y+90
    lambda x, y, z: rotate_y_90(*rotate_z_90(*rotate_z_90(*rotate_z_90(x,y,z)))), # Z+270, Y+90

    lambda x, y, z: rotate_z_90(*rotate_x_90(x,y,z)), # X+90, Z+90
    lambda x, y, z: rotate_z_90(*rotate_y_90(x,y,z)), # Y+90, Z+90
]

# --- Helper for normalisation and comparison ---
def normalize_and_compare(target_struct_dict, generated_struct_dict):
    """
    Finds the best rotational alignment of the generated structure to the target structure,
    and returns the Jaccard similarity and block-level TP, FP, FN counts for that best alignment.
    """
    if not target_struct_dict and not generated_struct_dict:
        return 1.0, 0, 0, 0 # Jaccard, TP, FP, FN for blocks
    if not target_struct_dict: # Generated exists, target doesn't
        return 0.0, 0, len(generated_struct_dict), 0
    if not generated_struct_dict: # Target exists, generated doesn't
        return 0.0, 0, 0, len(target_struct_dict)

    best_jaccard = -1
    best_tp_blocks = 0
    best_fp_blocks = 0
    best_fn_blocks = 0
    
    # Normalize target structure once (to its own origin)
    normalized_target_dict = normalize_structure_coords(target_struct_dict)

    # Iterate through all 24 possible rotations for the generated structure
    for rot_fn in ROTATION_FUNCTIONS:
        # Apply rotation to the generated structure
        rotated_generated_dict = apply_rotation_to_structure(generated_struct_dict, rot_fn)
        # Normalize the rotated generated structure (to its own origin)
        normalized_rotated_generated_dict = normalize_structure_coords(rotated_generated_dict)

        # --- Calculate block-level TP, FP, FN for this rotation ---
        tp_blocks_current = 0
        
        # True Positives: Blocks present in BOTH, with same coordinates AND same color
        for coords, color in normalized_rotated_generated_dict.items():
            if coords in normalized_target_dict and normalized_target_dict[coords] == color:
                tp_blocks_current += 1
        
        # False Positives: Blocks in generated that are NOT in target (or wrong color)
        # These are blocks in generated that are either not in target at all OR
        # are at the same coordinate but with a different color.
        fp_blocks_current = len(normalized_rotated_generated_dict) - tp_blocks_current
        
        # False Negatives: Blocks in target that are NOT in generated (or wrong color)
        # These are blocks in target that are either not in generated at all OR
        # are at the same coordinate but with a different color.
        fn_blocks_current = len(normalized_target_dict) - tp_blocks_current

        # Calculate Jaccard for this rotation
        union_size = tp_blocks_current + fp_blocks_current + fn_blocks_current
        jaccard = tp_blocks_current / union_size if union_size > 0 else 0

        if jaccard > best_jaccard:
            best_jaccard = jaccard
            best_tp_blocks = tp_blocks_current
            best_fp_blocks = fp_blocks_current
            best_fn_blocks = fn_blocks_current
            
    # Return best Jaccard along with the corresponding TP, FP, FN block counts
    return best_jaccard, best_tp_blocks, best_fp_blocks, best_fn_blocks


# --- Wrapper to get overall similarity ---
def compute_structure_metrics(target_struct_dict, generated_struct_dict):
    """
    Computes a similarity score (Jaccard Index) and block-level TP/FP/FN counts
    between two structures, considering all their canonical orientations.
    """
    # This now just calls normalize_and_compare directly
    return normalize_and_compare(target_struct_dict, generated_struct_dict)


# --- Main Evaluation Function ---
def evaluate_structures_and_metrics(results_dir, targets_dir, experimental_condition_name, main_path):
    """
    Evaluate the built structures against the target structures for a specific experimental condition,
    calculate block-wise metrics, and store detailed results.

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
            "num_generated_structures": len(results_map),
            "num_target_structures": len(targets_map),
            "total_block_tp": 0, "total_block_fp": 0, "total_block_fn": 0,
            "overall_block_precision": 0.0, "overall_block_recall": 0.0, "overall_block_f1_score": 0.0,
            "details": {}
        }

    all_results_with_scores = {}
    
    # Initialize totals for aggregate block-level metrics
    total_block_tp_sum = 0
    total_block_fp_sum = 0
    total_block_fn_sum = 0

    # --- Step 1: Process matched pairs ---
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

        # Compute Jaccard and block-level counts (TP, FP, FN) for the optimal rotation
        jaccard_score, tp_blocks, fp_blocks, fn_blocks = \
            compute_structure_metrics(target_struct_dict, result_struct_dict)
        
        # Calculate block-level precision, recall, f1 for this specific structure
        # These are computed directly from block counts, no threshold involved
        block_precision = tp_blocks / (tp_blocks + fp_blocks) if (tp_blocks + fp_blocks) > 0 else 0
        block_recall = tp_blocks / (tp_blocks + fn_blocks) if (tp_blocks + fn_blocks) > 0 else 0
        block_f1_score = (2 * block_precision * block_recall) / (block_precision + block_recall) \
                         if (block_precision + block_recall) > 0 else 0
        
        # Accumulate total block counts for overall averages
        total_block_tp_sum += tp_blocks
        total_block_fp_sum += fp_blocks
        total_block_fn_sum += fn_blocks

        # Store detailed results for this structure
        all_results_with_scores[name] = {
            "jaccard_similarity": jaccard_score, # Jaccard is still useful to see
            "generated_structure_path": result_path,
            "target_structure_path": target_path,
            "block_level_tp": tp_blocks,
            "block_level_fp": fp_blocks,
            "block_level_fn": fn_blocks,
            "block_level_precision": block_precision,
            "block_level_recall": block_recall,
            "block_level_f1_score": block_f1_score,
            "generated_data_preview": result_data_raw[:5] if isinstance(result_data_raw, list) else "..."
        }
        print(f"File: {name}, Jaccard: {jaccard_score:.2f}, Block P/R/F1: {block_precision:.2f}/{block_recall:.2f}/{block_f1_score:.2f}")

    # --- Step 2: Account for False Negatives (targets that were not generated) ---
    unmatched_target_ids = set(targets_map.keys()) - set(matching_names)
    for name in sorted(unmatched_target_ids):
        try:
            with open(targets_map[name], 'r', encoding='utf-8') as tf:
                target_data_raw = json.load(tf)
            num_target_blocks_missing = len(parse_structure(target_data_raw))
        except json.JSONDecodeError:
            num_target_blocks_missing = 0 # Cannot parse target, assume 0 for FN count
        
        total_block_fn_sum += num_target_blocks_missing # Add all blocks from missing targets to FN

        all_results_with_scores[name] = {
            "jaccard_similarity": 0.0, # Cannot compute Jaccard for missing generated structure
            "status": "NOT_GENERATED", # Keep this status for clarity
            "target_structure_path": targets_map[name],
            "generated_structure_path": None,
            "block_level_tp": 0,
            "block_level_fp": 0,
            "block_level_fn": num_target_blocks_missing,
            "block_level_precision": 0.0, # No generated blocks, so precision is 0
            "block_level_recall": 0.0,    # All target blocks are missing, so recall is 0
            "block_level_f1_score": 0.0,
            "generated_data_preview": None
        }
        print(f"File: {name}, Status: NOT_GENERATED")

    # --- Step 3: Account for additional False Positives (generated structures without a target) ---
    unmatched_generated_ids = set(results_map.keys()) - set(targets_map.keys())
    for name in sorted(unmatched_generated_ids):
        try:
            with open(results_map[name], 'r', encoding='utf-8') as f:
                generated_preview_data = json.load(f)
                num_generated_blocks_unexpected = len(parse_structure(generated_preview_data))
                generated_preview_data = generated_preview_data[:5] if isinstance(generated_preview_data, list) else "..."
        except json.JSONDecodeError:
            num_generated_blocks_unexpected = 0
            generated_preview_data = "Error loading JSON for preview"

        total_block_fp_sum += num_generated_blocks_unexpected # Add all blocks from unexpected generated to FP

        all_results_with_scores[name] = {
            "jaccard_similarity": 0.0, # Cannot compute Jaccard without target
            "status": "UNEXPECTED_GENERATED", # Keep this status for clarity
            "generated_structure_path": results_map[name],
            "target_structure_path": None,
            "block_level_tp": 0,
            "block_level_fp": num_generated_blocks_unexpected,
            "block_level_fn": 0,
            "block_level_precision": 0.0, # No correct blocks (no target), so precision is 0
            "block_level_recall": 0.0,    # No target blocks to miss, so recall is N/A (set to 0 for consistency)
            "block_level_f1_score": 0.0,
            "generated_data_preview": generated_preview_data
        }
        print(f"File: {name}, Status: UNEXPECTED_GENERATED")


    # Calculate overall aggregate block-level metrics for this experimental condition
    overall_block_precision = total_block_tp_sum / (total_block_tp_sum + total_block_fp_sum) if (total_block_tp_sum + total_block_fp_sum) > 0 else 0
    overall_block_recall = total_block_tp_sum / (total_block_tp_sum + total_block_fn_sum) if (total_block_tp_sum + total_block_fn_sum) > 0 else 0
    overall_block_f1_score = 2 * (overall_block_precision * overall_block_recall) / (overall_block_precision + overall_block_recall) if (overall_block_precision + overall_block_recall) > 0 else 0

    metrics = {
        "num_generated_structures": len(results_map),
        "num_target_structures": len(targets_map),
        "total_block_tp": total_block_tp_sum,
        "total_block_fp": total_block_fp_sum,
        "total_block_fn": total_block_fn_sum,
        "overall_block_precision": overall_block_precision,
        "overall_block_recall": overall_block_recall,
        "overall_block_f1_score": overall_block_f1_score
    }
    
    print(f"\nOverall Block-level Metrics for {experimental_condition_name}:")
    print(f" Total TP Blocks: {total_block_tp_sum}, FP Blocks: {total_block_fp_sum}, FN Blocks: {total_block_fn_sum}")
    print(f" Overall Block Precision: {overall_block_precision:.2f}")
    print(f" Overall Block Recall: {overall_block_recall:.2f}")
    print(f" Overall Block F1-Score: {overall_block_f1_score:.2f}")

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
    # This section now aggregates the *block-level* metrics from each condition
    for condition, metrics in all_conditions_metrics.items():
        print(f"\nCondition: {condition}")
        print(f" Number of Generated Structures: {metrics['num_generated_structures']}")
        print(f" Number of Target Structures: {metrics['num_target_structures']}")
        print(f" Total TP Blocks: {metrics['total_block_tp']}, FP Blocks: {metrics['total_block_fp']}, FN Blocks: {metrics['total_block_fn']}")
        print(f" Overall Block Precision: {metrics['overall_block_precision']:.2f}")
        print(f" Overall Block Recall: {metrics['overall_block_recall']:.2f}")
        print(f" Overall Block F1-Score: {metrics['overall_block_f1_score']:.2f}")

    # Optionally, save aggregated metrics
    aggregated_metrics_path = os.path.join(analysis_output_dir, "aggregated_metrics_new.json")
    with open(aggregated_metrics_path, 'w', encoding='utf-8') as outfile:
        json.dump(all_conditions_metrics, outfile, indent=2)
    print(f"\nAggregated metrics saved to {aggregated_metrics_path}")