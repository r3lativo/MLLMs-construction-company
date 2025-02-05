import argparse
import os
from natsort import natsorted
from renderer_utils import *


def get_args():
    parser = argparse.ArgumentParser(description="Render a 3D structure from XML files")
    parser.add_argument("-m", "--mode", default="dynamic", choices=["dynamic", "circular", "create_data"],
                        help="Dynamic interactive view / Circular view and screenshots / Process data inside 'gold-configurations'")
    parser.add_argument("-i", "--ID_structure", default="C1",
                        help="Structure ID to render")
    parser.add_argument("-t", "--render_terrain", default="render", choices=["render", "skip"],
                        help="Whether to render terrain or skip it (structure only)")
    parser.add_argument("-s", "--save_JSON", default="temp", choices=["save", "temp"],
                        help="Save JSON to file")
    args = parser.parse_args()
    return args


def handle_structure_data(structure_ID, structure_name, mode, gold_processed_path):
    """
    Orchestrates data loading, folder creation, and visualization for a specific structure.

    Args:
        structure_ID (str): Unique ID of the structure.
        structure_name (str): Name of the structure.
        mode (str): Rendering mode (e.g., 'dynamic', 'circular').
        gold_processed_path (str): Path to save processed data and visualizations.
    """
    # Define the path for processed data for this structure
    ID_processed_path = os.path.join(gold_processed_path, f"{structure_ID}_{structure_name}")
    os.makedirs(ID_processed_path, exist_ok=True)  # Create the directory if it doesn't exist

    # Load data and save JSON for the structure without rendering terrain
    structure_df = load_and_create_dataframe(
        structure_ID, structure_name, render_terrain="skip", ID_processed_path=ID_processed_path
    )
    save_dataframe(structure_df, structure_ID, structure_name, ID_processed_path)

    # Load data and render the structure *with terrain*
    structure_df = load_and_create_dataframe(
        structure_ID, structure_name, render_terrain="render", ID_processed_path=ID_processed_path
    )
    plotter = initialize_plotter("circular")
    render_structure(structure_ID,
                    structure_name,
                    plotter,
                    mode,
                    structure_df,
                    ID_processed_path)
    plotter.close()  # Close the plotter to free resources
    plotter.deep_clean()  # Perform deep cleanup of plotter resources
    

def process_all_structures(config_dict, mode, gold_configurations_path, gold_processed_path):
    """
    Handles all structures in the configuration folder.

    Args:
        config_dict (dict): Dictionary mapping structure IDs to their names.
        mode (str): Rendering mode (e.g., 'dynamic', 'circular').
        gold_configurations_path (str): Path to folder containing configuration files.
        gold_processed_path (str): Path to save processed data and visualizations.
    """
    # List all XML configuration files and sort them naturally (human-friendly order)
    files = list(filter(lambda f: f.endswith(".xml"), os.listdir(gold_configurations_path)))
    files = natsorted(files)

    # Loop through each file and process its structure
    for filename in (pbar := tqdm(files, desc="Processing structures", total=len(files))):
        structure_ID = os.path.splitext(filename)[0]  # Extract structure ID from filename
        structure_name = get_structure_name(structure_ID, config_dict)  # Get structure name from config
        pbar.set_postfix_str(f"{structure_ID}_{structure_name}")  # Update progress bar
        handle_structure_data(structure_ID, structure_name, mode, gold_processed_path)  # Process structure

    print(f"Done. Number of structures processed: {len(files)} ")
    print(f"JSONs and screenshots saved at '{gold_processed_path}'.")


def main():
    """
    Entry point for rendering and processing 3D structures based on arguments provided by the user.
    """
    args = get_args()

    # Paths
    gold_configurations_path = os.path.join(data_structures_path, "gold-configurations")
    gold_processed_path = os.path.join(data_structures_path, "gold-processed")
    config_path = os.path.join(data_structures_path, "configs-to-names.txt")
    config_dict = load_configurations(config_path)

    if args.mode == "create_data":
        # Process all structures in the gold-configurations folder
        process_all_structures(config_dict, args.mode, gold_configurations_path, gold_processed_path)
    else:
        # Process a single structure specified by the user
        structure_name = get_structure_name(args.ID_structure, config_dict)
        plotter = initialize_plotter(args.mode)
        ID_processed_path = os.path.join(gold_processed_path, f"{args.ID_structure}_{structure_name}")
        os.makedirs(ID_processed_path, exist_ok=True)

        # Load and process the specified structure
        structure_df = load_and_create_dataframe(
            args.ID_structure, structure_name, args.render_terrain, ID_processed_path, save=args.save_JSON
        )
        render_structure(
            args.ID_structure, structure_name, plotter, args.mode, structure_df, ID_processed_path
            )


if __name__ == "__main__":
    main()
