import argparse
import os
from natsort import natsorted
from utils import *


def handle_structure_data(structure_ID, structure_name, mode, gold_processed_path):
    """
    Orchestrates data loading, folder creation, and visualization for a structure.
    """
    ID_processed_path = os.path.join(gold_processed_path, f"{structure_ID}_{structure_name}")
    os.makedirs(ID_processed_path, exist_ok=True)

    # Plot structure witohut terrain to save JSON
    structure_df = load_and_create_dataframe(structure_ID, structure_name, render_terrain="skip", ID_processed_path=ID_processed_path)
    save_dataframe(structure_df, structure_ID, structure_name, ID_processed_path)

    # Plot structure with terrain
    structure_df = load_and_create_dataframe(structure_ID, structure_name, render_terrain="render", ID_processed_path=ID_processed_path)
    plotter = initialize_plotter("circular")
    render_structure(structure_ID,
                    structure_name,
                    plotter,
                    mode,
                    structure_df,
                    ID_processed_path)
    plotter.close()
    plotter.deep_clean()
    

def process_all_structures(config_dict, mode, gold_configurations_path, gold_processed_path):
    """
    Handles all structures in the configuration folder.
    """
    """
    for filename in filter(lambda f: f.endswith(".xml"), os.listdir("gold-configurations")):
        structure_ID = os.path.splitext(filename)[0]
        structure_name = get_structure_name(structure_ID, config_dict)
        handle_structure_data(structure_ID, structure_name, "render", mode)
    """
    # Get the list of XML files and calculate the total number of files
    files = list(filter(lambda f: f.endswith(".xml"), os.listdir(gold_configurations_path)))
    files = natsorted(files)

    for filename in (pbar := tqdm(files, desc="Processing structures", total=len(files))):
        structure_ID = os.path.splitext(filename)[0]
        structure_name = get_structure_name(structure_ID, config_dict)
        pbar.set_postfix_str(f"{structure_ID}_{structure_name}")
        handle_structure_data(structure_ID, structure_name, mode, gold_processed_path)
    print(f"Done. Number of structures processed: {len(files)}")
    print(f"JSONs and screenshots saved at '{gold_processed_path}'.")


def main():
    """
    Entry point for rendering and processing 3D structures.
    """
    parser = argparse.ArgumentParser(description="Render a 3D structure from XML files")
    parser.add_argument("-m", "--mode", default="dynamic", choices=["dynamic", "circular", "create_data"],
                        help="Dynamic interactive view / Circular view and screenshots / Process data inside 'gold-configurations'")
    parser.add_argument("-i", "--ID_structure", default="C1",
                        help="Structure ID to render")
    parser.add_argument("-t", "--render_terrain", default="render", choices=["render", "skip"],
                        help="Whether to render terrain or skip it (structure only)")
    parser.add_argument("-s", "--save_JSON", default="save", choices=["save", "temp"],
                        help="Save JSON to file")
    args = parser.parse_args()

    # Paths
    gold_configurations_path = os.path.join(data_structures_path, "gold-configurations")
    gold_processed_path = os.path.join(data_structures_path, "gold-processed")
    config_path = os.path.join(data_structures_path, "configs-to-names.txt")
    config_dict = load_configurations(config_path)

    if args.mode == "create_data":
        process_all_structures(config_dict, args.mode, gold_configurations_path, gold_processed_path)
    else:
        structure_name = get_structure_name(args.ID_structure, config_dict)
        plotter = initialize_plotter(args.mode)
        ID_processed_path = os.path.join(gold_processed_path, f"{args.ID_structure}_{structure_name}")
        os.makedirs(ID_processed_path, exist_ok=True)
        structure_df = load_and_create_dataframe(
            args.ID_structure, structure_name, args.render_terrain, ID_processed_path, save=args.save_JSON
        )
        render_structure(args.ID_structure, structure_name, plotter, args.mode, structure_df, ID_processed_path)


if __name__ == "__main__":
    main()
