import pyvista as pv
from tqdm import tqdm
import argparse
import os, sys
from natsort import natsorted
from utils import (
    load_and_create_dataframe,
    load_configurations,
    save_dataframe,
    main_data_path
)


def plot_structure(block_row, plotter):
    """
    Plots a 3D block for a single row in the DataFrame.
    """
    block_color, x, y, z = block_row
    cube = pv.Cube(center=(x, z, y), x_length=1, y_length=1, z_length=1)
    plotter.add_mesh(cube, color=block_color, show_edges=True)


def take_screenshots(plotter, structure_ID, structure_name, processed_path):
    """
    Takes four perspective screenshots and saves them.
    """
    angles = {'0_front': 90, '1_right': 180, '2_back': 270, '3_left': 360}
    plotter.view_xz()
    plotter.camera.elevation = 30

    for label, angle in angles.items():
        screenshot_path = os.path.join(processed_path, f"screenshot_{structure_ID}_{structure_name}_{label}.jpg")
        plotter.screenshot(screenshot_path)
        plotter.camera.azimuth = angle
        plotter.render()  # Render the structure again
        #print(f"'{screenshot_path}' created")

def initialize_plotter(mode):
    """
    Initializes the PyVista plotter based on mode and display preferences.
    """
    if mode == "dynamic":
        off_screen = False
    else: off_screen = True
    return pv.Plotter(off_screen=off_screen)


def get_structure_name(structure_ID, config_dict):
    """
    Fetches the structure name, handling missing IDs gracefully.
    """
    return config_dict.get(structure_ID, "NO_ID")


def process_structure(structure_ID, structure_name, plotter, mode, structure_df, processed_path):
    """
    Visualizes and captures screenshots of the structure.
    """

    if mode != "create_data":
        # Only use tqdm if the mode is not 'create_data'
        tqdm.pandas(desc=f"Plotting structure {structure_ID}_{structure_name}")
        structure_df.progress_apply(lambda row: plot_structure(row, plotter), axis=1)
    else:
        structure_df.apply(lambda row: plot_structure(row, plotter), axis=1)

    plotter.window_size = [640, 640]
    if mode == "dynamic":
        plotter.show()
    else:
        take_screenshots(plotter, structure_ID, structure_name, processed_path)


def handle_structure_data(structure_ID, structure_name, render_terrain, mode):
    """
    Orchestrates data loading, folder creation, and visualization for a structure.
    """
    processed_path = os.path.join("gold-processed", f"{structure_ID}_{structure_name}")
    os.makedirs(processed_path, exist_ok=True)

    # Plot structure witohut terrain to save JSON
    structure_df = load_and_create_dataframe(structure_ID, structure_name, render_terrain="skip", processed_folder_path=processed_path)
    save_dataframe(structure_df, structure_ID, structure_name, processed_path)

    # Plot structure with terrain
    structure_df = load_and_create_dataframe(structure_ID, structure_name, render_terrain="render", processed_folder_path=processed_path)
    plotter = initialize_plotter("circular")
    process_structure(structure_ID,
                      structure_name,
                      plotter,
                      mode,
                      structure_df,
                      processed_path)
    plotter.close()
    plotter.deep_clean()
    

def process_all_structures(config_dict, mode):
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
    files = list(filter(lambda f: f.endswith(".xml"), os.listdir("gold-configurations")))
    files = natsorted(files)

    for filename in (pbar := tqdm(files, desc="Processing structures", total=len(files))):
        structure_ID = os.path.splitext(filename)[0]
        structure_name = get_structure_name(structure_ID, config_dict)
        pbar.set_postfix_str(structure_ID)
        handle_structure_data(structure_ID, structure_name, "render", mode)


def main():
    """
    Entry point for rendering and processing 3D structures.
    """
    parser = argparse.ArgumentParser(description="Render a 3D structure from XML files.")
    parser.add_argument("-m", "--mode", default="dynamic", choices=["dynamic", "circular", "create_data"])
    parser.add_argument("-s", "--structure", default="C1", help="Structure ID to render.")
    parser.add_argument("-t", "--render_terrain", default="render", choices=["render", "skip"])
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), "configs-to-names.txt")
    config_dict = load_configurations(config_path)

    if args.mode == "create_data":
        process_all_structures(config_dict, args.mode)
    else:
        structure_name = get_structure_name(args.structure, config_dict)
        plotter = initialize_plotter(args.mode)
        structure_df = load_and_create_dataframe(
            args.structure, structure_name, args.render_terrain, None
        )
        process_structure(args.structure, structure_name, plotter, args.mode, structure_df, "gold-processed")


if __name__ == "__main__":
    main()
