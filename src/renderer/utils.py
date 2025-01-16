import os
import xml.etree.ElementTree as ET
import pandas as pd
import pyvista as pv
from tqdm import tqdm


def render_structure(structure_ID, structure_name, plotter, mode, structure_df, ID_processed_path):
    """
    Visualizes the 3D structure using PyVista and captures screenshots or displays interactively.
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
        take_screenshots(plotter, structure_ID, structure_name, ID_processed_path)


def plot_structure(block_row, plotter):
    """
    Plots a 3D block for a single row in the DataFrame.
    """
    block_color, x, y, z = block_row
    cube = pv.Cube(center=(x, z, y), x_length=1, y_length=1, z_length=1)
    plotter.add_mesh(cube, color=block_color, show_edges=True)


def initialize_plotter(mode):
    """
    Initializes the PyVista plotter based on mode.
    """
    if mode == "dynamic":
        off_screen = False
    else: off_screen = True
    return pv.Plotter(off_screen=off_screen)


def take_screenshots(plotter, structure_ID, structure_name, ID_processed_path):
    """
    Takes four perspective screenshots and saves them.
    """
    angles = {'0_front': 90, '1_right': 180, '2_back': 270, '3_left': 360}
    plotter.view_xz()
    plotter.camera.elevation = 30

    for label, angle in angles.items():
        screenshot_path = os.path.join(ID_processed_path, f"screenshot_{structure_ID}_{structure_name}_{label}.jpg")
        plotter.screenshot(screenshot_path)
        plotter.camera.azimuth = angle
        plotter.render()  # Render the structure again
        #print(f"'{screenshot_path}' created")


def get_structure_name(structure_ID, config_dict):
    """
    Fetches the structure name, handling missing IDs.
    """
    return config_dict.get(structure_ID, "NO_ID")


def get_data_structures_path():
    """
    Constructs and returns the path to the 'structures' folder in 'data'.
    """
    current_path = os.path.dirname(__file__)
    parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
    main_path = os.path.abspath(os.path.join(parent_path, os.pardir))
    data_structures_path = os.path.join(main_path, "data", "structures")
    return data_structures_path


def create_terrain():
    """
    Generates and saves a terrain XML file with predefined blocks.
    """
    block_type = "cwcmod:cwc_gray_rn"
    blocks = [create_draw_block(x, 0, z, block_type) for x in range(95, 106) for z in range(94, 105)]
    save_xml(os.path.join(data_structures_path, "terrain.xml"), blocks)
    print(f"XML file 'terrain.xml' successfully created at `{data_structures_path}`.")


def create_draw_block(x, y, z, block_type):
    """
    Creates a DrawBlock XML element for the given coordinates and block type.
    """
    draw_block = ET.Element("DrawBlock", {"type": block_type, "x": str(x), "y": str(y), "z": str(z)})
    return ET.tostring(draw_block, encoding="utf-8").decode("utf-8")


def save_xml(filepath, content):
    """
    Saves XML content to a file.
    """
    with open(filepath, "w") as file:
        file.write("\n".join(content))


def load_and_create_dataframe(structure_ID, structure_name, render_terrain, ID_processed_path, save="save"):
    """
    Combines terrain and structure XML data, parses it into a DataFrame, and optionally saves it.
    """
    terrain_path = ensure_terrain_file()
    structure_path = get_structure_path(structure_ID)

    if not os.path.exists(structure_path):
        print(f"File '{structure_ID}.xml' not found in 'gold-configurations'.")
        return None

    terrain_content = load_file(terrain_path)
    structure_content = load_file(structure_path)

    combined_content = combine_xml(terrain_content, structure_content, render_terrain)
    structure_df = parse_xml_to_dataframe(combined_content)
    if save == "save":
        save_dataframe(structure_df, structure_ID, structure_name, ID_processed_path)

    return structure_df


def ensure_terrain_file():
    """
    Ensures the terrain XML file exists, creating it if necessary.
    """
    terrain_path = os.path.join(data_structures_path, "terrain.xml")
    if not os.path.exists(terrain_path):
        print("Creating 'terrain.xml'...")
        create_terrain()
    return terrain_path


def get_structure_path(structure_ID):
    """
    Constructs the path to the structure XML file based on its ID.
    """
    return os.path.join(data_structures_path, "gold-configurations", f"{structure_ID}.xml")


def load_file(file_path):
    """
    Reads and returns the content of a file.
    """
    with open(file_path, "r") as file:
        return file.read().lstrip()


def combine_xml(terrain_content, structure_content, render_terrain):
    """
    Combines terrain and structure XML content based on rendering preferences.
    """
    if render_terrain == "render":
        return f"<root>{terrain_content}{structure_content}</root>"
    return f"<root>{structure_content}</root>"


def parse_xml_to_dataframe(xml_content):
    """
    Parses XML content into a pandas DataFrame with block attributes.
    """
    root = ET.fromstring(xml_content)
    data = [
        (
            block.get("type").split('_rn')[0].split('_')[-1],  # Block color
            int(block.get("x")),
            int(block.get("y")),
            int(block.get("z"))
        )
        for block in root.findall("DrawBlock")
    ]
    return pd.DataFrame(data, columns=["block_color", "x", "y", "z"])


def load_configurations(config_file):
    """
    Loads configurations from a file into a dictionary.
    """
    with open(config_file, "r") as file:
        return dict(line.strip().split("\t") for line in file if "\t" in line)


def save_dataframe(structure_df, structure_ID, structure_name, processed_path):
    """
    Saves a DataFrame as a JSON file in the processed directory.
    """
    filename = os.path.join(processed_path, f"{structure_ID}_{structure_name}.json")
    structure_df.to_json(filename, orient="records", indent=4)
    #print(f"File '{filename}' created successfully.")


data_structures_path = get_data_structures_path()
