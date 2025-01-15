import os
import xml.etree.ElementTree as ET
import pandas as pd
import json

def get_main_data_path():
    current_path = os.path.dirname(__file__)
    parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
    main_path = os.path.abspath(os.path.join(parent_path, os.pardir))
    data_path = os.path.join(main_path, "data")
    return data_path

def create_terrain():
    """
    Generates a terrain XML file with ground blocks in the specified region.
    """
    block_type = "cwcmod:cwc_gray_rn"
    blocks = [create_draw_block(x, 0, z, block_type) for x in range(95, 106) for z in range(94, 105)]
    save_xml("terrain.xml", blocks)
    print("XML file 'terrain.xml' created successfully.")


def create_draw_block(x, y, z, block_type):
    """
    Creates a DrawBlock XML element.
    """
    draw_block = ET.Element("DrawBlock", {"type": block_type, "x": str(x), "y": str(y), "z": str(z)})
    return ET.tostring(draw_block, encoding="utf-8").decode("utf-8")


def save_xml(filename, content):
    """
    Saves XML content to a file.
    """
    with open(filename, "w") as file:
        file.write("\n".join(content))


def load_and_create_dataframe(structure_ID, structure_name, render_terrain, processed_folder_path, save=False):
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
    if save:
        save_dataframe(structure_df, structure_ID, structure_name, processed_folder_path)

    return structure_df


def ensure_terrain_file():
    """
    Ensures the terrain file exists, creating it if necessary.
    """
    terrain_path = os.path.join(os.path.dirname(__file__), "terrain.xml")
    if not os.path.exists(terrain_path):
        print("Creating 'terrain.xml'...")
        create_terrain()
    return terrain_path


def get_structure_path(structure_ID):
    """
    Returns the path to the target structure XML file.
    """
    return os.path.join(os.path.dirname(__file__), "gold-configurations", f"{structure_ID}.xml")


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

main_data_path = get_main_data_path()
