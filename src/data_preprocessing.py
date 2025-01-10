import pathlib, os

# Retrieve the relative data path
current_script_path = pathlib.Path(__file__).parent.resolve()
relative_data_path = current_script_path.parent.resolve().joinpath("data_minecraft_corpus")

# List the days of data
data_days = [f for f in os.listdir(relative_data_path) if os.path.isdir(os.path.join(relative_data_path, f))]

