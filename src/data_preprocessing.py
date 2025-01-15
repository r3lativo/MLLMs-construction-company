import pathlib, os, sys, json
from worldstate_decompile import *  # Importing the WorldStateData class and related data structures
#
# "Folder", "directory" and "path" are used interchangeably
#


def list_folders(main_path):
    """
    Simply return a list of the folders inside a folder.

    Args:
        main_path (str or Path): The main directory to scan.

    Returns:
        List[str]: A list of folder names in the specified directory.
    """
    return [f for f in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, f))]


def retrieve_relative_data_path():
    """
    Retrieve the relative data path.
    This ensures that the code works properly no matter from where one runs it.

    Returns:
        Path: The resolved path to the 'data_minecraft_corpus' directory.
    """
    # The path of this script
    current_script_path = pathlib.Path(__file__).parent.resolve()

    # The path to the 'data_minecraft_corpus' directory
    relative_data_path = current_script_path.parent.resolve().joinpath("data_minecraft_corpus")
    return relative_data_path


def same_log_and_screenshots(day_path):
    """
    Check that logs and screenshots have the same sessions.

    Args:
        day_path (Path): The path to a day's data folder.

    Returns:
        List[str]: A list of session names that are common between logs and screenshots.

    Raises:
        SystemExit: If the logs and screenshots do not have matching session names.
    """
    logs_path = day_path.joinpath("logs")  # Path to the logs folder
    screenshots_path = day_path.joinpath("screenshots")  # Path to the screenshots folder

    # Get the session names for logs and screenshots
    logs_sessions = list_folders(logs_path)
    screenshots_sessions = list_folders(screenshots_path)

    # Compare the sessions
    if logs_sessions == screenshots_sessions:
        # Sessions match; return the list of session names
        return logs_sessions
    else:
        # Sessions do not match; raise an error and exit
        return sys.exit("logs and screenshots DO NOT have the same sessions: something went wrong")


def process_session_data(session_id, logs_path, verbose=False):
    """
    Process the JSON data for a specific session.

    Args:
        session_id (str): The session ID.
        logs_path (Path): The path to the logs folder.
        verbose (bool): If True, print detailed information about the session data.
    """
    # Path to the aligned observations JSON file for the session
    json_file_path = logs_path.joinpath(session_id, 'aligned-observations.json')
    
    # Skip processing if the JSON file does not exist
    if not json_file_path.exists():
        print(f"Skipping session {session_id} because JSON file does not exist.")
        return

    # Open and parse the JSON file
    with open(json_file_path, 'r') as f:
        parsed_data = json.load(f)

    # Deserialize the parsed JSON into WorldStateData objects
    world_state_data = WorldStateData.from_dict(parsed_data)
    
    # If verbose is enabled, print detailed session information
    if verbose:
        print(f"Processing session {session_id}...\n")
        for world_state in world_state_data.WorldStates:
            print(f"Timestamp: {world_state.Timestamp}")
            
            print("Chat History:")
            for message in world_state.ChatHistory:
                print(f"  - {message}")
            
            print("Builder Inventory:")
            for item in world_state.BuilderInventory:
                print(f"  - {item.Type} (Quantity: {item.Quantity})")
            
            print("Screenshots:")
            for key, screenshot in world_state.Screenshots.__dict__.items():
                print(f"  - {key}: {screenshot if screenshot else 'No screenshot'}")
            
            print("\n" + "-" * 50 + "\n")


def main():
    """
    Main function to process the Minecraft corpus data.
    """
    # Get the relative path to the data folder
    relative_data_path = retrieve_relative_data_path()

    # List all the folders representing days of data
    data_days = list_folders(relative_data_path)

    # Process each day of data
    for day in data_days:
        day_path = relative_data_path.joinpath(day)  # Path to the day's folder
        
        # Check if logs and screenshots sessions match
        session_names = same_log_and_screenshots(day_path)

        # Process data for each session
        for session_id in session_names:
            logs_path = day_path.joinpath("logs")  # Path to the logs folder
            process_session_data(session_id, logs_path, verbose=False)  # Set verbose=True for detailed output


# Entry point of the script
main()
