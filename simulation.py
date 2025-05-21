import json
import logging
from logging.handlers import RotatingFileHandler
import os
import copy

from architect import Architect
from builder import Builder
from action_handler import World
from config import config
from jinja2 import Environment, FileSystemLoader
import glob
from natsort import natsorted
import pandas as pd
import pyvista as pv

# --- Constants and Global Setup ---

MAX_TURNS = 20
RENDER_INTERVAL = 1

# Configure Jinja2 environment to load templates from the 'prompts' directory
JINJA_ENV = Environment(loader=FileSystemLoader('prompts'))

# Define base directories for structure data and simulation results
BASE_STRUCTURES_DIR = "data/structures/gold-processed"
RESULTS_ROOT_DIR = "new_results"

# Load terrain data once, as it's static
TERRAIN_DF = pd.read_json("data/structures/terrain.json")

# Model names, typically loaded from a config file
ARCHITECT_MODEL_NAME = config.get("architect_model_id")
BUILDER_MODEL_NAME = config.get("builder_model_id")
VLLM_API_BASE_URL = "http://localhost:8000/v1" # Example API base URL, if needed

# --- Helper Functions ---

def _setup_logging() -> logging.Logger:
    """
    Configures and returns a root logger that writes to a rotating file.

    The log file 'mllms.log' will rotate after 10 MB, keeping up to 5 backup files.
    """
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    log_handler = RotatingFileHandler(
        "mllms.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    log_handler.setFormatter(log_formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.ERROR)
    root_logger.addHandler(log_handler)
    return root_logger

def _save_agent_history(agent_name: str, history: list, file_path: str, logger: logging.Logger):
    """
    Saves an agent's complete conversation history to a specified JSON file.

    Args:
        agent_name (str): The name of the agent (e.g., "Architect", "Builder").
        history (list): The conversation history to save.
        file_path (str): The full path to the output JSON file.
        logger (logging.Logger): The logger instance.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {agent_name}'s history to {os.path.basename(file_path)}")
    except IOError as e:
        logger.error(f"Error saving {agent_name}'s history to {file_path}: {e}")

def _save_world_state(world_data: list[dict], turn: int, logger: logging.Logger, current_results_output_path: str):
    """
    Saves the current world state to a JSON file, named by the simulation turn.

    Args:
        world_data (list[dict]): The current state of the world as a list of dictionaries.
        turn (int): The current simulation turn number.
        logger (logging.Logger): The logger instance.
        current_results_output_path (str): The directory where the world state should be saved.
    """
    file_path = os.path.join(current_results_output_path, f"world_state_{turn:03d}.json")
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(world_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved world state for turn {turn} to {os.path.basename(file_path)}")
    except IOError as e:
        logger.error(f"Error saving world state for turn {turn} to {file_path}: {e}")

def _initialize_simulation_components(architect_model: str, builder_model: str, structure_info: dict, logger: logging.Logger):
    """
    Initializes the World, Architect, and Builder agents for the simulation.

    Args:
        architect_model (str): The model identifier for the Architect agent.
        builder_model (str): The model identifier for the Builder agent.
        structure_info (dict): A dictionary containing structure description JSON and image paths.
        logger (logging.Logger): The logger instance.

    Returns:
        Tuple[World, Architect, Builder]: The initialized World, Architect, and Builder objects.
    """
    world = World()
    architect = Architect(
        model=architect_model,
        system_prompt=JINJA_ENV.get_template('a_sys_prompt.jinja'),
        init_prompt=JINJA_ENV.get_template('a_init_prompt.jinja'),
        structure_description_json=structure_info.get('json_data'),
        structure_image_paths=structure_info.get('image_paths')
    )
    builder = Builder(
        model=builder_model,
        system_prompt=JINJA_ENV.get_template('b_sys_prompt.jinja'),
        action_handler=world
    )
    logger.info("Simulation components initialized.")
    return world, architect, builder

def _process_builder_turn(builder: Builder, architect_message: str, logger: logging.Logger):
    """
    Handles the Builder's turn: processes the Architect's instruction and extracts actions/communication.

    Args:
        builder (Builder): The Builder agent instance.
        architect_message (str): The message received from the Architect.
        logger (logging.Logger): The logger instance.

    Returns:
        Tuple[str, any]: A tuple containing the Builder's communication message and extracted actions.
    """
    logger.info("Builder: Processing Architect's instruction...")
    raw_builder_output, builder_communication, builder_actions = \
        builder.process_architect_instruction(architect_message)

    total_actions = sum(len(v) for v in builder_actions.actions.values()) if builder_actions else 0
    logger.info(f"Builder: Generated communication ({len(builder_communication)} chars) and {total_actions} actions.")
    return builder_communication, builder_actions

def _orchestrate_world_update(world: World, builder_actions: any, turn: int, logger: logging.Logger, builder: Builder, current_results_output_path: str):
    """
    Executes the Builder's actions and saves the resulting world state.

    Args:
        world (World): The current World object.
        builder_actions (any): The actions generated by the Builder.
        turn (int): The current simulation turn number.
        logger (logging.Logger): The logger instance.
        builder (Builder): The Builder agent instance (used for action execution).
        current_results_output_path (str): The directory to save world state.
    """
    if builder_actions:
        logger.info(f"Orchestrator: Executing Builder's actions for turn {turn}...")
        builder.execute_actions(builder_actions)
    else:
        logger.info(f"Orchestrator: No actions to execute for turn {turn}.")

    # Save the world state after actions for this turn
    current_world_state_data = world.get_state_as_json()
    _save_world_state(current_world_state_data, turn, logger, current_results_output_path)

def plot_structure(block_row: pd.Series, plotter: pv.Plotter):
    """
    Plots a 3D block for a single row of block data in a PyVista plotter.

    Args:
        block_row (pd.Series): A pandas Series containing 'block_color', 'x', 'y', 'z' for a block.
        plotter (pv.Plotter): The PyVista plotter instance to add the mesh to.
    """
    block_color, x, y, z = block_row
    cube = pv.Cube(center=(x, z, y), x_length=1, y_length=1, z_length=1)
    plotter.add_mesh(cube, color=block_color, show_edges=True)

def take_screenshots(plotter: pv.Plotter, turn: int, output_path: str) -> list[str]:
    """
    Generates and saves multiple screenshots of the current world state from different angles.

    Args:
        plotter (pv.Plotter): The PyVista plotter with the rendered structure.
        turn (int): The current simulation turn number, used for naming files.
        output_path (str): The directory to save the screenshots.

    Returns:
        List[str]: A list of file paths to the generated screenshots.
    """
    plotter.view_xz() 
    plotter.set_background("pv") # Sets background to PyVista's default gradient
    created_screenshot_paths = []
    # Define angles/views for screenshots
    angles = {'front': 0, 'three_q': 45}
    # ABOVE view
    plotter.camera.elevation = 90
    plotter.camera.azimuth = 0
    plotter.render()
    above_path = os.path.join(output_path, f"world_state_{turn:03d}_above.jpg")
    plotter.screenshot(above_path)
    created_screenshot_paths.append(above_path)
    # AROUND views
    plotter.camera.elevation = 30 # Set common elevation for around views
    for label, angle in angles.items():
        plotter.camera.azimuth = angle
        plotter.render()
        screenshot_path = os.path.join(output_path, f"world_state_{turn:03d}_{label}.jpg")
        plotter.screenshot(screenshot_path)
        created_screenshot_paths.append(screenshot_path)
    
    return created_screenshot_paths

def generate_world_feedback(world: World, turn: int, logger: logging.Logger, output_path: str, final: bool = False, run_type: str = "img_json") -> tuple[str, list[str]]:
    """
    Generates text and/or image feedback for the architect based on the current world state.

    Args:
        world (World): The current world object.
        turn (int): The current simulation turn number.
        logger (logging.Logger): The logger instance.
        output_path (str): The base directory to save output files (e.g., results/structure_name/).
        final (bool): If True, generates and saves final JSON and a specific JPEG, ignoring run_type.
                      Defaults to False.
        run_type (str): Specifies the type of feedback for non-final turns
                        ("json_only", "img_only", "img_json"). Defaults to "img_json".

    Returns:
        Tuple[str, List[str]]: A tuple containing the text description and a list of paths
                                to the generated image files for Architect's feedback.
    """
    # Initialize plotter and plot the structure (needed for both intermediate and final renders)
    plotter = pv.Plotter(off_screen=True, window_size=[640, 640])
    current_world_state_df = pd.DataFrame(world.get_state_as_json())
    
    # Combine terrain with current world state for plotting
    combined_df = pd.concat([TERRAIN_DF, current_world_state_df], ignore_index=True)
    combined_df.apply(lambda row: plot_structure(row, plotter), axis=1)

    world_state_for_architect = ""
    generated_image_paths = []

    if not final:
        # Intermediate feedback for Architect based on run_type
        if run_type in ["json_only", "img_json"]:
            world_state_for_architect = world.get_state_description_for_architect()
            logger.debug(f"Generated text feedback for architect (turn {turn}).") # Keep as debug, as it's just a generated string

        if run_type in ["img_only", "img_json"]:
            generated_image_paths = take_screenshots(plotter, turn, output_path)
            logger.info(f"Generated {len(generated_image_paths)} image(s) for architect feedback (turn {turn}).")
        
        plotter.close()
        return world_state_for_architect, generated_image_paths
    else:
        # Final output for the simulation, always includes JSON and a specific JPEG
        logger.info(f"Generating final world state outputs for turn {turn}.")

        # 1. Save final world state as JSON
        final_world_state = world.get_state_as_json()
        final_json_path = os.path.join(output_path, "final_world_state.json")
        with open(final_json_path, 'w') as f:
            json.dump(final_world_state, f, indent=2)

        # 2. Save final three_quarters JPEG
        plotter.view_xz() 
        plotter.set_background("pv")
        plotter.camera.elevation = 30
        plotter.camera.azimuth = 45
        plotter.render()
        final_jpeg_path = os.path.join(output_path, f"final_world_state.jpg")
        plotter.screenshot(final_jpeg_path)
        plotter.close()
        return "", [] # Return empty as architect doesn't need "feedback" after final save

## --- Main Simulation Function ---

def run_simulation(
    architect_model: str,
    builder_model: str,
    structure_info: dict,
    max_turns: int = 10,
    render_interval: int = 3
):
    """
    Runs the Architect-Builder collaborative simulation for a given structure.

    The simulation proceeds turn-by-turn, with the Architect instructing the Builder,
    the Builder executing actions, and the Orchestrator updating the world and
    providing feedback to the Architect.

    Args:
        architect_model (str): The identifier for the Architect's language model.
        builder_model (str): The identifier for the Builder's language model.
        structure_info (dict): Contains the structure's name, JSON data, and image paths.
        max_turns (int): The maximum number of turns for the simulation. Defaults to 10.
        render_interval (int): How often (in turns) to generate visual feedback for the Architect.
                               Set to 0 to disable intermediate rendering. Defaults to 3.
    """
    logger = _setup_logging()

    run_type = structure_info.get("run")
    structure_name = structure_info.get("name")
    logger.info(f"--- Starting simulation for '{structure_name}' ({run_type}) ---")

    # Define output paths for this specific simulation run
    current_results_output_path = os.path.join(RESULTS_ROOT_DIR, structure_name, run_type)
    os.makedirs(current_results_output_path, exist_ok=True)
    architect_history_file = os.path.join(current_results_output_path, "architect_history.json")
    builder_history_file = os.path.join(current_results_output_path, "builder_history.json")

    # 1. Initialize Agents and World
    world, architect, builder = _initialize_simulation_components(architect_model, builder_model, structure_info, logger)

    # 2. Architect Starts the Project
    logger.info("\n--- Architect: Initiating project ---")
    # Architect's initial instruction to the Builder
    architect_message_to_builder = architect.start_project()
    _save_agent_history(architect.name, architect.get_history(), architect_history_file, logger)

    turn = 0
    architect_message_to_builder_final = None # Capture the last message for final check

    # --- Main Simulation Loop ---
    while turn < max_turns and architect_message_to_builder is not None:
        turn += 1
        logger.info(f"\n--- Turn {turn} ---")

        # 3. Builder's Phase: Receive instruction, process, and identify actions
        logger.info("Builder's Phase: Processing instructions and preparing actions.")
        builder_communication, builder_actions = _process_builder_turn(builder, architect_message_to_builder, logger)
        _save_agent_history(builder.name, builder.get_history(), builder_history_file, logger)

        # 4. Orchestrator: Update World Based on Builder's Actions
        # This executes the physical changes in the simulated world
        _orchestrate_world_update(world, builder_actions, turn, logger, builder, current_results_output_path)

        # 5. Orchestrator: Generate World Feedback for Architect (and optionally render)
        # This prepares the environment's response for the Architect's next turn
        if render_interval > 0 and turn % render_interval == 0:
            world_state_description, generated_image_paths = generate_world_feedback(world, turn, logger, current_results_output_path, run_type=run_type)
            architect.add_world_state_to_history(world_state_description, generated_image_paths)

        # 6. Architect's Phase: Process feedback and formulate next instruction
        logger.info("Architect's Phase: Receiving feedback and planning next steps.")
        architect_next_message_to_builder = architect.process_builder_output(builder_communication)
        
        # Store the last message *before* potentially breaking, for final status check
        architect_message_to_builder_final = architect_next_message_to_builder 
        architect_message_to_builder = architect_next_message_to_builder # Update message for next turn
        _save_agent_history(architect.name, architect.get_history(), architect_history_file, logger)
        
        # Check for [FINISH] token to allow early exit from simulation
        if architect_message_to_builder_final and "[FINISH]" in architect_message_to_builder_final:
            logger.info(f"--- [FINISH] token detected in Architect's message. Ending simulation early at Turn {turn}. ---")
            break # Exit the while loop

    # --- Post-Loop Cleanup: Always generate final world state and image ---
    logger.info("\n--- Simulation loop finished. Generating final world state outputs. ---")
    # The 'turn' variable will hold the last completed turn number
    generate_world_feedback(world, turn, logger, current_results_output_path, final=True, run_type=run_type)
    
    logger.info(f"--- Simulation for '{structure_name}' ({run_type}) Complete ---")

## --- Main Execution Block ---

if __name__ == "__main__":
    # Ensure the base results directory exists
    os.makedirs(RESULTS_ROOT_DIR, exist_ok=True)

    # Get a list of all structure directories (e.g., ['C1_bell', 'C4_flower_new'])
    structure_names = [
        d for d in os.listdir(BASE_STRUCTURES_DIR)
        if os.path.isdir(os.path.join(BASE_STRUCTURES_DIR, d))
    ]

    if not structure_names:
        logging.warning(f"No structure directories found in {BASE_STRUCTURES_DIR}. Exiting.")
        exit(0)

    logging.info(f"Found {len(structure_names)} structures to simulate: {structure_names}")

    # Sort structure names naturally for consistent processing order
    structure_names = natsorted(structure_names)
    
    # Iterate through each structure to run simulations
    # Currently set to run only the first structure found, change `[0]` to `[:]` to run all
    for structure_name in structure_names[:20]:
        logging.info(f"\n--- Processing structure: {structure_name} ---")

        current_structure_data_path = os.path.join(BASE_STRUCTURES_DIR, structure_name)
        
        # 4a. Load Structure JSON Data
        structure_data = None
        # Find any JSON file in the current structure's folder.
        # We'll pick the first one found, or you can add logic to pick a specific name
        json_files_in_current_structure = glob.glob(os.path.join(current_structure_data_path, "*.json"))
        
        if not json_files_in_current_structure:
            logging.error(f"No JSON file found in {current_structure_data_path}. Skipping this structure.")
            continue # Skip to the next structure in the loop
        
        json_to_load = json_files_in_current_structure[0] # Take the first JSON file found
        
        try:
            with open(json_to_load, 'r', encoding='utf-8') as f:
                structure_data = json.load(f)
            logging.info(f"Loaded structure JSON from: {os.path.basename(json_to_load)}")
        except IOError as e:
            logging.error(f"Error loading JSON from {json_to_load}: {e}. Skipping this structure.")
            continue
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON format in {json_to_load}: {e}. Skipping this structure.")
            continue

        # 4b. Collect All Relevant Image Files for the Current Structure
        image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
        required_keywords = ["front", "three_quarters", "above"] # Keywords to look for in the filename
        all_image_files = []
        
        for root, _, files in os.walk(current_structure_data_path):
            for filename in files:
                if os.path.splitext(filename)[1].lower() in image_extensions:
                    if any(keyword in filename.lower() for keyword in required_keywords):
                        all_image_files.append(os.path.join(root, filename))
        
        if not all_image_files:
            logging.warning(f"No images with 'front', 'three_quarters', or 'above' found for structure {structure_name} in {current_structure_data_path}. Proceeding without images for 'img_only' or 'img_json' runs.")

        # --- Create different structure info variants for different simulation runs ---
        # Each variant represents a different mode of input for the Architect (images, JSON, or both)
        original_structure_info = {'name': structure_name, 'json_data': structure_data, 'image_paths': all_image_files, 'run': "img_json"}
        
        json_only_structure_info = copy.deepcopy(original_structure_info)
        json_only_structure_info['image_paths'] = [] # Set to empty list as no images are provided
        json_only_structure_info['run'] = "json_only"

        img_only_structure_info = copy.deepcopy(original_structure_info)
        img_only_structure_info['json_data'] = None # No JSON data provided
        img_only_structure_info['run'] = "img_only"

        all_structure_variants = [
            original_structure_info,
            img_only_structure_info,
            json_only_structure_info
        ]

        # --- SIMULATION START ---
        logging.info(f"Starting simulations for structure '{structure_name}' across {len(all_structure_variants)} variants.")
        for current_structure_info in all_structure_variants:
            try:
                # Call run_simulation for each variant
                run_simulation(
                    architect_model=ARCHITECT_MODEL_NAME,
                    builder_model=BUILDER_MODEL_NAME,
                    structure_info=current_structure_info,
                    max_turns=MAX_TURNS,
                    render_interval=RENDER_INTERVAL
                )
                
            except Exception as e:
                logging.error(f"Error during simulation for '{structure_name}' ({current_structure_info.get('run')}): {e}", exc_info=True)
                # traceback.format_exc() is typically used with logging.error(exc_info=True)
                # logging.error(traceback.format_exc()) # Already handled by exc_info=True
                continue # Skip to the next structure variant if this one fails
    
    logging.info("\n--- All structure simulations finished! ---")
