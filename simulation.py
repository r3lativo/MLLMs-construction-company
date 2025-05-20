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
import glob # We'll use this to find JSON files dynamically
from natsort import natsorted
import pandas as pd # New: For DataFrame operations
import pyvista as pv # New: For 3D visualization

# --- Helper Functions ---

def _setup_logging():
    """Configures the root logger for file output."""
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    log_handler = RotatingFileHandler(
        f"mllms.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    log_handler.setFormatter(log_formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(log_handler)
    return root_logger

def _save_agent_history(agent_name: str, history: list, file_path: str, logger: logging.Logger):
    """Saves an agent's full conversation history to a JSON file."""
    #os.makedirs(HISTORY_DIR, exist_ok=True)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {agent_name}'s history to {file_path}")
    except Exception as e:
        logger.error(f"Error saving {agent_name}'s history to {file_path}: {e}")

def _save_world_state(world_data: list[dict], turn: int, logger: logging.Logger, current_results_output_path):
    """Saves the current world state to a JSON file, named by turn."""
    #os.makedirs(WORLD_STATE_DIR, exist_ok=True)
    file_path = os.path.join(current_results_output_path, f"world_state_turn_{turn:03d}.json")
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(world_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved world state for turn {turn} to {file_path}")
    except Exception as e:
        logger.error(f"Error saving world state for turn {turn} to {file_path}: {e}")

def _initialize_simulation_components(architect_model: str, builder_model: str, structure_info: dict, logger: logging.Logger):
    """Initializes the World, Architect, and Builder agents."""
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
    """Handles the Builder's turn: processes instruction and extracts actions/communication."""
    logger.info("Builder: Processing Architect's instruction...")
    raw_builder_output, builder_communication, builder_actions = \
        builder.process_architect_instruction(architect_message)
    logger.info(f"Builder: Communication: {builder_communication[:100]}...")
    logger.info(f"Builder: Actions identified: {len(builder_actions.actions)} types, {sum(len(v) for v in builder_actions.actions.values())} total actions.")
    return builder_communication, builder_actions

def _orchestrate_world_update(world: World, builder_actions: any, turn: int, logger: logging.Logger, builder: Builder, current_results_output_path):
    """Executes Builder's actions and saves the resulting world state."""
    if builder_actions:
        logger.info(f"Orchestrator: Executing Builder's actions for turn {turn}...")
        builder.execute_actions(builder_actions)
    
    # Save the world state after actions for this turn
    current_world_state_data = world.get_state_as_json()
    _save_world_state(current_world_state_data, turn, logger, current_results_output_path)


def plot_structure(block_row, plotter):
    """
    Plots a 3D block for a single row in the DataFrame.
    """
    block_color, x, y, z = block_row
    cube = pv.Cube(center=(x, z, y), x_length=1, y_length=1, z_length=1)
    plotter.add_mesh(cube, color=block_color, show_edges=True)

def take_screenshots(plotter, turn, output_path):
    # Initialize plotter
    plotter.set_background("pv") # Sets background to PyVista's default gradient

    created_screenshot_paths = []
    
    # Define angles/views for screenshots
    angles = {'front': 0, 'three_q': 45}

    # ABOVE view
    plotter.camera.elevation = 90
    plotter.camera.azimuth = 0
    plotter.render()
    above_path = os.path.join(output_path, f"ws_turn_{turn:03d}_above.jpg")
    plotter.screenshot(above_path)
    created_screenshot_paths.append(above_path)

    # AROUND views
    plotter.camera.elevation = 30 # Set common elevation for around views
    for label, angle in angles.items():
        plotter.camera.azimuth = angle
        plotter.render() # Render the structure again for each new view
        screenshot_path = os.path.join(output_path, f"ws_turn_{turn:03d}_{label}.jpg")
        plotter.screenshot(screenshot_path)
        created_screenshot_paths.append(screenshot_path)
    
    return created_screenshot_paths


def generate_world_feedback(world: World, turn: int, logger: logging.Logger, output_path: str):

    # --- Text feedback generation ---
    world_state_description = f"Current World State:\n{world.get_state_description_for_architect()}"

    # --- Image feedback generation ---
    plotter = pv.Plotter(off_screen=True, window_size=[640, 640])

    current_world_state_df = pd.DataFrame(world.get_state_as_json())
    combined_df = pd.concat([TERRAIN_DF, current_world_state_df], ignore_index=True)

    if not combined_df.empty:
        combined_df.apply(lambda row: plot_structure(row, plotter), axis=1)
    
    # Take screenshots and get their paths
    generated_image_paths = take_screenshots(plotter, turn, output_path)
    plotter.close()

    # --- return things to pass to architect ---
    return world_state_description, generated_image_paths


"""
def _generate_world_feedback(world: World, turn: int, renderer_command: list, logger: logging.Logger) -> str:
    #Generates feedback about the world state for the Architect, optionally
    #using an external renderer.
    logger.info(f"Orchestrator: Generating world feedback for Architect (Turn {turn}).")
    world_state_feedback = None

    if renderer_command:
        # Get the world state as JSON data (list of dicts)
        world_state_json_data = world.get_state_as_json()
        
        # Write to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            temp_file_path = tmp_file.name
            json.dump(world_state_json_data, tmp_file, indent=2)
            logger.info(f"World state saved to temporary file for renderer: {temp_file_path}")
        
        try:
            full_command = renderer_command + [temp_file_path]
            logger.info(f"Running renderer command: {' '.join(full_command)}")
            process = subprocess.run(full_command, capture_output=True, text=True, check=True)
            render_output = process.stdout.strip()
            logger.info(f"Renderer output (first 100 chars): {render_output[:100]}...")
            world_state_feedback = f"Renderer reported: {render_output}"
        except FileNotFoundError:
            logger.error(f"Renderer command not found: {full_command[0]}. Falling back to direct world state description.")
            world_state_feedback = world.get_state_description_for_architect()
        except subprocess.CalledProcessError as e:
            logger.error(f"Renderer command failed with error code {e.returncode}: {e.stderr.strip()}. Falling back to direct world state description.")
            world_state_feedback = world.get_state_description_for_architect()
        except Exception as e:
            logger.error(f"Unexpected error running renderer command: {e}. Falling back to direct world state description.")
            world_state_feedback = world.get_state_description_for_architect()
        finally:
            try:
                os.remove(temp_file_path)
                logger.info(f"Removed temporary file: {temp_file_path}")
            except OSError as e:
                logger.error(f"Error removing temporary file {temp_file_path}: {e}")
    else:
        # If no renderer command, just use the direct textual description (which is JSON string or "empty")
        world_state_feedback = world.get_state_description_for_architect()
        logger.info("Using direct world state description for feedback.")

    print(f"World state feedback for Architect:\n{world_state_feedback[:200]}...") # Print a snippet for console visibility
    return world_state_feedback
"""

"""
def _process_architect_turn(architect: Architect, builder_communication: str, world_state_feedback: str, logger: logging.Logger):
    #Handles the Architect's turn: processes feedback and generates next instruction.
    logger.info("Architect: Processing Builder's communication and world feedback...")
    architect_next_message = architect.process_builder_output(builder_communication, world_state_feedback)
    logger.info(f"Architect: Next instruction to Builder: {architect_next_message[:100]}...")
    return architect_next_message
"""

def save_final_states(world, current_results_output_path):
    final_world_state = world.get_state_as_json()
    world_state_path = os.path.join(current_results_output_path, "final_world_state.json")
    with open(world_state_path, 'w') as f:
        json.dump(final_world_state, f, indent=2)

    plotter = pv.Plotter(off_screen=True, window_size=[640, 640])

    current_world_state_df = pd.read_json(final_world_state)
    terrain_df = pd.read_json(TERRAIN_DF)
    combined_df = pd.concat([terrain_df, current_world_state_df], ignore_index=True)

    if not combined_df.empty:
        combined_df.apply(lambda row: plot_structure(row, plotter), axis=1)

    plotter.set_background("pv") # Sets background to PyVista's default gradient
    # AROUND views
    plotter.camera.elevation = 30 # Set common elevation for around views
    plotter.camera.azimuth = 45
    plotter.render() # Render the structure again for each new view
    screenshot_path = os.path.join(current_results_output_path, f"final_world_state.jpg")
    plotter.screenshot(screenshot_path)
    plotter.close()

# --- Main Simulation Function ---

def run_simulation(
        architect_model: str,
        builder_model: str,
        structure_info: dict,
        max_turns: int = 10,
        render_interval: int = 3
    ):
    """
    Runs the Architect-Builder collaborative simulation.
    """
    logger = _setup_logging()

    run = current_structure_info.get("run")
    structure_name = structure_info.get("name")
    logger.info(f"Running simulation for '{structure_name}' ({run})...")

    current_results_output_path = os.path.join(RESULTS_ROOT_DIR, structure_name, run)
    os.makedirs(current_results_output_path, exist_ok=True)
    architect_history_file = os.path.join(current_results_output_path, "architect_history.json")
    builder_history_file = os.path.join(current_results_output_path, "builder_history.json")

    # 1. Initialize Agents and World
    world, architect, builder = _initialize_simulation_components(architect_model, builder_model, structure_info, logger)

    # 2. Architect Starts the Project
    logger.info("\n--- Architect: Initiating project ---")
    architect_message_to_builder = architect.start_project()
    _save_agent_history(architect.name, architect.get_history(), architect_history_file, logger)


    turn = 0
    while turn < max_turns and architect_message_to_builder is not None:
        turn += 1
        logger.info(f"\n===== Turn {turn} =====")

        # 3. Builder's Turn: Receive, Process, Act
        logger.info(f"\n--- Turn {turn}: Builder's Phase ---")
        builder_communication, builder_actions = _process_builder_turn(builder, architect_message_to_builder, logger)
        _save_agent_history(builder.name, builder.get_history(), builder_history_file, logger)

        # 4. Orchestrator: Update World Based on Builder's Actions
        _orchestrate_world_update(world, builder_actions, turn, logger, builder, current_results_output_path)

        # 5. Orchestrator: Generate World Feedback for Architect (and optionally render)
        if render_interval > 0 and turn % render_interval == 0:
            world_state_description, generated_image_paths = generate_world_feedback(world, turn, logger, current_results_output_path)
            architect.add_world_state_to_history(world_state_description, generated_image_paths)
        #else:
            # If not rendering this turn, still provide Architect with current state description
            #world_state_feedback = world.get_state_description_for_architect()
            #logger.info(f"Orchestrator: Skipping external render this turn. Using direct world state description as feedback.")

        # --- FINISH if the token is in the architect's last message
        if "[FINISH]" in architect_message_to_builder:
            save_final_states(world, current_results_output_path)
            logger.info("\n--- Simulation Ended ---")
            break

        # 6. Architect's Turn: Process Feedback and Instruct Next
        logger.info(f"\n--- Turn {turn}: Architect's Phase ---")
        #architect_next_message_to_builder = _process_architect_turn(architect, builder_communication, world_state_feedback, logger)
        architect_next_message_to_builder = architect.process_builder_output(builder_communication)
        architect_message_to_builder = architect_next_message_to_builder # Update message for next turn
        _save_agent_history(architect.name, architect.get_history(), architect_history_file, logger)



ARCHITECT_MODEL_NAME = config.get("architect_model_id")
BUILDER_MODEL_NAME = config.get("builder_model_id")
VLLM_API_BASE_URL = "http://localhost:8000/v1"

JINJA_ENV = Environment(loader=FileSystemLoader('prompts'))

BASE_STRUCTURES_DIR = "data/structures/gold-processed"
RESULTS_ROOT_DIR = "new_results" # New root directory for all results

RENDERER_CMD = None # Or your actual command
TERRAIN_DF = pd.read_json("data/structures/terrain.json")

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

    # ... (start of the for loop) ...
    structure_names = natsorted(structure_names)
    
    for structure_name in [structure_names[0]]: # sorted() helps with consistent order
        logging.info(f"\n--- Starting processing for structure: {structure_name} ---")

        current_structure_data_path = os.path.join(BASE_STRUCTURES_DIR, structure_name)
        main_results_output_path = os.path.join(RESULTS_ROOT_DIR, structure_name)

        # Create the specific results directory for this structure
        os.makedirs(main_results_output_path, exist_ok=True)

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
            with open(json_to_load, 'r') as f:
                structure_data = json.load(f)
        except Exception as e:
            logging.error(f"An unexpected error occurred loading JSON from {json_to_load}: {e}. Skipping this structure.")
            continue

        # 4b. Collect All Image Files for the Current Structure
        image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
        # Keywords to look for in the filename (case-insensitive)
        required_keywords = ["front", "three_quarters", "above"]
        all_image_files = []
        
        for root, _, files in os.walk(current_structure_data_path):
            for filename in files:
                # Check for image extension
                if os.path.splitext(filename)[1].lower() in image_extensions:
                    # Check if any of the required keywords are in the filename (case-insensitive)
                    if any(keyword in filename.lower() for keyword in required_keywords):
                        all_image_files.append(os.path.join(root, filename))
        
        if not all_image_files:
            logging.warning(f"No images with 'front', 'three_quarters', or 'above' found for structure {structure_name} in {current_structure_data_path}. Proceeding without images.")


        # --- Create different structure info for different runs ---
        original_structure_info = {'name': structure_name, 'json_data': structure_data, 'image_paths': all_image_files, 'run': "img_json"}
        
        json_only_structure_info = copy.deepcopy(original_structure_info)
        json_only_structure_info['image_paths'] = None
        json_only_structure_info['run'] = "json_only"

        img_only_structure_info = copy.deepcopy(original_structure_info)
        img_only_structure_info['json_data'] = None
        img_only_structure_info['run'] = "img_only"

        all_structure_variants = [
            original_structure_info,
            img_only_structure_info,
            json_only_structure_info
        ]

        # --- SIMULATION START ---
        
        logging.info(f"Starting simulations for structure '{structure_name}'...")
        for current_structure_info in all_structure_variants:
            try:
                # Call run_simulation and capture its return values
                run_simulation(
                    architect_model=ARCHITECT_MODEL_NAME,
                    builder_model=BUILDER_MODEL_NAME,
                    structure_info=current_structure_info,
                    max_turns=20,
                    render_interval=3
                )
                
            except Exception as e:
                logging.error(f"Error encountered during simulation for '{structure_name}': {e}")
                import traceback
                logging.error(traceback.format_exc()) # Print full traceback for debugging
                continue # Skip to the next structure if this one fails
    
    # ... (end of the for loop) ...

    logging.info("\n--- All structure simulations finished! ---")
