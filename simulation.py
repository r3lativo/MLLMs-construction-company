import json
import logging
from logging.handlers import RotatingFileHandler
import os
import shutil
import tempfile
import subprocess # Needed for running the renderer command

from architect import Architect
from builder import Builder
from action_handler import World
from config import config
from jinja2 import Environment, FileSystemLoader

# --- Global Constants & Setup ---
HISTORY_DIR = "chat_histories"
WORLD_STATE_DIR = "world_states"
ARCHITECT_HISTORY_FILE = os.path.join(HISTORY_DIR, "architect_history.json")
BUILDER_HISTORY_FILE = os.path.join(HISTORY_DIR, "builder_history.json")

JINJA_ENV = Environment(loader=FileSystemLoader('prompts'))

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

def _cleanup_output_directories(logger: logging.Logger):
    """Removes existing output directories to ensure a clean slate for a new simulation."""
    for directory in [HISTORY_DIR, WORLD_STATE_DIR]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            logger.info(f"Removed old directory: {directory}")

def _save_agent_history(agent_name: str, history: list, file_path: str, logger: logging.Logger):
    """Saves an agent's full conversation history to a JSON file."""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {agent_name}'s history to {file_path}")
    except Exception as e:
        logger.error(f"Error saving {agent_name}'s history to {file_path}: {e}")

def _save_world_state(world_data: list[dict], turn: int, logger: logging.Logger):
    """Saves the current world state to a JSON file, named by turn."""
    os.makedirs(WORLD_STATE_DIR, exist_ok=True)
    file_path = os.path.join(WORLD_STATE_DIR, f"world_state_turn_{turn:03d}.json")
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

def _orchestrate_world_update(world: World, builder_actions: any, turn: int, logger: logging.Logger, builder: Builder): # Add builder to parameters
    """Executes Builder's actions and saves the resulting world state."""
    if builder_actions:
        logger.info(f"Orchestrator: Executing Builder's actions for turn {turn}...")
        builder.execute_actions(builder_actions)
    
    # Save the world state after actions for this turn
    current_world_state_data = world.get_state_as_json()
    _save_world_state(current_world_state_data, turn, logger)

def _generate_world_feedback(world: World, turn: int, renderer_command: list, logger: logging.Logger) -> str:
    """
    Generates feedback about the world state for the Architect, optionally
    using an external renderer.
    """
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

def _process_architect_turn(architect: Architect, builder_communication: str, world_state_feedback: str, logger: logging.Logger):
    """Handles the Architect's turn: processes feedback and generates next instruction."""
    logger.info("Architect: Processing Builder's communication and world feedback...")
    architect_next_message = architect.process_builder_output(builder_communication, world_state_feedback)
    logger.info(f"Architect: Next instruction to Builder: {architect_next_message[:100]}...")
    return architect_next_message


# --- Main Simulation Function ---

def run_simulation(
        architect_model: str,
        builder_model: str,
        structure_info: dict,
        max_turns: int = 5,
        render_interval: int = 1,
        renderer_command: list = None
    ):
    """
    Runs the Architect-Builder collaborative simulation.
    """
    logger = _setup_logging()
    logger.info("--- Starting Simulation ---")

    _cleanup_output_directories(logger)

    # 1. Initialize Agents and World
    world, architect, builder = _initialize_simulation_components(architect_model, builder_model, structure_info, logger)

    # 2. Architect Starts the Project
    logger.info("\n--- Architect: Initiating project ---")
    architect_message_to_builder = architect.start_project()
    _save_agent_history(architect.name, architect.get_history(), ARCHITECT_HISTORY_FILE, logger)


    turn = 0
    while turn < max_turns and architect_message_to_builder is not None:
        turn += 1
        logger.info(f"\n===== Turn {turn} =====")

        # 3. Builder's Turn: Receive, Process, Act
        logger.info(f"\n--- Turn {turn}: Builder's Phase ---")
        builder_communication, builder_actions = _process_builder_turn(builder, architect_message_to_builder, logger)
        _save_agent_history(builder.name, builder.get_history(), BUILDER_HISTORY_FILE, logger)

        # 4. Orchestrator: Update World Based on Builder's Actions
        _orchestrate_world_update(world, builder_actions, turn, logger, builder)

        # 5. Orchestrator: Generate World Feedback for Architect (and optionally render)
        world_state_feedback = None
        if render_interval > 0 and turn % render_interval == 0:
            world_state_feedback = _generate_world_feedback(world, turn, renderer_command, logger)
        else:
            # If not rendering this turn, still provide Architect with current state description
            world_state_feedback = world.get_state_description_for_architect()
            logger.info(f"Orchestrator: Skipping external render this turn. Using direct world state description as feedback.")

        if "[FINISH]" in architect_message_to_builder:
            break

        # 6. Architect's Turn: Process Feedback and Instruct Next
        logger.info(f"\n--- Turn {turn}: Architect's Phase ---")
        architect_next_message_to_builder = _process_architect_turn(architect, builder_communication, world_state_feedback, logger)
        architect_message_to_builder = architect_next_message_to_builder # Update message for next turn
        _save_agent_history(architect.name, architect.get_history(), ARCHITECT_HISTORY_FILE, logger)


    logger.info("\n--- Simulation Ended ---")
    logger.info(f"\nFinal World State:\n{world}")


if __name__ == "__main__":
    ARCHITECT_MODEL_NAME = config.get("architect_model_id")
    BUILDER_MODEL_NAME = config.get("builder_model_id")
    VLLM_API_BASE_URL = "http://localhost:8000/v1" # Your vLLM server address

    # Load structure information
    flower_json = "data/structures/gold-processed/C4_flower_new/C4_flower_new.json"
    try:
        with open(flower_json, 'r') as f:
            structure_json_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Structure JSON file not found: {flower_json}. Please ensure the path is correct.")
        exit(1)
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from: {flower_json}. Please check file format.")
        exit(1)

    directory_to_search = "data/structures/gold-processed/C1_bell/"
    image_extensions = ('.jpg', '.jpeg', '.png', '.webp')

    all_image_files = [
        os.path.join(root, filename)
        for root, _, files in os.walk(directory_to_search) # Traverses through main folder and all subfolders
        for filename in files                               # Iterates over each file found
        if os.path.splitext(filename)[1].lower() in image_extensions # Checks if the file's extension is an image type
    ]

    structure_info = {
        #'json_data': structure_json_data,
        'image_paths': all_image_files
    }

    # --- Configure Renderer Command ---
    # IMPORTANT: Replace with the actual command to run your external renderer.
    # It should accept a single argument: the path to the temporary JSON file.
    # Example: RENDERER_CMD = ["/path/to/your/renderer_executable", "--output_image", "output.png"]
    # For testing, you might use a simple script that just echoes the file path or processes it.
    RENDERER_CMD = None # Set to None if you don't have a renderer or don't want to run it
    # RENDERER_CMD = ["python", "scripts/mock_renderer.py"] # Example for a mock renderer script

    run_simulation(
        architect_model=ARCHITECT_MODEL_NAME,
        builder_model=BUILDER_MODEL_NAME,
        structure_info=structure_info,
        max_turns=10,
        render_interval=1, # Provide world state feedback to Architect every turn
        renderer_command=RENDERER_CMD
    )
