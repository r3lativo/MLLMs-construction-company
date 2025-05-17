import json
import logging
from logging.handlers import RotatingFileHandler
from architect import Architect
from builder import Builder
from action_handler import World
from config import config
from jinja2 import Environment, FileSystemLoader

# --- Logging Setup ---

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
log_handler = RotatingFileHandler(
    f"coco.log",
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=5
)
log_handler.setFormatter(log_formatter)

logger = logging.getLogger()  # Get the root logger
logger.setLevel(logging.DEBUG)
logger.addHandler(log_handler)

JINJA_ENV = Environment(loader=FileSystemLoader('prompts'))


def run_simulation(
        architect_model: str,
        builder_model: str,
        structure_info: dict,
        max_turns: int = 5,
        render_interval: int = 1,
        world_output_format: str = 'json',
        renderer_command: list = None
    ):
    """
    Runs the Architect-Builder simulation.
    """
    logger.info("--- Starting Simulation ---")

    world = World()

    architect = Architect(
        model=architect_model,
        system_prompt=JINJA_ENV.get_template('a_sys_prompt.jinja'),
        init_prompt=JINJA_ENV.get_template('a_init_prompt.jinja'),
        structure_description_json=structure_info.get('json_data'),
        structure_image_path=structure_info.get('image_path')
    )
    builder = Builder(
        model=builder_model,
        system_prompt=JINJA_ENV.get_template('b_sys_prompt.jinja'),
        action_handler=world
    )

    architect_message_to_builder = architect.start_project()

    turn = 0
    while turn < max_turns and architect_message_to_builder is not None:
        turn += 1
        logger.info(f"\n===== Turn {turn} =====")

        # --- Builder's Turn ---
        # Builder processes the Architect's text message
        (
            raw_builder_output,
            builder_communication,
            builder_actions
        ) = builder.process_architect_instruction(architect_message_to_builder)

        # --- Orchestrator handles Builder's Actions ---
        if builder_actions: # builder_actions is now a list of Action Pydantic models
            logger.info(f"Orchestrator: Executing Builder's actions...")
            builder.execute_actions(builder_actions)

        # --- Orchestrator handles World Rendering and Feedback ---
        world_state_feedback = None
        if render_interval > 0 and turn % render_interval == 0:
            logger.info(f"Orchestrator: Generating world feedback based on state.")
            world_state_data = None
            if world_output_format == 'json':
                world_state_data = world.get_state_as_json()
            elif world_output_format == 'xml':
                world_state_data = world.get_state_as_xml()
            else:
                logger.warning(f"Unsupported world_output_format: {world_output_format}. Skipping render/feedback.")

            if world_state_data is not None:
                import tempfile
                import os
                file_extension = 'json' if world_output_format == 'json' else 'xml'
                with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{file_extension}', delete=False) as tmp_file:
                    temp_file_path = tmp_file.name
                    if world_output_format == 'json':
                        json.dump(world_state_data, tmp_file, indent=2)
                    elif world_output_format == 'xml':
                         tmp_file.write(world_state_data)
                    logger.info(f"World state saved to temporary file: {temp_file_path}")
                
                """
                render_output = None
                if renderer_command:
                    full_command = renderer_command + [temp_file_path]
                    try:
                        logger.info(f"Running renderer command: {' '.join(full_command)}")
                        # import subprocess
                        # process = subprocess.run(full_command, capture_output=True, text=True, check=True)
                        # render_output = process.stdout.strip()
                        # logger.info(f"Renderer output (first 100 chars): {render_output[:100]}...")
                        simulated_render_output = f"Rendering complete for {os.path.basename(temp_file_path)}. World contains {len(world.blocks)} blocks."
                        render_output = simulated_render_output
                        logger.info(f"Simulated renderer output: {render_output}")

                    except FileNotFoundError:
                        logger.error(f"Renderer command not found: {full_command[0]}")
                    except Exception as e:
                        logger.error(f"Error running renderer command: {e}")
                    finally:
                         try:
                              os.remove(temp_file_path)
                              logger.info(f"Removed temporary file: {temp_file_path}")
                         except OSError as e:
                              logger.error(f"Error removing temporary file {temp_file_path}: {e}")
                
                if render_output:
                    world_state_feedback = f"Renderer reported: {render_output}"
                else:
                    world_state_feedback = world.get_state_description_for_architect()
                    logger.info("Using direct world state description for feedback.")
                """
                world_state_feedback = world.get_state_description_for_architect()
                logger.info(f"World state feedback for Architect: {world_state_feedback}...")
                print(f"World state feedback for Architect: {world_state_feedback}...")

        print(builder.get_history())

        # --- Architect's Turn ---
        architect_next_message_to_builder = architect.process_builder_output(builder_communication, world_state_feedback)
        architect_message_to_builder = architect_next_message_to_builder

        print(architect.get_history())

        print(world.get_state_description_for_architect)


    logger.info("\n--- Simulation Ended ---")
    logger.info("\nFinal World State:")
    logger.info(str(world))


if __name__ == "__main__":

    # IMPORTANT: Replace with your vLLM model names that support chat completion
    # If using images, ensure your Architect model is multi-modal (like LLaVA)
    # and your vLLM server is configured for it.
    ARCHITECT_MODEL_NAME = config.get("architect_model_id")
    BUILDER_MODEL_NAME = config.get("builder_model_id")
    VLLM_API_BASE_URL = "http://localhost:8000/v1" # Your vLLM server address

    # Ensure vLLM is running with these models and --openai-compatible flag
    flower_json = "data/structures/gold-processed/C4_flower_new/C4_flower_new.json"
    with open(flower_json, 'r') as f:
        structure_json_data = json.load(f)

    # --- Example Structure Information ---
    structure_info = {'json_data': structure_json_data}#, 'image_path': 'data/structures/gold-processed/C4_flower_new/screenshot_C4_flower_new_0_front.jpg'}

    RENDERER_CMD = None # Set to None if you don't have a renderer or don't want to run it

    run_simulation(
        architect_model=ARCHITECT_MODEL_NAME,
        builder_model=BUILDER_MODEL_NAME,
        structure_info=structure_info,
        max_turns=10, # Adjust max turns as needed
        render_interval=3, # Provide world state feedback to Architect every 3 turns
        world_output_format='json',
        renderer_command=RENDERER_CMD # Pass your renderer command here
    )