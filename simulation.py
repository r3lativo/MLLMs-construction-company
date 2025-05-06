import openai
import json
import base64
import logging
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Jinja Setup ---
# Assuming prompt templates are in a 'prompts' directory relative to your script
try:
    jinja_env = Environment(loader=FileSystemLoader('prompts'))
    logger.info("Jinja environment initialized with 'prompts/' directory.")
except Exception as e:
    logger.error(f"Failed to initialize Jinja environment: {e}")
    jinja_env = None # Handle case where directory might not exist

def load_template(template_name: str, **kwargs) -> str:
    """Loads and renders a Jinja template."""
    if jinja_env is None:
        logger.error("Jinja environment not initialized. Cannot load template.")
        return f"Error: Template environment not available for {template_name}."
    try:
        template = jinja_env.get_template(template_name)
        return template.render(**kwargs)
    except TemplateNotFound:
        logger.error(f"Jinja template '{template_name}' not found.")
        return f"Error: Template '{template_name}' not found."
    except Exception as e:
        logger.error(f"Error rendering template '{template_name}': {e}")
        return f"Error rendering template '{template_name}': {e}"


class Agent:
    """Base class for a virtual agent interacting with vLLM."""
    def __init__(self, name: str, model: str, system_prompt: str, vllm_api_base: str = "http://localhost:8000/v1"):
        self.name = name
        self.model = model
        self.vllm_api_base = vllm_api_base
        # History starts with system prompt. Message content can be a string or a list for multi-modal (only used by Architect initially).
        self.history = [{"role": "system", "content": system_prompt}]
        self.client = openai.OpenAI(
            api_key="EMPTY",
            base_url=self.vllm_api_base,
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"{self.name} initialized.")

    def _call_model(self, messages):
        """Internal method to call the vLLM API with the given messages."""
        try:
            # Use the chat completions endpoint
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            # The response content can be a string or None
            return chat_completion.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error calling vLLM API for {self.name}: {e}")
            return None

    def add_message_to_history(self, role: str, content):
        """Adds a message to the agent's history."""
        # Content can be string or list of content blocks for multi-modal
        self.history.append({"role": role, "content": content})

    def get_history(self):
        """Returns the agent's current conversation history."""
        return self.history

    def clear_history(self):
        """Clears the conversation history except for the initial system prompt."""
        self.history = [self.history[0]] # Keep only the system prompt
        self.logger.info(f"{self.name}'s history cleared.")

class Architect(Agent):
    """Represents the Architect agent."""
    def __init__(self, model: str, system_prompt: str, structure_description_json: dict = None, structure_image_path: str = None, vllm_api_base: str = "http://localhost:8000/v1"):
        """
        Args:
            structure_description_json (dict): Dictionary representing the structure (e.g., loaded from JSON).
            structure_image_path (str): Path to an image file of the structure.
        """
        super().__init__("Architect", model, system_prompt, vllm_api_base)
        self.structure_description_json = structure_description_json
        self.structure_image_path = structure_image_path
        self._started = False

    def _prepare_initial_architect_input(self):
        """Prepares the *first* input (potentially multi-modal) for the Architect's model."""
        content = []

        # Add textual description if available
        if self.structure_description_json:
             # Convert JSON dict to a readable string format for the model
             json_string = json.dumps(self.structure_description_json, indent=2)
             content.append({"type": "text", "text": f"Here is the description of the structure to build in JSON format:\n```json\n{json_string}\n```"})

        # Add image if available (requires multi-modal model/vLLM support)
        if self.structure_image_path:
            try:
                # Read image and encode to base64
                with open(self.structure_image_path, "rb") as f:
                    image_data = f.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
                # Use data URL format for the image
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
                self.logger.info(f"Architect: Included image {self.structure_image_path} in initial model input.")
            except FileNotFoundError:
                 self.logger.error(f"Error: Image file not found at {self.structure_image_path}")
            except Exception as e:
                 self.logger.error(f"Error encoding image {self.structure_image_path}: {e}")


        if not content:
             content.append({"type": "text", "text": "No structure description or image provided."})
             self.logger.warning("Architect initialized without structure description or image.")

        # Add initial instruction text to prompt the model for its first response
        content.append({"type": "text", "text": "Based on the above structure details, what is the very first instruction I should give the Builder? Address the Builder directly."})

        return content if len(content) > 1 else content[0] # Return list if multi-modal, else single text block


    def start_project(self):
        """Generates the initial instruction to the Builder after processing structure info."""
        if not self._started:
            initial_architect_input_content = self._prepare_initial_architect_input()

            # Add this structured input to the Architect's history as the first 'user' message
            # This is the Architect feeding itself the project brief.
            self.add_message_to_history("user", initial_architect_input_content)

            self.logger.info(f"\n--- {self.name} processing initial project brief and generating first instruction ---")
            # Architect calls the model with its history (containing system prompt + initial brief)
            first_instruction_text = self._call_model(self.history)

            if first_instruction_text:
                 self.logger.info(f"{self.name} first instruction: {first_instruction_text[:200]}...")
                 # Add the generated instruction to history as assistant's response
                 self.add_message_to_history("assistant", first_instruction_text)
                 self._started = True
                 return first_instruction_text # This is the text message sent to the Builder
            else:
                 self.logger.error(f"{self.name}: Failed to generate first instruction.")
                 return None
        else:
            self.logger.warning(f"{self.name}: Project already started. Cannot start again.")
            return None

    def process_builder_output(self, builder_output_text: str, world_state_description: str = None):
        """
        Receives the Builder's raw text output (which might contain communication/actions)
        and generates the next instruction. Optionally includes world state feedback.
        Communication is text-only from this point on.
        """
        self.logger.info(f"\n--- {self.name} received output from Builder ---")
        self.logger.info(f"Builder raw output: {builder_output_text[:200]}...")

        # Add Builder's raw text output to Architect's history as if it's the user's turn
        # The Architect's model needs to understand this format (text + optional JSON/XML)
        self.add_message_to_history("user", builder_output_text)

        # Optionally, add world state description as system or user feedback
        if world_state_description:
             # Using 'user' role for feedback seems more natural for a chat model
             # Prepends 'World Feedback:' to make it clear to the model
             self.add_message_to_history("user", f"World Feedback: {world_state_description}")
             self.logger.info(f"Architect: Received world state feedback.")


        self.logger.info(f"\n--- {self.name} is generating next instruction ---")
        # Architect calls the model with the updated history
        architect_response_text = self._call_model(self.history)

        if architect_response_text:
            self.logger.info(f"{self.name}: {architect_response_text[:200]}...")
            # Add Architect's text response to history as assistant
            self.add_message_to_history("assistant", architect_response_text)
        else:
            self.logger.error(f"{self.name}: Failed to generate response.")

        return architect_response_text # This is the text message sent to the Builder


class Builder(Agent):
    """Represents the Builder agent."""
    # Define the expected structure for the JSON output block
    ACTION_JSON_SCHEMA = {
      "type": "object",
      "properties": {
        "communication": {
          "type": "string",
          "description": "Optional message to the Architect."
        },
        "actions": {
          "type": "array",
          "description": "List of actions to perform.",
          "items": {
            "type": "object",
            "properties": {
              "type": {"type": "string", "description": "Action type (e.g., place_block, remove_block)."},
              "params": {
                "type": "array",
                "description": "List of parameters for the action (e.g., [block_type, x, y, z]).",
                "items": {} # Can be any type
              }
            },
            "required": ["type", "params"]
          }
        }
      },
      "required": ["communication", "actions"] # Or make them not required depending on desired flexibility
    }
    # Define the tags for the JSON block within the text response
    JSON_BLOCK_START = "```json"
    JSON_BLOCK_END = "```"

    def __init__(self, model: str, system_prompt: str, action_handler, vllm_api_base: str = "http://localhost:8000/v1"):
        """
        Args:
            action_handler: An object with methods like place_block.
        """
        # Render the system prompt, including instructions on the JSON format
        action_json_schema_description = json.dumps(self.ACTION_JSON_SCHEMA, indent=2)
        # Use a Jinja template for the Builder's system prompt
        rendered_system_prompt = load_template(
            'builder_system_prompt.j2',
            json_block_start=self.JSON_BLOCK_START,
            json_block_end=self.JSON_BLOCK_END,
            action_json_schema_description=action_json_schema_description # Pass the schema description
        )
        if "Error:" in rendered_system_prompt:
             self.logger.error(f"Using fallback system prompt due to Jinja error: {rendered_system_prompt}")
             # Fallback prompt if template loading/rendering fails
             rendered_system_prompt = (
                 f"You are the Builder. Follow Architect instructions. "
                 f"Respond with communication text and a JSON block like ```json{{\"communication\": \"...\", \"actions\": [{{...}}]}}``` for actions."
            )


        super().__init__("Builder", model, rendered_system_prompt, vllm_api_base)
        self.action_handler = action_handler


    def process_architect_instruction(self, architect_message_text: str):
        """
        Receives instruction (text) from Architect, generates response (text + optional JSON actions).
        Returns raw model output (text), parsed communication text, and list of action details.
        """
        self.logger.info(f"\n--- {self.name} received instruction from Architect ---")
        self.logger.info(f"Architect: {architect_message_text[:200]}...")

        # Add Architect's text message to Builder's history
        self.add_message_to_history("user", architect_message_text)

        self.logger.info(f"\n--- {self.name} is generating response and potential actions ---")
        # Builder calls the model with its history (which is text-only from Architect)
        raw_model_output = self._call_model(self.history)

        if raw_model_output is None:
            self.logger.error(f"{self.name}: Failed to generate raw output.")
            return None, "Failed to generate response.", [] # raw_output, communication, actions

        # Add the raw model output to the Builder's history as assistant's turn
        self.add_message_to_history("assistant", raw_model_output)

        # Parse the raw output for structured communication and actions
        communication_text, actions = self._parse_output(raw_model_output)

        self.logger.info(f"{self.name}: Parsed communication: {communication_text[:100]}...")
        self.logger.info(f"{self.name}: Parsed actions: {actions}")

        return raw_model_output, communication_text, actions

    def _parse_output(self, raw_output: str):
        """Parses the raw model output for communication text and action details from a JSON block."""
        communication = raw_output # Default to treating whole output as communication
        actions = [] # List of dictionaries

        if raw_output is None:
             return communication, actions

        # Attempt to find and parse the JSON block
        json_start_index = raw_output.find(self.JSON_BLOCK_START)
        json_end_index = raw_output.find(self.JSON_BLOCK_END, json_start_index + len(self.JSON_BLOCK_START))

        if json_start_index != -1 and json_end_index != -1:
            json_string = raw_output[json_start_index + len(self.JSON_BLOCK_START):json_end_index].strip()
            # Clean up potential markdown code block indicator (e.g., 'json\n{...}')
            if json_string.lower().startswith("json"):
                 json_string = json_string[4:].strip()

            try:
                parsed_data = json.loads(json_string)
                # Extract communication from the JSON block if present
                communication_from_json = parsed_data.get("communication", "")
                # Extract actions from the JSON block
                actions = parsed_data.get("actions", [])

                # Combine communication text before the JSON block with text inside JSON
                communication_before_json = raw_output[:json_start_index].strip()
                communication = communication_before_json + ("\n" + communication_from_json if communication_from_json else "")

            except json.JSONDecodeError as e:
                self.logger.error(f"Error decoding JSON from Builder output: {e}. Treating as communication only.")
                # If JSON parsing fails, keep the whole raw output as communication
                communication = raw_output
                actions = [] # No reliable actions found
            except Exception as e:
                 self.logger.error(f"Unexpected error parsing Builder output: {e}. Treating as communication only.")
                 communication = raw_output
                 actions = []

        else:
             # If no JSON block, treat the whole output as communication
             self.logger.info("No JSON block found in Builder output. Treating as communication only.")
             communication = raw_output
             actions = [] # No structured actions found


        # Validate basic structure of parsed actions
        valid_actions = []
        for action in actions:
             if isinstance(action, dict) and "type" in action and "params" in action and isinstance(action["params"], list):
                  valid_actions.append(action)
             else:
                  self.logger.warning(f"Skipping invalid action format: {action}")
                  # Optionally add a message to communication about invalid action?

        return communication, valid_actions


    def execute_actions(self, actions: list):
        """Executes a list of parsed action dictionaries."""
        if not actions:
            self.logger.info(f"{self.name}: No actions to execute.")
            return True # Indicate success for no actions

        self.logger.info(f"\n--- {self.name} executing actions ---")
        all_succeeded = True
        for action in actions:
            action_type = action.get("type")
            params = action.get("params", [])

            if action_type is None:
                 self.logger.warning(f"Skipping action with no type: {action}")
                 all_succeeded = False
                 continue

            self.logger.info(f"Attempting action: {action_type} with params {params}")
            try:
                if hasattr(self.action_handler, action_type):
                    handler_method = getattr(self.action_handler, action_type)
                    # Basic parameter passing - World methods need to handle type conversion
                    success = handler_method(*params) # Call the method on the World object
                    self.logger.info(f"Action '{action_type}' executed. Success: {success}")
                    if not success:
                        all_succeeded = False
                else:
                    self.logger.error(f"Error: Unknown action type '{action_type}' on action handler.")
                    all_succeeded = False
            except TypeError as e:
                 self.logger.error(f"Error executing action '{action_type}' with params {params}: Parameter mismatch or incorrect arguments. {e}")
                 all_succeeded = False
            except Exception as e:
                self.logger.error(f"Unexpected error executing action '{action_type}': {e}")
                all_succeeded = False

        self.logger.info(f"--- {self.name} finished executing actions. Overall success: {all_succeeded} ---")
        return all_succeeded


class World:
    """Represents the virtual world where blocks are placed."""
    def __init__(self):
        self.blocks = {} # Stores blocks as {(x, y, z): color}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("World initialized.")

    def place_block(self, color: str, x: any, y: any, z: any):
        """Places a block at the specified coordinates with a color."""
        # Basic type conversion for parameters received from the model
        try:
            x, y, z = int(x), int(y), int(z)
        except ValueError:
            self.logger.error(f"Error: Invalid coordinates for place_block: ({x}, {y}, {z})")
            return False

        coords = (x, y, z)
        if coords in self.blocks:
            self.logger.warning(f"World: Replacing existing block at {coords}")
        self.blocks[coords] = color # Store block color
        self.logger.info(f"World: Placed {color} block at {coords}")
        return True # Indicate success

    def remove_block(self, x: any, y: any, z: any):
        """Removes a block at the specified coordinates."""
        try:
            x, y, z = int(x), int(y), int(z)
        except ValueError:
            self.logger.error(f"Error: Invalid coordinates for remove_block: ({x}, {y}, {z})")
            return False

        coords = (x, y, z)
        if coords in self.blocks:
            del self.blocks[coords]
            self.logger.info(f"World: Removed block at {coords}")
            return True
        else:
            self.logger.warning(f"World: No block found at {coords} to remove.")
            return False

    def get_block(self, x: int, y: int, z: int):
        """Gets the block color at the specified coordinates."""
        try:
            x, y, z = int(x), int(y), int(z)
        except ValueError:
             self.logger.error(f"Error: Invalid coordinates for get_block: ({x}, {y}, {z})")
             return None # Invalid coordinates

        return self.blocks.get((x, y, z))

    def get_state_as_json(self):
        """Returns the world state as a JSON-serializable dictionary."""
        # Convert tuple keys to strings for JSON compatibility
        return {f"{x},{y},{z}": color for (x, y, z), color in self.blocks.items()}

    def get_state_as_xml(self):
         """Returns the world state as a simple XML string (example)."""
         xml_string = "<world>\n"
         for (x, y, z), color in self.blocks.items():
              xml_string += f'  <block x="{x}" y="{y}" z="{z}" color="{color}"/>\n'
         xml_string += "</world>"
         return xml_string

    def get_state_description_for_architect(self):
        """Generates a textual description of the current world state for the Architect."""
        if not self.blocks:
            return "The world is currently empty."

        description = "Current blocks in the world:\n"
        # Limit the description length for context windows if needed
        items = []
        for (x, y, z), color in self.blocks.items():
            items.append(f"- {color} block at ({x}, {y}, {z})")
        # Sort for consistent output (optional)
        # items.sort()
        description += "\n".join(items[:20]) # List up to 20 blocks
        if len(items) > 20:
             description += f"\n... and {len(items) - 20} more blocks."
        return description

    def __str__(self):
        return self.get_state_description_for_architect()


# --- Simulation Orchestrator ---
def run_simulation(architect_model: str, builder_model: str, structure_info: dict, max_turns: int = 20, render_interval: int = 5):
    """
    Runs the Architect-Builder simulation.

    Args:
        architect_model (str): vLLM model name for the Architect.
        builder_model (str): vLLM model name for the Builder.
        structure_info (dict): Dictionary containing 'json_data' (dict) and/or 'image_path' (str).
        max_turns (int): Maximum turns to run the simulation.
        render_interval (int): How often (in turns) to generate a world render and feed back to Architect.
                                Set to 0 to disable rendering feedback.
    """
    logger.info("--- Starting Simulation ---")

    world = World()

    # --- Load and Render Prompts using Jinja ---
    # Prepare data to pass to Jinja templates
    architect_template_vars = {}
    builder_template_vars = {
        'json_block_start': Builder.JSON_BLOCK_START,
        'json_block_end': Builder.JSON_BLOCK_END,
        'action_json_schema_description': json.dumps(Builder.ACTION_JSON_SCHEMA, indent=2)
    }

    # Render system prompts
    architect_system_prompt = load_template('architect_system_prompt.j2', **architect_template_vars)
    builder_system_prompt = load_template('builder_system_prompt.j2', **builder_template_vars)

    # Initialize agents
    architect = Architect(
        model=architect_model,
        system_prompt=architect_system_prompt,
        structure_description_json=structure_info.get('json_data'),
        structure_image_path=structure_info.get('image_path')
    )
    builder = Builder(
        model=builder_model,
        system_prompt=builder_system_prompt, # The prompt is already rendered in Builder's __init__
        action_handler=world # World object handles the actions
    )

    # Start the conversation by the Architect processing the brief
    # The method returns the *text* message for the Builder
    architect_message_to_builder = architect.start_project()

    turn = 0
    # Continue simulation as long as max_turns not reached and Architect sends a message
    while turn < max_turns and architect_message_to_builder is not None:
        turn += 1
        logger.info(f"\n===== Turn {turn} =====")

        # --- Builder's Turn ---
        # Builder processes the Architect's text message
        raw_builder_output, builder_communication, builder_actions = builder.process_architect_instruction(architect_message_to_builder)

        # --- Orchestrator handles Builder's Actions ---
        if builder_actions:
            logger.info(f"Orchestrator: Executing Builder's actions...")
            success = builder.execute_actions(builder_actions)
            # Note: Action success/failure is logged but not explicitly fed back
            # to the Builder's history in this design. Can be added if needed.

        # --- Orchestrator handles World Rendering and Feedback ---
        world_state_feedback = None
        if render_interval > 0 and turn % render_interval == 0:
             logger.info(f"Orchestrator: Generating world feedback based on state (simulated rendering).")
             # In a real scenario, you'd call your renderer program here:
             # world_json = world.get_state_as_json() or world.get_state_as_xml()
             # save to temp file
             # run external renderer command (e.g., using subprocess)
             # Based on render (or just the world state), generate textual feedback
             world_state_feedback = world.get_state_description_for_architect()
             logger.info(f"World state description for Architect: {world_state_feedback[:100]}...")


        # --- Architect's Turn ---
        # The Architect receives the Builder's raw output text (communication + potential JSON)
        # and the optional world state feedback.
        architect_next_message_to_builder = architect.process_builder_output(raw_builder_output, world_state_feedback)

        # The Architect's response becomes the input for the Builder in the next turn
        architect_message_to_builder = architect_next_message_to_builder


    logger.info("\n--- Simulation Ended ---")
    logger.info("\nFinal World State:")
    logger.info(str(world))

# --- Example Usage ---
if __name__ == "__main__":
    # Create a dummy 'prompts' directory and template files for testing
    import os
    if not os.path.exists('prompts'):
        os.makedirs('prompts')
        with open('prompts/architect_system_prompt.j2', 'w') as f:
            f.write("""You are the Architect. Your role is to guide the Builder step-by-step to construct a structure.
The structure you need to build is described at the beginning of our conversation based on provided details.

Break down the building process into simple, clear instructions.
Receive the Builder's messages and any feedback on the current world state (preceded by 'World Feedback:').
Based on the Builder's progress and questions, provide the next instruction or clarification.
Be patient and ensure the Builder understands.
Do not try to perform actions yourself; only instruct the Builder.
""")
        with open('prompts/builder_system_prompt.j2', 'w') as f:
            f.write("""You are the Builder. Your role is to follow the Architect's instructions to build a structure.
The Architect will tell you what to build and how, piece by piece.
You can communicate back to the Architect to ask questions, provide updates, or ask for clarification.
You can also perform building actions by including a JSON object in your response wrapped in {{ json_block_start }} ... {{ json_block_end }} tags.
The JSON object must follow this schema:
{{ action_json_schema_description }}

Format your response with communication text followed by the JSON block if actions are needed.
Always report back after attempting actions.
""")
        logger.info("Created dummy 'prompts' directory and templates for example usage.")


    # IMPORTANT: Replace with your vLLM model names that support chat completion
    # If using images, ensure your Architect model is multi-modal (like LLaVA)
    # and your vLLM server is configured for it.
    ARCHITECT_MODEL_NAME = "NousResearch/Meta-Llama-3-8B-Instruct" # Example text model
    BUILDER_MODEL_NAME = "NousResearch/Meta-Llama-3-8-Instruct"   # Example text model - use same model or similar
    VLLM_API_BASE_URL = "http://localhost:8000/v1" # Your vLLM server address

    # Ensure vLLM is running with these models and --openai-compatible flag

    # --- Example Structure Information ---
    # Option 1: JSON description
    structure_json_data = {
      "name": "Simple Cube",
      "description": "A 1x1x1 cube made of a red block at (0,0,0).",
      "blocks": [
        {"position": [0, 0, 0], "color": "red"}
      ]
    }
    structure_info_json = {'json_data': structure_json_data}

    # Option 2: Image path (Requires a multi-modal Architect model and vLLM setup)
    # structure_info_image = {'image_path': 'path/to/your/structure_image.png'} # Replace with a real path

    # Option 3: Both
    # structure_info_both = {'json_data': structure_json_data, 'image_path': 'path/to/your/structure_image.png'} # Replace with a real path

    # Choose which structure info to use
    structure_info_for_sim = structure_info_json


    run_simulation(
        architect_model=ARCHITECT_MODEL_NAME,
        builder_model=BUILDER_MODEL_NAME,
        structure_info=structure_info_for_sim,
        max_turns=10, # Adjust max turns as needed
        render_interval=3 # Provide world state feedback to Architect every 3 turns
    )