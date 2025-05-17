from helper import Agent
import logging
import json

logger = logging.getLogger(__name__)


class Architect(Agent):
    """Represents the Architect agent."""
    def __init__(self, model: str, system_prompt: str, init_prompt: str, structure_description_json: dict = None, structure_image_path: str = None, vllm_api_base: str = "http://localhost:8000/v1"):
        """
        Args:
            structure_description_json (dict): Dictionary representing the structure (e.g., loaded from JSON).
            structure_image_path (str): Path to an image file of the structure.
        """
        super().__init__("Architect", model, system_prompt, vllm_api_base)
        self.init_prompt = init_prompt
        self.structure_description_json = structure_description_json
        self.structure_image_path = structure_image_path
        self._started = False

    def _prepare_initial_architect_input(self):
        """Prepares the *first* input (potentially multi-modal) for the Architect's model using Jinja2."""
        content = []

        # Determine use_json and use_img status
        use_json = bool(self.structure_description_json)
        use_img = bool(self.structure_image_path)

        # Prepare the JSON string for the template if needed
        json_string_for_template = None
        if use_json:
            try:
                # Convert JSON dict to a readable string format for the model
                json_string_raw = json.dumps(self.structure_description_json)
                # Wrap JSON in markdown code block for clarity
                json_string_for_template = f"```json\n{json_string_raw}\n```"
            except TypeError as e:
                self.logger.error(f"Error serializing JSON description: {e}")
                use_json = False # Disable JSON inclusion if serialization fails
                json_string_for_template = None

        # Render the text part using the Jinja template
        initial_text = self.init_prompt.render(
            use_json=use_json,
            use_img=use_img,
            json_description=json_string_for_template # Pass the formatted JSON string
        )

        # Add the rendered text part to the content list if it's not empty
        # The template handles producing an empty string if neither image nor JSON is used.
        if initial_text.strip():
            content.append({"type": "text", "text": initial_text.strip()})

        # Add image if available
        if use_img:
            # Assume the model API accepts a file path or URL directly in the 'url' field.
            # The original error handling for file not found and encoding is removed
            # as base64 encoding is skipped. If the file might not exist, you might
            # want to add a check here (e.g., using os.path.exists).
            image_url_or_path = self.structure_image_path # Use the path directly

            # Add the image part to the content list
            # The exact format might depend on the specific API, this is a common pattern
            try:
                content.append({"type": "image_url", "image_url": {"url": image_url_or_path}})
                self.logger.debug(f"Architect: Included image {image_url_or_path} in initial model input.")
            except Exception as e:
                self.logger.error(f"Error adding image {image_url_or_path} to input: {e}")

        # Handle the case where neither description nor image was provided
        if not content:
            self.logger.warning("Architect initialized without structure description or image.")

        # Return the content list. Multi-modal inputs are typically always lists.
        return content

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
                 self.logger.debug(f"{self.name} first instruction: {first_instruction_text}")
                 print(f"{self.name} first instruction: {first_instruction_text}")
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
             self.add_message_to_history("user", f"Current World State: {world_state_description}")
             self.logger.info(f"Architect: Received world state feedback.")

        self.logger.info(f"\n--- {self.name} is generating next instruction ---")
        # Architect calls the model with the updated history
        architect_response_text = self._call_model(self.history)

        if architect_response_text:
            self.logger.debug(f"{self.name}: {architect_response_text}")
            print(f"{self.name}: {architect_response_text}")
            # Add Architect's text response to history as assistant
            self.add_message_to_history("assistant", architect_response_text)
        else:
            self.logger.error(f"{self.name}: Failed to generate response.")

        return architect_response_text # This is the text message sent to the Builder
