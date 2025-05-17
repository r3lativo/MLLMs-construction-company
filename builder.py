from helper import Agent
import logging
import json
from helper import GroupedActionOutput
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class Builder(Agent):
    """Represents the Builder agent."""
    # Define the tags for the JSON block within the text response
    JSON_BLOCK_START = "```json"
    JSON_BLOCK_END   = "```"

    def __init__(self, model: str, system_prompt: str, action_handler, vllm_api_base: str = "http://localhost:8000/v1"):
        """
        Args:
            jinja_template (str): The prompt for the Builder.
            action_handler: An object with methods like place_block.
        """
        # The system prompt will be a base, and specific prompts will be used for each model call
        super().__init__("Builder", model, system_prompt, vllm_api_base)
        self.action_handler = action_handler

    def process_architect_instruction(self, architect_message_text: str):
        """
        Receives instruction (text) from Architect, generates response (JSON actions + optional text).
        Returns raw model output (text), parsed communication text, and list of action details.
        """
        self.logger.info(f"\n--- {self.name} received instruction from Architect ---")

        # Add Architect's text message to Builder's history
        self.add_message_to_history("user", architect_message_text)

        self.logger.info(f"\n--- {self.name} is generating response and potential actions ---")
        # Builder calls the model with its history (which is text-only from Architect)
        raw_model_output = self._call_model(self.history)
        self.logger.debug(f"Builder raw output: {raw_model_output}")
        print(f"Builder raw output: {raw_model_output}")

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

        # Attempt to find and parse the JSON block
        json_start_index = raw_output.find(self.JSON_BLOCK_START)
        json_end_index = raw_output.find(self.JSON_BLOCK_END, json_start_index + len(self.JSON_BLOCK_START))

        if json_start_index != -1 and json_end_index != -1:
            json_string = raw_output[json_start_index + len(self.JSON_BLOCK_START):json_end_index].strip()
            # Clean up potential markdown code block indicator (e.g., 'json\n{...}')

            try:
                parsed_data = json.loads(json_string)
                actions, communication = self._validate_and_parse_grouped_actions(parsed_data)

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

        return communication, actions

    def _validate_and_parse_grouped_actions(data: dict):
        """
        Validates a dictionary against the GroupedActionOutput Pydantic model.

        Args:
            data: The dictionary received, expected to be in the grouped format.

        Returns:
            A validated GroupedActionOutput object if validation succeeds.
            Returns None and logs a warning if validation fails.
        """
        try:
            # Pydantic handles the validation automatically when you instantiate the model
            validated_output = GroupedActionOutput(**data)
            logger.info("Dictionary validated successfully against GroupedActionOutput.")
            return validated_output
        except ValidationError as e:
            # If validation fails, a ValidationError is raised
            logger.warning(f"Validation failed for grouped action output: {e}")
            # You might want to log the failed data for debugging:
            # logger.debug(f"Failed data: {data}")
            return None

    def execute_actions(self, grouped_output: GroupedActionOutput):
        """
        Executes actions described in a GroupedActionOutput model.
        Actions are iterated by type, then by instance parameters.

        Args:
            grouped_output: The validated GroupedActionOutput model object.

        Returns:
            True if all actions attempted were successful, False otherwise.
            Returns True if there were no actions to execute.
        """
        # Check if the actions dictionary is empty or contains no actions
        # This checks if there's any list of parameters that isn't empty
        has_actions_to_execute = any(param_list for param_list in grouped_output.actions.values())

        if not has_actions_to_execute:
            self.logger.info(f"{self.name}: No actions specified in the grouped output to execute.")
            return True # Indicate success for no actions

        self.logger.info(f"\n--- {self.name} executing grouped actions ---")
        all_succeeded = True

        # Iterate through the action types and their lists of parameters
        # .items() gives you key, value pairs (action_type, list_of_params_lists)
        for action_type, list_of_params_lists in grouped_output.actions.items():

            # Skip action types that have no instances listed
            if not list_of_params_lists:
                self.logger.info(f"No instances specified for action type: {action_type}. Skipping.")
                continue

            self.logger.info(f"Executing actions of type: {action_type}")

            # Now iterate through each list of parameters for this action type
            # Each 'params' here is a single list like ["red", 5, 10, 3]
            for i, params in enumerate(list_of_params_lists):
                self.logger.info(f"  Attempting instance {i+1}: {action_type} with params {params}")

                try:
                    # The logic for finding and calling the handler remains similar
                    if hasattr(self.action_handler, action_type):
                        handler_method = getattr(self.action_handler, action_type)
                        # Call the method, passing parameters from the current list
                        success = handler_method(*params) # Use * to unpack the params list
                        self.logger.info(f"  Instance {i+1} of '{action_type}' executed. Success: {success}")
                        if not success:
                            all_succeeded = False
                    else:
                        self.logger.error(f"  Error: Unknown action type '{action_type}' on action handler. Instance {i+1} failed.")
                        all_succeeded = False

                except TypeError as e:
                     # Catch errors specific to function arguments
                     self.logger.error(f"  Error executing action '{action_type}' instance {i+1} with params {params}: Parameter mismatch or incorrect arguments. {e}")
                     all_succeeded = False
                except Exception as e:
                    # Catch any other unexpected errors during execution
                    self.logger.error(f"  Unexpected error executing action '{action_type}' instance {i+1}: {e}")
                    all_succeeded = False

        self.logger.info(f"--- {self.name} finished executing grouped actions. Overall success: {all_succeeded} ---")
        return all_succeeded
