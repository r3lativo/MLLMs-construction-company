from helper import Agent
import json
from helper import GroupedActionOutput
from pydantic import ValidationError
from typing import Any, List, Optional


class Builder(Agent):
    """
    Represents the Builder agent, responsible for interpreting Architect's instructions,
    executing building actions (like placing or removing blocks), and managing its inventory.
    """
    # Define the tags for identifying the JSON block within the model's text response
    JSON_BLOCK_START = "```json"
    JSON_BLOCK_END   = "```"

    def __init__(self, model: str, system_prompt: str, action_handler, vllm_api_base: str = "http://localhost:8000/v1"):
        """
        Initializes the Builder agent.

        Args:
            model (str): The name of the language model to use (e.g., 'gemini-pro').
            system_prompt (str): The system prompt for guiding the Builder's behavior.
            action_handler: An object that provides concrete methods for world interactions
                            (e.g., `place_block`, `remove_block`, `get_block` from the World class).
            vllm_api_base (str): The base URL for the vLLM API endpoint.
        """
        super().__init__("Builder", model, system_prompt, vllm_api_base)

        # Initialize the Builder's inventory with a predefined quantity of blocks per color.
        self.inventory = {"blue": 20, "yellow": 20, "green": 20, "orange": 20, "purple": 20, "red": 20}
        self.action_handler = action_handler

    def process_architect_instruction(self, architect_message_text: str):
        """
        Processes an instruction received from the Architect, including current inventory information.
        1. Combine the Architect's instruction with the Builder's current inventory.
        2. Add this combined message to the Builder's conversation history.
        3. Call the language model to generate a response.
        4. Parse the raw model output to separate communicative text from structured actions.

        Args:
            architect_message_text (str): The textual instruction from the Architect.

        Returns:
            tuple[str, str, GroupedActionOutput]: A tuple containing:
                - raw_model_output (str): The complete, unparsed text response from the model.
                - communication_text (str): The parsed human-readable communication from the Builder.
                - grouped_actions_output (GroupedActionOutput): An object containing validated actions
                                                                extracted from the model's response.
        """
        self.logger.info("Builder is processing Architect's instruction and checking inventory.")

        # Prepare the inventory string
        inventory_str = str(self.inventory)
        
        # Combine Architect's message directly with inventory information
        combined_message_content = (
            f"{architect_message_text}\n\n"
            f"Your current inventory: {inventory_str}"
        )

        # Add the combined message to the Builder's history
        self.add_message_to_history("user", combined_message_content)
        self.logger.debug("Builder received combined instruction with inventory: %s", combined_message_content)
        
        # Call the language model to get the Builder's raw output.
        raw_model_output = self._call_model(self.history)

        if raw_model_output is None:
            self.logger.error("Builder failed to generate a response from the model.")
            # Return empty/default values if model generation fails
            return None, "Failed to generate response.", GroupedActionOutput(actions={}, communication="Failed to generate response.")

        # Add the raw output to the Builder's history as its own response.
        self.add_message_to_history("assistant", raw_model_output)

        # Parse the raw output to extract communication and structured actions.
        communication_text, grouped_actions_output = self._parse_output(raw_model_output)

        self.logger.info("Builder processed instruction: Communication (%.100s...), Actions (%d types).",
                         communication_text, len(grouped_actions_output.actions))

        return raw_model_output, communication_text, grouped_actions_output

    def _parse_output(self, raw_output: str) -> tuple[str, GroupedActionOutput]:
        """
        Parses the raw text output from the language model to extract a JSON block
        containing structured actions and any accompanying human-readable communication.

        Args:
            raw_output (str): The complete text response from the language model.

        Returns:
            tuple[str, GroupedActionOutput]: A tuple containing:
                - communication (str): The extracted communication text.
                - default_actions_output (GroupedActionOutput): The validated actions object.
        """
        # Attempt to find the JSON block delimited by `JSON_BLOCK_START` and `JSON_BLOCK_END`.
        json_start_index = raw_output.find(self.JSON_BLOCK_START)
        json_end_index = raw_output.find(self.JSON_BLOCK_END, json_start_index + len(self.JSON_BLOCK_START))

        # Default to an empty GroupedActionOutput and the raw output as communication
        default_actions_output = GroupedActionOutput(actions={}, communication="")
        communication = raw_output 

        if json_start_index != -1 and json_end_index != -1:
            # Extract the JSON string and attempt to clean it from potential markdown `json\n` prefix.
            json_string = raw_output[json_start_index + len(self.JSON_BLOCK_START):json_end_index].strip()
            if json_string.startswith("json\n"):
                json_string = json_string[len("json\n"):].strip()

            try:
                # Parse the JSON string and validate it against the Pydantic model.
                parsed_data = json.loads(json_string)
                validated_output = self._validate_and_parse_grouped_actions(parsed_data)
                
                if validated_output:
                    default_actions_output = validated_output
                    # Use the communication field from the validated JSON if present, otherwise default to empty string.
                    communication = validated_output.communication if validated_output.communication is not None else ""
                else: 
                    # If validation fails, treat the entire raw output as communication.
                    communication = raw_output 

            except json.JSONDecodeError as e:
                self.logger.warning("Error decoding JSON from Builder output: %s. Treating as communication only.", e)
                communication = raw_output
            except Exception as e:
                self.logger.error("Unexpected error parsing Builder output: %s. Treating as communication only.", e)
                communication = raw_output
        else:
            self.logger.info("No structured JSON block found in Builder output. Treating as communication only.")
            # If no JSON block is found, the entire raw output is considered communication.
            communication = raw_output

        return communication, default_actions_output

    def _validate_and_parse_grouped_actions(self, data: dict) -> Optional[GroupedActionOutput]:
        """
        Validates a dictionary against the `GroupedActionOutput` Pydantic model.

        Args:
            data (dict): The dictionary to validate, expected to conform to the grouped action format.

        Returns:
            GroupedActionOutput | None: A validated `GroupedActionOutput` object if validation succeeds,
                                        otherwise None.
        """
        try:
            validated_output = GroupedActionOutput(**data)
            self.logger.debug("Successfully validated grouped action output.")
            return validated_output
        except ValidationError as e:
            self.logger.warning("Validation failed for grouped action output: %s", e)
            self.logger.debug("Failed data: %s", data) # Keep detailed failed data at debug level
            return None

    def _execute_single_action(self, action_type: str, params: list, instance_index: int) -> bool:
        """
        Executes a single building action (e.g., 'place_block', 'remove_block')
        and manages the Builder's inventory accordingly.

        Args:
            action_type (str): The type of action to execute (e.g., "place_block").
            params (list): A list of parameters required for the action (e.g., [color, x, y, z]).
            instance_index (int): The index of this action instance (for logging purposes).

        Returns:
            bool: True if the action was successfully executed, False otherwise.
        """
        self.logger.debug("Attempting action #%d: %s with params %s", instance_index, action_type, params)
        success = False

        if action_type == "place_block":
            if len(params) == 4: # Expected: ["color", x, y, z]
                color, x, y, z = params
                if color in self.inventory and self.inventory[color] > 0:
                    self.logger.debug("Inventory check: %s blocks remaining for %s.", self.inventory[color], color)
                    if hasattr(self.action_handler, "place_block"):
                        success = self.action_handler.place_block(color, x, y, z)
                        if success:
                            self.inventory[color] -= 1
                            self.logger.info("Placed %s block at (%d,%d,%d). Inventory: %s remaining.", color, x, y, z, self.inventory[color])
                        else:
                            self.logger.warning("Action handler failed to place %s block at (%d,%d,%d).", color, x, y, z)
                    else:
                        self.logger.error("Action handler does not have 'place_block' method.")
                else:
                    self.logger.warning("Cannot place %s block: Insufficient inventory or invalid color '%s'. Current count: %s.",
                                       color, color, self.inventory.get(color, 0))
            else:
                self.logger.error("Invalid parameters for 'place_block': %s. Expected [color, x, y, z].", params)
        
        elif action_type == "remove_block":
            if len(params) == 3: # Expected: [x, y, z]
                x, y, z = params
                if hasattr(self.action_handler, "get_block"):
                    removed_block_color = self.action_handler.get_block(x, y, z)
                    if removed_block_color:
                        self.logger.debug("Found %s block at (%d,%d,%d) to remove.", removed_block_color, x, y, z)
                        if hasattr(self.action_handler, "remove_block"):
                            success = self.action_handler.remove_block(x, y, z)
                            if success:
                                # Return removed block to inventory if its color is recognized
                                if removed_block_color in self.inventory:
                                    self.inventory[removed_block_color] += 1
                                    self.logger.info("Removed %s block at (%d,%d,%d). Inventory: %s returned.", removed_block_color, x, y, z, self.inventory[removed_block_color])
                                else:
                                    self.logger.info("Removed unidentifiable block at (%d,%d,%d). Not added to inventory.", x, y, z)
                            else:
                                self.logger.warning("Action handler failed to remove block at (%d,%d,%d).", x, y, z)
                        else:
                            self.logger.error("Action handler does not have 'remove_block' method.")
                    else:
                        self.logger.info("No block found at (%d,%d,%d) to remove.", x, y, z)
            else:
                self.logger.error("Invalid parameters for 'remove_block': %s. Expected [x, y, z].", params)

        else:
            self.logger.error("Unknown action type '%s' received from model.", action_type)

        return success

    def execute_actions(self, grouped_output: GroupedActionOutput) -> bool:
        """
        Iterates through and executes all actions contained within a `GroupedActionOutput` object.
        This method ensures actions are performed in a structured manner, managing the Builder's
        inventory and logging the outcome of each action.

        Args:
            grouped_output (GroupedActionOutput): The validated object containing actions to be executed.

        Returns:
            bool: True if all attempted actions were successful, False if any action failed.
                  Returns True if there were no actions specified to execute.
        """
        if not grouped_output:
            self.logger.error("Invalid grouped_output received. Cannot execute actions.")
            return False

        # Check if there are any actions at all in the grouped output
        has_actions_to_execute = any(grouped_output.actions.values())

        if not has_actions_to_execute:
            self.logger.info("No actions specified in the grouped output to execute.")
            return True # Successfully "executed" no actions

        self.logger.info("Builder is executing grouped actions. Current inventory: %s", self.inventory)
        all_succeeded = True

        for action_type, list_of_params_lists in grouped_output.actions.items():
            if not list_of_params_lists:
                self.logger.debug("No instances specified for action type: %s. Skipping.", action_type)
                continue

            self.logger.info("Executing %d instances of action type: %s", len(list_of_params_lists), action_type)

            for i, params in enumerate(list_of_params_lists):
                try:
                    success = self._execute_single_action(action_type, params, i + 1)
                    if not success:
                        all_succeeded = False # Mark overall failure if any single action fails
                except TypeError as e:
                    self.logger.error("Error executing action '%s' instance %d with params %s: Parameter mismatch or incorrect arguments. %s",
                                      action_type, i + 1, params, e)
                    all_succeeded = False
                except Exception as e:
                    self.logger.error("Unexpected error executing action '%s' instance %d with params %s: %s",
                                      action_type, i + 1, params, e)
                    all_succeeded = False

        self.logger.info("Builder finished executing actions. Overall success: %s. Final inventory: %s",
                         all_succeeded, self.inventory)
        return all_succeeded
