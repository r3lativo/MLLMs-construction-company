from helper import Agent
import json
from helper import GroupedActionOutput
from pydantic import ValidationError


class Builder(Agent):
    """Represents the Builder agent."""
    # Define the tags for the JSON block within the text response
    JSON_BLOCK_START = "```json"
    JSON_BLOCK_END   = "```"

    def __init__(self, model: str, system_prompt: str, action_handler, vllm_api_base: str = "http://localhost:8000/v1"):
        """
        Args:
            model (str): The name of the language model to use.
            system_prompt (str): The system prompt for the Builder.
            action_handler: An object with methods like place_block, remove_block, and get_block.
            vllm_api_base (str): The base URL for the vLLM API.
        """
        super().__init__("Builder", model, system_prompt, vllm_api_base)

        self.inventory = { "blue": 20, "yellow": 20, "green": 20, "orange": 20, "purple": 20, "red": 20}
        self.action_handler = action_handler
        self.logger.info(f"Builder initialized with inventory: {self.inventory}")

    def process_architect_instruction(self, architect_message_text: str):
        """
        Receives instruction (text) from Architect, generates response (JSON actions + optional text).
        Returns raw model output (text), parsed communication text, and a GroupedActionOutput object.
        """
        # Add Architect's text message to Builder's history
        self.add_message_to_history("user", architect_message_text)

        self.logger.debug(f"\n--- {self.name} is generating response and potential actions ---")
        
        raw_model_output = self._call_model(self.history)
        self.logger.debug(f"Builder raw output: {raw_model_output}")

        if raw_model_output is None:
            self.logger.error(f"{self.name}: Failed to generate raw output.")
            # Return an empty GroupedActionOutput if generation fails
            return None, "Failed to generate response.", GroupedActionOutput(actions={"place_block": [], "remove_block": []}, communication="Failed to generate response.")

        # Add the raw model output to the Builder's history as assistant's turn
        self.add_message_to_history("assistant", raw_model_output)

        # Parse the raw output for structured communication and actions
        communication_text, grouped_actions_output = self._parse_output(raw_model_output)

        self.logger.debug(f"{self.name}: Parsed communication: {communication_text[:100]}...")
        self.logger.debug(f"{self.name}: Parsed actions: {grouped_actions_output}")

        return raw_model_output, communication_text, grouped_actions_output

    def _parse_output(self, raw_output: str) -> tuple[str, GroupedActionOutput]:
        """Parses the raw model output for communication text and action details from a JSON block."""

        # Attempt to find and parse the JSON block
        json_start_index = raw_output.find(self.JSON_BLOCK_START)
        json_end_index = raw_output.find(self.JSON_BLOCK_END, json_start_index + len(self.JSON_BLOCK_START))

        # Default empty GroupedActionOutput
        default_actions_output = GroupedActionOutput(actions={"place_block": [], "remove_block": []}, communication="")
        communication = raw_output # Default to raw output as communication

        if json_start_index != -1 and json_end_index != -1:
            json_string = raw_output[json_start_index + len(self.JSON_BLOCK_START):json_end_index].strip()
            
            # Clean up potential markdown code block indicator (e.g., 'json\n{...}')
            if json_string.startswith("json\n"):
                json_string = json_string[len("json\n"):].strip()

            try:
                parsed_data = json.loads(json_string)
                validated_output = self._validate_and_parse_grouped_actions(parsed_data)
                
                if validated_output:
                    default_actions_output = validated_output
                    communication = validated_output.communication if validated_output.communication is not None else ""
                else: 
                    # Validation failed, use raw output as communication and empty actions
                    communication = raw_output 

            except json.JSONDecodeError as e:
                self.logger.error(f"Error decoding JSON from Builder output: {e}. Treating as communication only.")
                communication = raw_output
            except Exception as e:
                self.logger.error(f"Unexpected error parsing Builder output: {e}. Treating as communication only.")
                communication = raw_output

        else:
            self.logger.info("No JSON block found in Builder output. Treating as communication only.")
            communication = raw_output

        return communication, default_actions_output

    def _validate_and_parse_grouped_actions(self, data: dict):
        """
        Validates a dictionary against the GroupedActionOutput Pydantic model.

        Args:
            data: The dictionary received, expected to be in the grouped format.

        Returns:
            A validated GroupedActionOutput object if validation succeeds, None otherwise.
        """
        try:
            validated_output = GroupedActionOutput(**data)
            self.logger.debug("Dictionary validated successfully against GroupedActionOutput.")
            return validated_output
        except ValidationError as e:
            self.logger.warning(f"Validation failed for grouped action output: {e}")
            self.logger.debug(f"Failed data: {data}")
            return None

    def _execute_single_action(self, action_type: str, params: list, instance_index: int) -> bool:
        """
        Executes a single building action (place_block or remove_block) and manages inventory.
        Returns True on success, False otherwise.
        """
        self.logger.debug(f"  Attempting instance {instance_index}: {action_type} with params {params}")
        success = False

        if action_type == "place_block":
            if len(params) == 4: # Expected: ["color", x, y, z]
                color, x, y, z = params
                if color in self.inventory and self.inventory[color] > 0:
                    self.logger.info(f"  Current inventory of {color}: {self.inventory[color]}. Attempting to place at ({x},{y},{z}).")
                    if hasattr(self.action_handler, "place_block"):
                        success = self.action_handler.place_block(color, x, y, z)
                        if success:
                            self.inventory[color] -= 1
                            self.logger.debug(f"  Successfully placed {color} block. Remaining: {self.inventory[color]}")
                        else:
                            self.logger.warning(f"  Action handler failed to place {color} block at ({x},{y},{z}).")
                else:
                    self.logger.error(f"  Cannot place {color} block: Insufficient inventory or invalid color '{color}'. Current {color} count: {self.inventory.get(color, 0)}")
            else:
                self.logger.error(f"  Invalid parameters for place_block: {params}. Expected [color, x, y, z].")
        
        elif action_type == "remove_block":
            if len(params) == 3: # Expected: [x, y, z]
                x, y, z = params
                if hasattr(self.action_handler, "get_block"):
                    # Assuming get_block returns the color of the block at x,y,z or None if no block
                    removed_block_color = self.action_handler.get_block(x, y, z)
                    if removed_block_color: # Block exists at location
                        self.logger.info(f"  Attempting to remove {removed_block_color} block at ({x},{y},{z}).")
                        if hasattr(self.action_handler, "remove_block"):
                            success = self.action_handler.remove_block(x, y, z)
                            if success:
                                # Add removed block back to inventory
                                if removed_block_color in self.inventory:
                                    self.inventory[removed_block_color] += 1
                                    self.logger.debug(f"  Successfully removed block. Added {removed_block_color} back to inventory. New count: {self.inventory[removed_block_color]}")
                            else:
                                self.logger.warning(f"  Action handler failed to remove block at ({x},{y},{z}).")
            else:
                self.logger.error(f"  Invalid parameters for remove_block: {params}. Expected [x, y, z].")

        else:
            self.logger.error(f"  Error: Unknown action type '{action_type}' received from model.")

        return success


    def execute_actions(self, grouped_output: GroupedActionOutput) -> bool:
        """
        Executes actions described in a GroupedActionOutput model, managing inventory.
        Actions are iterated by type, then by instance parameters.

        Args:
            grouped_output: The validated GroupedActionOutput model object.

        Returns:
            True if all actions attempted were successful, False otherwise.
            Returns True if there were no actions to execute.
        """
        # Ensure grouped_output is valid before proceeding
        if not grouped_output:
            self.logger.error(f"{self.name}: Invalid grouped_output received. Cannot execute actions.")
            return False

        has_actions_to_execute = any(param_list for param_list in grouped_output.actions.values())

        if not has_actions_to_execute:
            self.logger.debug(f"{self.name}: No actions specified in the grouped output to execute.")
            return True # Indicate success for no actions

        self.logger.debug(f"\n--- {self.name} executing grouped actions. Current inventory: {self.inventory} ---")
        all_succeeded = True

        for action_type, list_of_params_lists in grouped_output.actions.items():
            if not list_of_params_lists:
                self.logger.debug(f"No instances specified for action type: {action_type}. Skipping.")
                continue

            self.logger.debug(f"Executing actions of type: {action_type}")

            for i, params in enumerate(list_of_params_lists):
                try:
                    success = self._execute_single_action(action_type, params, i + 1)
                    if not success:
                        all_succeeded = False
                except TypeError as e:
                    self.logger.error(f"  Error executing action '{action_type}' instance {i+1} with params {params}: Parameter mismatch or incorrect arguments. {e}")
                    all_succeeded = False
                except Exception as e:
                    self.logger.error(f"  Unexpected error executing action '{action_type}' instance {i+1}: {e}")
                    all_succeeded = False

        self.logger.info(f"--- {self.name} finished executing grouped actions. Current inventory: {self.inventory}. Overall success: {all_succeeded} ---")
        return all_succeeded
