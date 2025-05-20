from helper import Agent # Ensure this import is correct
from typing import Any, List, Optional

class Architect(Agent):
    """Represents the Architect agent."""

    def __init__(self, model: str, system_prompt: str, init_prompt: str,
                 structure_description_json: dict, structure_image_paths: Optional[List[str]] = None,
                 vllm_api_base: str = "http://localhost:8000/v1"):
        """
        Args:
            model (str): The name of the language model to use.
            system_prompt (str): The system prompt template for the Architect.
            init_prompt (str): The initial prompt template for the Architect to start the project.
            structure_description_json (dict): The target structure in JSON format.
            structure_image_paths (list[str], optional): A list of local file paths to images. Defaults to None.
            vllm_api_base (str): The base URL for the vLLM API.
        """
        super().__init__("Architect", model, system_prompt, vllm_api_base)
        self.init_prompt = init_prompt
        self.structure_description_json = structure_description_json
        self.structure_image_paths = structure_image_paths
        self.logger.info("Architect initialized.")


    def start_project(self) -> str:
        """
        Initiates the project by instructing the Builder based on the structure description.
        This is the Architect's first message to the Builder.
        """
        self.logger.info("\n--- Architect is initiating the project ---")

        # Determine use_json and use_img status
        use_json = bool(self.structure_description_json)
        use_img = bool(self.structure_image_paths)

        # Render the initial prompt using the structure description
        init_message_content = self.init_prompt.render(
            use_json=use_json,
            use_img=use_img,
            json_description=self.structure_description_json
        )
        
        # Add the initial message to Architect's history.
        # Now supporting image_path for local files
        if use_img:
            self.add_message_to_history("user", init_message_content, image_paths=self.structure_image_paths)
            self.logger.debug(f"Architect added initial instruction with image from path: {self.structure_image_paths}")
        else:
            self.add_message_to_history("user", init_message_content)
            self.logger.debug("Architect added initial instruction (text-only).")

        # Call the model to get the Architect's first response to the Builder
        response = self._call_model(self.history)
        
        if response is None:
            self.logger.error("Architect failed to generate an initial project instruction.")
            return None
        
        # Add the model's response to the Architect's history
        self.add_message_to_history("assistant", response)
        
        self.logger.info(f"Architect's first instruction generated: {response[:200]}...")
        return response

    def process_builder_output(self, builder_communication: str): #, world_state_feedback: str = None):
        """
        Processes the Builder's communication and world state feedback,
        then generates the next instruction for the Builder.

        Args:
            builder_communication (str): The textual communication from the Builder.
            world_state_feedback (str, optional): A description of the current world state. Defaults to None.

        Returns:
            str | None: The Architect's next instruction to the Builder, or None if generation fails.
        """
        self.logger.debug("\n--- Architect is processing Builder's output and world state ---")
        
        # Combine Builder's communication and world state feedback into a single user message
        #user_message_content = f"Builder's message:\n{builder_communication}"
        #if world_state_feedback:
        #    user_message_content += f"\n\nCurrent World State:\n{world_state_feedback}"
        
        #self.add_message_to_history("user", user_message_content)
        self.add_message_to_history("user", f"Builder's message:\n{builder_communication}")
        #self.logger.debug("Architect processing: Added combined Builder comms/world state to history.")

        response = self._call_model(self.history)

        if response is None:
            self.logger.error("Architect failed to generate a response to Builder's output.")
            return None
        
        self.add_message_to_history("assistant", response)
        self.logger.debug(f"Architect's next instruction generated: {response[:200]}...")
        return response

    def add_world_state_to_history(self, world_state_description, generated_image_paths):

        self.add_message_to_history("user", world_state_description, generated_image_paths)

        return