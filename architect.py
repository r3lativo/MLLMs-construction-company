from helper import Agent # Ensure this import is correct
from typing import Any, List, Optional

class Architect(Agent):
    """
    Represents the Architect agent, responsible for initiating the project,
    interpreting the Builder's progress and world state, and issuing
    subsequent instructions.
    """

    def __init__(self, model: str, system_prompt: str, init_prompt: str,
                 structure_description_json: dict, structure_image_paths: Optional[List[str]] = None,
                 vllm_api_base: str = "http://localhost:8000/v1"):
        """
        Initializes the Architect agent.

        Args:
            model (str): The name of the language model to use (e.g., 'gemini-pro-vision').
            system_prompt (str): The system prompt template for guiding the Architect's behavior.
            init_prompt (str): The initial prompt template to kick off the building project.
            structure_description_json (dict): The target structure's blueprint in JSON format.
            structure_image_paths (list[str], optional): A list of local file paths to reference images
                                                        of the target structure. Defaults to None.
            vllm_api_base (str): The base URL for the vLLM API endpoint.
        """
        super().__init__("Architect", model, system_prompt, vllm_api_base)
        self.init_prompt = init_prompt
        self.structure_description_json = structure_description_json
        self.structure_image_paths = structure_image_paths


    def start_project(self) -> Optional[str]:
        """
        Initiates the building project by sending the first set of instructions
        to the Builder. This involves rendering an initial prompt that may include
        both textual (JSON) and visual (image) descriptions of the target structure.

        Returns:
            str | None: The Architect's initial instruction message to the Builder,
                        or None if the model fails to generate a response.
        """
        self.logger.info("Architect is preparing initial project instructions.")

        use_json = bool(self.structure_description_json)
        use_img = bool(self.structure_image_paths)

        # Render the initial prompt using the structure description and indicating input types
        init_message_content = self.init_prompt.render(
            use_json=use_json,
            use_img=use_img,
            json_description=self.structure_description_json
        )
        
        # Add the initial message to the Architect's conversation history.
        # Images are included if available.
        if use_img:
            self.add_message_to_history("user", init_message_content, image_paths=self.structure_image_paths)
            self.logger.info("Architect sent initial instruction with text and reference images.")
        else:
            self.add_message_to_history("user", init_message_content)
            self.logger.info("Architect sent initial instruction with text only.")

        # Call the language model to generate the Architect's first response
        response = self._call_model(self.history)
        
        if response is None:
            self.logger.error("Architect failed to generate an initial project instruction from the model.")
            return None
        
        # Add the model's response to the Architect's history
        self.add_message_to_history("assistant", response)
        
        self.logger.info("Architect's first instruction generated (%.100s...)", response) # Log snippet of response
        return response

    def process_builder_output(self, builder_communication: str) -> Optional[str]:
        """
        Processes the Builder's textual communication and the current world state feedback.
        Based on this information, the Architect generates the next set of instructions
        for the Builder.

        Args:
            builder_communication (str): The textual message received from the Builder
                                         (e.g., status updates, questions).

        Returns:
            str | None: The Architect's next instruction message to the Builder,
                        or None if the model fails to generate a response.
        """
        self.logger.info("Architect is processing Builder's message and world state.")
        
        # Add the Builder's communication to the Architect's history as a user message.
        # World state feedback (images/JSON) is added via `add_world_state_to_history` before this call.
        self.add_message_to_history("user", f"Builder's message:\n{builder_communication}")

        response = self._call_model(self.history)

        if response is None:
            self.logger.error("Architect failed to generate a follow-up instruction for the Builder.")
            return None
        
        self.add_message_to_history("assistant", response)
        self.logger.info("Architect's next instruction generated (%.100s...)", response) # Log snippet of response
        return response

    def add_world_state_to_history(self, world_state_description: str, generated_image_paths: List[str]):
        """
        Adds a description of the current world state, potentially including generated images,
        to the Architect's conversation history. This feedback allows the Architect to
        monitor progress and adjust instructions accordingly.

        Args:
            world_state_description (str): A textual description of the world's current state.
            generated_image_paths (List[str]): A list of local file paths to images
                                                representing the current world state.
        """
        # The world state feedback is added as a 'user' message, as it's input to the Architect.
        self.add_message_to_history("user", world_state_description, image_paths=generated_image_paths)
        if generated_image_paths:
            self.logger.info("Architect received world state feedback with images.")
        else:
            self.logger.info("Architect received world state feedback (text-only).")
