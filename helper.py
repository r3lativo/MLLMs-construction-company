from pydantic import BaseModel, Field
from typing import Any
import logging
import openai

logger = logging.getLogger(__name__)


class Agent:
    """Base class for a virtual agent interacting with vLLM."""
    def __init__(self, name: str, model: str, system_prompt: str, vllm_api_base: str = "http://localhost:8000/v1"):
        self.name = name
        self.model = model
        self.vllm_api_base = vllm_api_base
        # History starts with system prompt. Message content can be a string or a list for multi-modal (only used by Architect initially).
        self.history = [{"role": "system", "content": system_prompt.render()}]
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


class GroupedActionOutput(BaseModel):
    """
    Represents the Builder's action output where actions are grouped by type.
    Matches the target dictionary structure:
    {
        "actions": {
            "action_type_1": [
                [param1_val1, param2_val1, ...], # parameters for instance 1 of type 1
                [param1_val2, param2_val2, ...], # parameters for instance 2 of type 1
                ...
            ],
            "action_type_2": [
                [param1_valA, ...], # parameters for instance 1 of type 2
                ...
            ]
        },
        "communication": "..."
    }
    """
    # The 'actions' field is a dictionary:
    # - Keys are strings (the action types like "place_block")
    # - Values are lists of lists (each inner list is the params for one action instance)
    actions: dict[str, list[list[Any]]] = Field(
        default_factory=dict,
        description="Dictionary where keys are action types and values are lists of parameter lists."
    )

    # The 'communication' field is similar to the old model
    communication: str = Field(
        default="",
        description="Optional message to the Architect after performing actions."
    )
