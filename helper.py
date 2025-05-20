import requests
import json
import logging
import base64 # Import base64 for encoding
import os     # Import os for path handling
from config import config
from pydantic import BaseModel, Field
from typing import Any, List, Optional

# Assume pydantic models are defined elsewhere or passed correctly
# from your_pydantic_models_file import GroupedActionOutput # Example

def encode_image_to_data_uri(image_path: str):
    """
    Reads an image file, Base64 encodes it, and returns a data URI.
    Returns None if the file is not found or encoding fails.
    """
    if not os.path.exists(image_path):
        logging.error(f"Image file not found at {image_path}")
        return None

    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        mime_type = "image/jpeg"
        if image_path.lower().endswith(".png"): mime_type = "image/png"
        elif image_path.lower().endswith(".gif"): mime_type = "image/gif"
        elif image_path.lower().endswith(".webp"): mime_type = "image/webp"

        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        logging.error(f"Error encoding image {image_path}: {e}")
        return None


class Agent:
    """Base class for Architect and Builder agents."""

    def __init__(self, name: str, model: str, system_prompt: str, vllm_api_base: str = "http://localhost:8000/v1"):
        self.name = name
        self.model = model
        self.vllm_api_base = vllm_api_base
        self.logger = logging.getLogger(self.__class__.__name__)
        self.history = []
        
        # Initialize with system prompt
        # The content of the system prompt is always text
        self.add_message_to_history("system", system_prompt.render())
        self.logger.info(f"{self.name} initialized with model '{self.model}' and system prompt.")

    def add_message_to_history(self, role, text_content=None, image_paths=None, image_urls=None):
        """
        Adds a message (text, multiple images, or both) to the agent's history.
        Conditionally formats 'content' as a string or a list of parts based on message type.

        Args:
            role: The role of the message sender (e.g., "user", "assistant", "system").
            text_content: The text part of the message. Defaults to None.
            image_paths: A list of local file paths to images. Images will be Base64 encoded. Defaults to None.
            image_urls: A list of external URLs to images. Defaults to None.
        """
        # For multimodal messages (e.g., user with image), content is a list of parts
        content = []

        if text_content and text_content.strip():
            content.append({"type": "text", "text": text_content})
        if image_paths:
            for p in image_paths:
                    content.append({"type": "image_url", "image_url": {"url": encode_image_to_data_uri(p)}})
        if image_urls:
            for u in image_urls:
                content.append({"type": "image_url", "image_url": {"url": u}})

        if not content:  # If nothing was added, don't add an empty message
            return
                
        # Simple string content for system/assistant if no images
        if not (image_paths or image_urls) and len(content) == 1 and content[0]["type"] == "text":
            self.history.append({"role": role, "content": content[0]["text"]})
        else:
            self.history.append({"role": role, "content": content})


    def get_history(self) -> List[dict]:
        """Returns the agent's conversation history."""
        return self.history

    def _call_model(self, messages: List[dict]) -> Optional[str]:
        """Makes an API call to the language model."""
        headers = {"Content-Type": "application/json"}
        payload = {"model": self.model, "messages": messages }

        api_timeout_seconds = config.get("api_timeout_seconds", 600)

        try:
            self.logger.debug(f"Payload sent to '{self.model}': {json.dumps(payload, indent=2)}") # Log the full payload

            response = requests.post(
                f"{self.vllm_api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=api_timeout_seconds
            )
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            response_json = response.json()
            return response_json['choices'][0]['message']['content']

        except Exception as e:
            self.logger.error(f"An unexpected error occurred in {self.name}'s _call_model: {e}")
            return None


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
