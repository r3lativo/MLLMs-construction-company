import requests
import json
import logging
import base64 # Import base64 for encoding
import os     # Import os for path handling
from config import config
from pydantic import BaseModel, Field
from typing import Any


# Assume pydantic models are defined elsewhere or passed correctly
# from your_pydantic_models_file import GroupedActionOutput # Example

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

    def add_message_to_history(self, role: str, text_content: str = None, image_path: str = None, image_url: str = None):
        """
        Adds a message (text, image, or both) to the agent's history.
        Multimodal messages are stored as a list of 'parts' in the 'content' field.

        Args:
            role (str): The role of the message sender (e.g., "user", "assistant", "system").
            text_content (str, optional): The text part of the message. Defaults to None.
            image_path (str, optional): Local file path to an image. Image will be Base64 encoded. Defaults to None.
            image_url (str, optional): URL to an image. Defaults to None.
        """
        message_content = []

        if text_content:
            message_content.append({"type": "text", "text": text_content})

        if image_path:
            try:
                if not os.path.exists(image_path):
                    self.logger.error(f"Image file not found: {image_path}. Skipping image attachment.")
                else:
                    with open(image_path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    
                    # Infer MIME type from file extension
                    mime_type = "image/jpeg" # Default
                    if image_path.lower().endswith(".png"):
                        mime_type = "image/png"
                    elif image_path.lower().endswith(".gif"):
                        mime_type = "image/gif"
                    # Add more types as needed

                    image_part = {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{encoded_string}"}
                    }
                    message_content.append(image_part)
                    self.logger.debug(f"Encoded image from {image_path} to Base64 and added to history.")
            except Exception as e:
                self.logger.error(f"Error processing image {image_path}: {e}. Skipping image attachment.")
        elif image_url:
            image_part = {
                "type": "image_url",
                "image_url": {"url": image_url}
            }
            message_content.append(image_part)
            self.logger.debug(f"Added image URL {image_url} to history.")

        if not message_content:
            self.logger.warning(f"Attempted to add empty message for role {role}. No text or image provided.")
            return

        self.history.append({
            "role": role,
            "content": message_content
        })

    def get_history(self) -> list:
        """Returns the agent's conversation history."""
        return self.history

    def _call_model(self, messages: list):
        """Makes an API call to the language model."""
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": messages, # This 'messages' structure now supports multimodal content
        }

        try:
            self.logger.debug(f"Calling model '{self.model}' at {self.vllm_api_base}/chat/completions...")
            response = requests.post(
                f"{self.vllm_api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=config.get("api_timeout_seconds", 600) # Use config for timeout
            )
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            response_json = response.json()

            # Extract the content from the response
            if 'choices' in response_json and len(response_json['choices']) > 0:
                # Assuming the model returns a simple text content
                # For multimodal models, content might still be a string, or require specific parsing
                return response_json['choices'][0]['message']['content']
            else:
                self.logger.warning(f"Model response has no choices: {response_json}")
                return None

        except requests.exceptions.Timeout:
            self.logger.error(f"API call to {self.name}'s model timed out after {config.get('api_timeout_seconds', 600)} seconds.")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error during API call to {self.name}'s model: {e}")
            self.logger.error(f"Response content: {response.text}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON response from {self.name}'s model: {e}")
            return None
        except KeyError as e:
            self.logger.error(f"Unexpected JSON structure from {self.name}'s model: Missing key {e}. Response: {response_json}")
            return None
        except Exception as e:
            self.logger.error(f"An unexpected error occurred in {self.name}'s _call_model: {e}")
            return None

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
