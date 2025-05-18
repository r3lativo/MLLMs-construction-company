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

    def add_message_to_history(self,
                               role: str,
                               text_content: Optional[str] = None,
                               image_paths: Optional[List[str]] = None, # NOW ACCEPTS A LIST OF LOCAL PATHS
                               image_urls: Optional[List[str]] = None): # NOW ACCEPTS A LIST OF EXTERNAL URLs
        """
        Adds a message (text, multiple images, or both) to the agent's history.
        Conditionally formats 'content' as a string or a list of parts based on message type.

        Args:
            role (str): The role of the message sender (e.g., "user", "assistant", "system").
            text_content (str, optional): The text part of the message. Defaults to None.
            image_paths (list[str], optional): A list of local file paths to images. Images will be Base64 encoded. Defaults to None.
            image_urls (list[str], optional): A list of external URLs to images. Defaults to None.
        """
        # Determine if this message requires multimodal content (list of parts)
        # It's multimodal if either image_paths or image_urls lists are provided and non-empty.
        is_multimodal_message = (
            (image_paths is not None and len(image_paths) > 0) or
            (image_urls is not None and len(image_urls) > 0)
        )

        if is_multimodal_message:
            # For multimodal messages (e.g., user with image), content is a list of parts
            message_content_parts = []

            # Add text content if provided
            if text_content and text_content.strip():
                message_content_parts.append({"type": "text", "text": text_content})

            # Process multiple local image paths if provided
            if image_paths:
                for img_path in image_paths:
                    image_data_uri = encode_image_to_data_uri(img_path) # Call the global helper
                    if image_data_uri:
                        message_content_parts.append({"type": "image_url", "image_url": {"url": image_data_uri}})
                        self.logger.debug(f"Encoded image from {img_path} to Base64 and added to history.")
                    else:
                        self.logger.error(f"Could not encode image from {img_path}. Skipping this image attachment.")

            # Process multiple image URLs if provided
            if image_urls:
                for img_url in image_urls:
                    if img_url and img_url.strip(): # Ensure URL is not empty/whitespace
                        message_content_parts.append({"type": "image_url", "image_url": {"url": img_url}})
                        self.logger.debug(f"Added image URL {img_url} to history.")
                    else:
                        self.logger.warning("Empty or invalid image URL provided. Skipping attachment.")

            # If no content parts were successfully added, skip the message
            if not message_content_parts:
                self.logger.warning(f"Attempted to add empty multimodal message for role {role}. No text, valid images, or valid URLs provided. Message skipped.")
                return

            self.history.append({
                "role": role,
                "content": message_content_parts # This will be a list of dictionaries (parts)
            })
        else:
            # For purely text-based messages (system, assistant, or text-only user), content is a string
            if text_content is None or not text_content.strip():
                self.logger.warning(f"Attempted to add empty text message for role {role}. Message skipped.")
                return

            self.history.append({
                "role": role,
                "content": text_content # This will be a plain string
            })

    def get_history(self) -> List[dict]:
        """Returns the agent's conversation history."""
        return self.history

    def _call_model(self, messages: List[dict]) -> Optional[str]:
        """Makes an API call to the language model."""
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": messages,
        }

        api_timeout_seconds = config.get("api_timeout_seconds", 600)

        try:
            self.logger.debug(f"Calling model '{self.model}' at {self.vllm_api_base}/chat/completions...")
            self.logger.debug(f"Payload sent to model: {json.dumps(payload, indent=2)}") # Log the full payload

            response = requests.post(
                f"{self.vllm_api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=api_timeout_seconds
            )
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            response_json = response.json()

            if 'choices' in response_json and len(response_json['choices']) > 0:
                # Assuming the model returns a simple text content for its response
                return response_json['choices'][0]['message']['content']
            else:
                self.logger.warning(f"Model response has no choices: {response_json}")
                return None

        except requests.exceptions.Timeout:
            self.logger.error(f"API call to {self.name}'s model timed out after {api_timeout_seconds} seconds.")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error during API call to {self.name}'s model: {e}")
            if hasattr(e, 'response') and e.response is not None:
                self.logger.error(f"Response status code: {e.response.status_code}")
                self.logger.error(f"Response content: {e.response.text}")
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
