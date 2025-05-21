import requests
import json
import logging
import base64
import os
from config import config
from pydantic import BaseModel, Field
from typing import Any, List, Optional

def encode_image_to_data_uri(image_path: str) -> Optional[str]:
    """
    Reads an image file, Base64 encodes it, and returns a data URI.
    This format is suitable for embedding images directly into JSON payloads for APIs.

    Args:
        image_path (str): The local file path to the image.

    Returns:
        str | None: A data URI string (e.g., "data:image/png;base64,...") if successful,
                    otherwise None if the file is not found or an error occurs during encoding.
    """
    if not os.path.exists(image_path):
        logging.error("Image file not found at '%s'.", image_path)
        return None

    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        # Determine MIME type based on file extension
        mime_type = "image/jpeg"
        if image_path.lower().endswith(".png"): mime_type = "image/png"
        elif image_path.lower().endswith(".gif"): mime_type = "image/gif"
        elif image_path.lower().endswith(".webp"): mime_type = "image/webp"

        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        logging.error("Error encoding image '%s': %s", image_path, e)
        return None


class Agent:
    """
    Base class for all agents (e.g., Architect, Builder) in the simulation.
    Manages common functionalities like conversation history, logging, and API calls to the language model.
    """

    def __init__(self, name: str, model: str, system_prompt: str, vllm_api_base: str = "http://localhost:8000/v1"):
        """
        Initializes the base Agent.

        Args:
            name (str): The name of the agent (e.g., "Architect", "Builder").
            model (str): The identifier of the language model to use (e.g., "gemini-pro-vision").
            system_prompt (str): The initial system prompt that defines the agent's role and constraints.
                                 This is rendered and added to history upon initialization.
            vllm_api_base (str): The base URL for the vLLM API server.
        """
        self.name = name
        self.model = model
        self.vllm_api_base = vllm_api_base
        self.logger = logging.getLogger(self.__class__.__name__)
        self.history = []
        
        # Add the initial system prompt to the conversation history.
        # System prompts are always text-only.
        self.add_message_to_history("system", system_prompt.render())
        self.logger.info("%s initialized with model '%s'.", self.name, self.model)

    def add_message_to_history(self, role: str, text_content: Optional[str] = None, 
                               image_paths: Optional[List[str]] = None, 
                               image_urls: Optional[List[str]] = None):
        """
        Adds a message to the agent's conversation history. Messages can contain
        text, local image paths (which are Base64 encoded), or external image URLs.
        The content is formatted according to the model's expected message structure.

        Args:
            role (str): The role of the message sender (e.g., "user", "assistant", "system").
            text_content (str, optional): The textual part of the message. Defaults to None.
            image_paths (list[str], optional): A list of local file paths to images to be included.
                                                These will be converted to data URIs. Defaults to None.
            image_urls (list[str], optional): A list of external URLs to images to be included. Defaults to None.
        """
        content_parts = []

        if text_content and text_content.strip():
            content_parts.append({"type": "text", "text": text_content})
        
        if image_paths:
            for path in image_paths:
                data_uri = encode_image_to_data_uri(path)
                if data_uri:
                    content_parts.append({"type": "image_url", "image_url": {"url": data_uri}})
                else:
                    self.logger.warning("Failed to encode image from path: '%s'. Skipping.", path)

        if image_urls:
            for url in image_urls:
                content_parts.append({"type": "image_url", "image_url": {"url": url}})

        # If no content was successfully added, do not append an empty message.
        if not content_parts:
            self.logger.debug("Attempted to add an empty message to history for role '%s'. Skipping.", role)
            return
                
        # For simple text-only messages (e.g., system or assistant messages without images),
        # use a string directly instead of a list of parts for 'content'.
        if len(content_parts) == 1 and content_parts[0]["type"] == "text" and role != "user":
            self.history.append({"role": role, "content": content_parts[0]["text"]})
        else:
            self.history.append({"role": role, "content": content_parts})
        
        self.logger.debug("Message added to %s's history (role: %s, text_len: %d, images: %d).",
                          self.name, role, len(text_content) if text_content else 0,
                          len(image_paths if image_paths else []) + len(image_urls if image_urls else []))


    def get_history(self) -> List[dict]:
        """
        Returns the agent's complete conversation history.

        Returns:
            list[dict]: A list of message dictionaries.
        """
        return self.history

    def _call_model(self, messages: List[dict]) -> Optional[str]:
        """
        Makes an API call to the configured language model with the given messages.

        Args:
            messages (list[dict]): The list of messages to send to the model, in the expected API format.

        Returns:
            str | None: The content of the model's response if successful, otherwise None.
        """
        headers = {"Content-Type": "application/json"}
        payload = {"model": self.model, "messages": messages}

        # Retrieve API timeout from configuration, defaulting to 600 seconds.
        api_timeout_seconds = config.get("api_timeout_seconds", 600)

        try:
            self.logger.debug("Sending request to model '%s'. Payload size: %d bytes.",
                              self.model, len(json.dumps(payload)))

            response = requests.post(
                f"{self.vllm_api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=api_timeout_seconds
            )
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            response_json = response.json()
            
            # Extract and return the content from the first choice in the response.
            model_content = response_json['choices'][0]['message']['content']
            self.logger.info("Successfully received response from model '%s'.", self.model)
            return model_content

        except requests.exceptions.Timeout:
            self.logger.error("API call to model '%s' timed out after %d seconds.", self.model, api_timeout_seconds)
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error("HTTP request error to model '%s': %s", self.model, e)
            return None
        except KeyError:
            self.logger.error("Unexpected response structure from model '%s'. Missing 'choices' or 'message' content.", self.model)
            self.logger.debug("Full API response: %s", response.text) # Log full response for debugging malformed JSON
            return None
        except Exception as e:
            self.logger.exception("An unexpected error occurred in %s's _call_model during API interaction: %s", self.name, e)
            return None


class GroupedActionOutput(BaseModel):
    """
    Pydantic model representing the Builder's structured output, where building
    actions are grouped by their type (e.g., 'place_block', 'remove_block').
    It also includes an optional communication message back to the Architect.

    Expected JSON structure:
    ```json
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
        "communication": "Optional text message from Builder to Architect."
    }
    ```
    """
    # The 'actions' field is a dictionary where:
    # - Keys are strings representing the action types (e.g., "place_block", "remove_block").
    # - Values are lists of lists, where each inner list contains the parameters for one
    #   instance of that action type (e.g., `["red", 0, 0, 0]` for `place_block`).
    actions: dict[str, list[list[Any]]] = Field(
        default_factory=dict,
        description="A dictionary where keys are action types and values are lists of parameter lists for each action instance."
    )

    # The 'communication' field is an optional string that allows the Builder to send
    # a textual message back to the Architect, e.g., for status updates or questions.
    communication: str = Field(
        default="",
        description="An optional textual message from the Builder to the Architect after processing instructions and actions."
    )
