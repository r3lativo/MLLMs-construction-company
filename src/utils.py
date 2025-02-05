import datetime
import logging
import os
import torch
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig
    )


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception:
        pass


def set_seed(seed):
    """
    Random seed to ensure reproducibility
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def set_logger(args):

    logging.basicConfig()

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_path = f"{datetime.datetime.now().strftime('%Y-%m-%d-%H%M-%S')}.log"
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
        level=logging.DEBUG,
        filemode="w",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger


def initialize_model(model_id, device, quantization):
    """
    Initialize model
    """
    # Quantization
    if quantization == 2:
        quantization_config = BitsAndBytesConfig(
            load_in_2bit=True,
            bnb_2bit_compute_dtype=torch.float16,  # The compute dtype for 2-bit operations
            bnb_2bit_use_double_quant=True,        # Whether to use double quantization (optional)
            bnb_2bit_quant_type="nf4"              # The quantization type, e.g. "nf4" or "fp4"
        )
    elif quantization == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # The compute dtype for 4-bit operations
            bnb_4bit_use_double_quant=True,        # Whether to use double quantization (optional)
            bnb_4bit_quant_type="nf4"              # The quantization type, e.g. "nf4" or "fp4"
        )
    else:
        quantization_config = None

    # Load Processor and Model from id
    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    return model, processor


def setup_roles():

    # Use the "target" field in system messages to restrict visibility.
    conversation_history = []

    # System message for Architect (visible only to Architect)
    conversation_history.append({
        "role": "system",
        "target": "Architect",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are an agent playing a collaborative building task along with a partner."
                    "Your role is that of the Architect, while your partner is the Builder."
                    "You will be shown images of a target structure built in a voxel world,"
                    "and your job is to guide the Builder in order to replicate it."
                    "Give clear and easy to follow instructions."
                    "Proceed step by step and avoid providing too many instructions all at once."
                    "The Builder will reply with the actions it took and, possibly, clarification questions."
                    "Acknowledge the Builder's actions and feedback in order to understand whether they are on the right track or not and to help them."
                    "When you think that the Builder correctly completed the structure, output '[FINISH]' to trigger the end of the game."
                    "Here are the images of the target structure from four points of view: "
                )
            },
            {"type": "image"}, {"type": "image"}, {"type": "image"}, {"type": "image"}
        ]
    })

    # System message for Builder (visible only to Builder)
    conversation_history.append({
        "role": "system",
        "target": "Builder",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are an agent playing a collaborative building task along with a partner."
                    "Your role is that of the Builder, while your partner is the Architect."
                    "Your job is to follow the Architect's instructions to build what they describe."
                    "Output the corresponding actions in terms of XYZ coordinates,"
                    "like 'add.(red, 0, 1, 0)' or 'remove.(green, 0, 1, 3)'."
                    "If you have any doubts or need clarification, ask the Architect."
                    "The Architect will tell you when the structure is complete."
                )
            }
        ]
    })

    return conversation_history


def filter_conversation(conversation, target_model):
    """
    Returns a new conversation list containing:
      - All non-system messages.
      - System messages only if they are intended for target_model.
    Additionally, removes the "target" key from the message so that the
    processor receives only the expected keys.
    """
    filtered = []
    for message in conversation:
        if message["role"] == "system":
            # If a system message has a "target" field, only include it if it matches the Architectt hand.
            if "target" in message:
                if message["target"] == target_model:
                    # Rewrite the message without the "target" key.
                    filtered.append({k: v for k, v in message.items() if k != "target"})
            # Else just include the message.
            else:
                filtered.append(message)
        else:
            filtered.append(message)
    return filtered

