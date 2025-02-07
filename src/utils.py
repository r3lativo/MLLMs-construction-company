import datetime
import logging
import os
import torch
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig
    )
import json
from PIL import Image
import pandas as pd


def mkdirs(dirpath):
    try: os.makedirs(dirpath)
    except Exception: pass


def set_seed(seed):
    """
    Random seed to ensure reproducibility
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)


def set_logger(args, combo_id):

    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    
    log_time = datetime.datetime.now().strftime('%Y-%m-%d-%H%M-%S')
    
    log_path = f"{log_time}_{combo_id}.log"
    logging.basicConfig(
        filename=os.path.join(args.results_dir, log_path),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
        level=logging.INFO,
        filemode="w",
    )

    logger = logging.getLogger()
    return log_time, logger


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
        print("Quantization has to either be 2 or 4.\n Quantization set to None.")
        quantization_config = None

    # Load Processor and Model from id
    processor = LlavaNextProcessor.from_pretrained(model_id,
                                                   padding_side="left")  
    model_A = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)

    model_B = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    return model_A, model_B, processor


def setup_roles():

    # Use the "target" field in system messages to restrict visibility.
    conversation_history = []

    # System message for Architect (visible only to Architect)
    conversation_history.append({
        "role": "user",
        "target": "Architect",
        "content": [
            {"type": "image"}, {"type": "image"}, {"type": "image"}, {"type": "image"},
            {
                "type": "text",
                "text": (
                    "You are an expert Architect working on a collaborative task in a voxel world with the Builder. "
                    "You have four high-resolution images showing the target structure from front, back, left, and right. "
                    "The Builder has no access to the target structure images. "
                    "Guide the Builder with clear, step-by-step instructions and limit directives to one at a time. "
                    "The voxel world is bounded by these coordinates: north (0,0,94), south (0,0,104), "
                    "west (95,0,0), east (105,0,0), with the y-axis representing height (y=0 is ground level). "
                    "The Builder will reply with a JSON with the actions it took and, possibly, a chat text. "
                    "Acknowledge the Builder's actions and feedback in order to understand whether they are on the right track or not and to help them. "
                    "When you think that the Builder correctly completed the structure, output '[FINISH]' to trigger the end of the game. "
                )
            },
        ]
    })


        # System message for Builder (visible only to Builder)
    conversation_history.append({
        "role": "user",
        "target": "Builder",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are an expert Builder working on a collaborative task in a voxel world with the Architect. "
                    "Follow instructions carefully. Your goal is to replicate the target structure as described by the Architect, "
                    "step by step. The voxel world is bounded by these coordinates: north (0,0,94), south (0,0,104), "
                    "west (95,0,0), east (105,0,0), with the y-axis representing height (y=0 is ground level). "
                    "IMPORTANT: Each action must add or remove exactly one block at a specific coordinate. "
                    "Ensure that every coordinate is specified using integers only (e.g., 2, not 2.5). "
                    "When acting, provide your moves in JSON format that lists block coordinates, actions, and "
                    "chosen colors. Valid colors are: blue, yellow, green, orange, purple, and red. "
                    "Include your confidence level (from 0.0 to 1.0) and, optionally, communicate with the \"chat\" parameter. "
                    "Example format: {\"add\": [[x,y,z,\"color\"]], \"remove\": [[x,y,z,\"color\"]], "
                    "\"confidence\": 0.0, \"chat\": \"...\"}. Do not add any extra dialogue."
                )
            }
        ]
    })


    # System message for Builder (visible only to Builder)
    conversation_history.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "[START]"
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
        if message["role"] == "user":
            # If a system message has a "target" field, only include it if it matches the role at hand.
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


def load_structure(structure_id):
    # Where the rendered structures are
    gold_processed_path = "../data/structures/gold-processed"
    lookup_file = "../data/structures/configs-to-names.txt"

    df = pd.read_csv(lookup_file, sep="\t", header=None, names=["code", "name"])
    df["combined"] = df["code"] + "_" + df["name"]

    # Check if the query matches any of the three columns.
    matched_row = df[(df["code"] == structure_id) |
                     (df["name"] == structure_id) |
                     (df["combined"] == structure_id)]
    if not matched_row.empty:
        # Return the combined value from the first match.
        combo_id = matched_row.iloc[0]["combined"]
    else:
        raise FileNotFoundError(f"'{structure_id}' is incorrect.")

    # Construct the path to the structure
    structure_path = os.path.join(gold_processed_path, combo_id)

    # Load the structure JSON
    try:
        json_path = os.path.join(structure_path, f"{combo_id}.json")
        s_json = json.load(open(json_path, "r"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Structure {combo_id} not found.")
    
    s_images_list = []

    # Load the images
    for filename in os.listdir(structure_path):
        # Check if the file has a JPG or JPEG extension (case insensitive)
        if filename.lower().endswith(('.jpg', '.jpeg')):
            img_path = os.path.join(structure_path, filename)
            try:
                # Open the image file using PIL
                img = Image.open(img_path)
                s_images_list.append(img)
            except IOError:
                raise IOError(f"Warning: Could not open image {img_path}")
    
    return combo_id, s_json, s_images_list
