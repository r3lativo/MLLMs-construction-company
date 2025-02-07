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
                    "You are an agent playing a collaborative building task along with a partner. "
                    "Your role is that of the Architect, while your partner is the Builder. "
                    "You will be shown images of a target structure built in a voxel world, "
                    "and your job is to guide the Builder in order to replicate it. "
                    "Give clear and easy to follow instructions. "
                    "Proceed step by step and avoid providing too many instructions all at once. "
                    "The Builder will reply with the actions it took and, possibly, clarification questions. "
                    "Acknowledge the Builder's actions and feedback in order to understand whether they are on the right track or not and to help them. "
                    "When you think that the Builder correctly completed the structure, output '[FINISH]' to trigger the end of the game. "
                    "Here are the images of the target structure from four points of view: "
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
                    'You are an agent playing a collaborative building task along with a partner. '
                    "Your role is that of the Builder, while your partner is the Architect. "
                    "Your job is to follow the Architect's instructions to build what they describe. "

                    'You are in a voxel world, '
                    'where the most northernly point is 0,0,-5; the most westerly point -5,0,0; '
                    'the most eastern point is 5,0,0; the most southern 0,0,5 and the y-axis is up and down, '
                    'with y=0 being the minimum. '
                    'Describe the coordinates of the blocks **you want to interact with** and their colours '
                    '(must be one of: blue, yellow, green, orange, purple, red) and whether the action is '
                    'to add or remove them, your confidence in your interpretation of the instruction and optionally '
                    'a question if the instruction is potentially unclear, in the JSON format: '
                    '{"add": [[x,y,z,"color"], ...], "remove": [[x,y,z,"color"], ...], "confidence": 0.0, "question": "..."}. '
                    'Give the JSON only, no additional dialog.'
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
