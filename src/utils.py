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

current_path = os.path.dirname(__file__)
main_path = os.path.abspath(os.path.join(current_path, os.pardir))


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
    
    log_file_path = os.path.join(main_path, "results", f"{log_time}_{combo_id}.log")
    #log_path = f"{log_time}_{combo_id}.log"
    logging.basicConfig(
        filename=log_file_path,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
        level=logging.INFO,
        filemode="w",
    )

    logger = logging.getLogger()
    return log_time, logger


def initialize_model(model_id, device, q):
    """
    Initialize model
    """
    # Quantization
    if q == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # The compute dtype for 4-bit operations
            bnb_4bit_use_double_quant=True,        # Whether to use double quantization (optional)
            bnb_4bit_quant_type="nf4"              # The quantization type, e.g. "nf4" or "fp4"
        )
    elif q == 8:
         quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
         )
    else:
        quantization_config = None

    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device,
    )

    return model

def build_arch_prompt(use_img, use_json, json_text):
    """
    Build the prompt for the Architect given image or JSON optionals.
    Returns the conversation history with the new chat text appended.
    """

    img_prompt = "images of a target structure built in a voxel world, "
    json_prompt = "a JSON text file representing the target structure, "

    ### Prompt for the Architect ###
    # IMG AND JSON
    if use_img and use_json:
        arch_prompt = {
            "role": "user",
            "speaker": "system",
            "target": "Architect",
            "content": [
                {"type": "image"}, {"type": "image"}, {"type": "image"}, {"type": "image"},
                {
                    "type": "text",
                    "text":
                        (
                            "You are an agent playing a collaborative building task along with a partner. "
                            "Your role is that of the Architect, while your partner is the Builder. You will be shown "
                            "images of a target structure built in a voxel world, "
                            f"{img_prompt}and {json_prompt}"
                            "and your job is to guide the Builder "
                            "in order to replicate it. Give clear and easy to follow instructions. Proceed step by step "
                            "and avoid providing too many instructions all at once. The Builder will reply with the "
                            "actions it took and, possibly, clarification questions. Acknowledge the Builder's actions "
                            "and feedback in order to understand whether they are on the right track or not and to help "
                            "them. When you think that the Builder correctly completed the structure, output '[FINISH]' "
                            "to trigger the end of the game. Here are the images of the target structure from four points of view and "
                            "the JSON text:\n"
                            f"{json_text}"
                        )
                },
            ]
        }
    # JSON ONLY
    elif use_json:
        arch_prompt = {
            "role": "user",
            "speaker": "system",
            "target": "Architect",
            "content": [
                {
                    "type": "text",
                    "text":
                        (
                            "You are an agent playing a collaborative building task along with a partner. "
                            "Your role is that of the Architect, while your partner is the Builder. You will be shown "
                            "images of a target structure built in a voxel world, "
                            f"{json_prompt}"
                            "and your job is to guide the Builder "
                            "in order to replicate it. Give clear and easy to follow instructions. Proceed step by step "
                            "and avoid providing too many instructions all at once. The Builder will reply with the "
                            "actions it took and, possibly, clarification questions. Acknowledge the Builder's actions "
                            "and feedback in order to understand whether they are on the right track or not and to help "
                            "them. When you think that the Builder correctly completed the structure, output '[FINISH]' "
                            "to trigger the end of the game. Here is "
                            "the JSON text:\n"
                            f"{json_text}"
                        )
                },
            ]
        }
    # IMG ONLY
    else:
        arch_prompt = {
            "role": "user",
            "speaker": "system",
            "target": "Architect",
            "content": [
                {"type": "image"}, {"type": "image"}, {"type": "image"}, {"type": "image"},
                {
                    "type": "text",
                    "text":
                        (
                            "You are an agent playing a collaborative building task along with a partner. "
                            "Your role is that of the Architect, while your partner is the Builder. You will be shown "
                            "images of a target structure built in a voxel world, "
                            f"{img_prompt}"
                            "and your job is to guide the Builder "
                            "in order to replicate it. Give clear and easy to follow instructions. Proceed step by step "
                            "and avoid providing too many instructions all at once. The Builder will reply with the "
                            "actions it took and, possibly, clarification questions. Acknowledge the Builder's actions "
                            "and feedback in order to understand whether they are on the right track or not and to help "
                            "them. When you think that the Builder correctly completed the structure, output '[FINISH]' "
                            "to trigger the end of the game. Here are the images of the target structure from four points of view: "
                        )
                },
            ]
        }

    return arch_prompt


def setup_roles(use_img, use_json, shot, json_text):
    """
    Initialize the conversation by setting up the roles for Architect and Builder.
    """

    conversation_history = []

    ### Prompt for the Architect
    architect_prompt = build_arch_prompt(use_img, use_json, json_text)
    conversation_history.append(architect_prompt)

    # If one-shot, add an example
    if shot:
        one_shot_text = {
        "role": "user",
        "speaker": "system",
        "content": [
                # Description and gold
                {"type": "text", "text": "Here is also an example of a task completed by two humans. You will see the images of the gold configuration and images of the structure being built.\n[EXAMPLE]\n"},
                {"type": "image"},
                {"type": "text", "text": (
                                            'hello\n'
                                            '{"feedback": "hello"}\n'
                                            'are u rdy to get to work?\n'
                                            '{"feedback": "yes"}\n'
                                            'ok\nbuild a 2x1 structure that is blue\n'
                                            '{"feedback": "is the structure extending upwards?"}\n'
                                            'no, it goes across\n'
                                        )},
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": (
                                            '{"add": [[0,0,-2,\"blue\"], [0,0,-3,\"blue\"]], "feedback": "is that good?"}\n'
                                            'now place 1 blue piece on the left block extending upwards\n'
                                            'yes that is correct\n'
                                        )},
                {"type": "image"},
                {"type": "text", "text": '{"add": [[0,1,-2,\"blue\"]], "feedback": "like that?"}\n'},
                {"type": "image"},
                {"type": "text", "text": (
                                            'yes, now it is finished\n'
                                            '{"feedback": "good job!}\n'
                                            'you too builder\n[/EXAMPLE]'
                                        )}
            ],
        }
        conversation_history.append(one_shot_text)

    ### Prompt for Builder ###
    conversation_history.append({
        "role": "user",
        "speaker": "system",
        "target": "Builder",
        "content": [
            {
                "type": "text",
                "text": 
                    ( 
                        "You are an agent playing a collaborative building task along with a partner. Your role is "
                        "that of the Builder, while your partner is the Architect. Your job is to follow the Architect's "
                        "instructions to build what they describe. You are in a voxel world, where the most northernly point "
                        "is 0,0,-5; the most westerly point -5,0,0; the most eastern point is 5,0,0; the most southern 0,0,5 "
                        "and the y-axis is up and down, with y=0 being the minimum. Describe the coordinates of the blocks "
                        "**you want to interact with** and their colours (must be one of: blue, yellow, green, orange, purple, "
                        "red) and whether the action is to add or remove them, your confidence in your interpretation of the "
                        #"instruction and optionally a question if the instruction is potentially unclear, in the JSON format: "
                        "instruction and textual feedback, in the JSON format: "
                        "{\"add\": [[x,y,z,\"color\"], ...], \"remove\": [[x,y,z,\"color\"], ...], \"confidence\": 0.0, "
                        "\"feedback\": \"...\"}. Give the JSON only, no additional dialog."
                    )
            }
        ]
    })


    ### Start ##
    conversation_history.append({
        "role": "user",
        "speaker": "system",
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
                    new_message = {k: v for k, v in message.items() if k != "target"}
                    if "speaker" in new_message:
                        new_message.pop("speaker")  # Remove "speaker" if it exists
                    filtered.append(new_message)
            else:
                # Create a new dictionary copy to avoid modifying the original
                new_message = message.copy()
                if "speaker" in new_message:
                    new_message.pop("speaker")  # Remove "speaker" only in filtered
                filtered.append(new_message)
        else:
        # Else just include the message.
            filtered.append(message)
    return filtered


def generate_response(model, processor, conversation, target_name, images=None, max_new_tokens=2048, rep_penalty=1.1):
    """
    Generates a response for the given model using the (filtered) conversation history.
    
    - Filters out system messages that are not intended for the current model.
    - role_name should be the current model's identifier (e.g., "Architect" or "Builder").
    """

    # Filter conversation for the current model.
    filtered_conversation = filter_conversation(conversation, target_model=target_name)
    
    # Build the prompt using the processor's chat template.
    prompt = processor.apply_chat_template(filtered_conversation, add_generation_prompt=True)

    inputs = processor(
        images=images,
        text=prompt,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    generate_ids = model.generate(
        **inputs,
        do_sample=False,                                # Deterministic generation
        pad_token_id=processor.tokenizer.eos_token_id,  # explicitly setting pad_token_id
        max_new_tokens=max_new_tokens,
        repetition_penalty=rep_penalty                  # Avoid infinite loops
    )  
    output = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    parsed = output[0].split("[/INST] ")[-1]

    # Since [INST] and [/INST] are NOT single tokens (i.e., they are processede like '['+'/'+'INST'+'])
    # the model picks it up and understands that when [/ is happening, there is a switch of role.
    # The model then tries to force the switch of role and complete the task on its own (lol).
    if "[/" in parsed:
        #Truncate at the point of the unwanted token
        parsed = parsed.split("[/")[0]
    
    #roles = ["[ARCHITECT]", "[BUILDER]"]
    #for r in roles:
    #    if r in parsed:
    #        parsed = parsed.split(r)[0]

    return parsed


def load_structure(structure_id):
    # Where the rendered structures are
    gold_processed_path = os.path.join(main_path, "data", "structures", "gold-processed")
    #gold_processed_path = "../data/structures/gold-processed"
    lookup_file = os.path.join(main_path, "data", "structures", "configs-to-names.txt")
    #lookup_file = "../data/structures/configs-to-names.txt"

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
        json_text = json.load(open(json_path, "r"))
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
    
    return combo_id, json_text, s_images_list


def load_one_shot():

    one_path = os.path.join(main_path, "data", "minecraft_corpus", "one_shot")
    #one_path = "../data/minecraft_corpus/one_shot"
    one_shot_images = []

    # Load the images
    for filename in os.listdir(one_path):
        # Check if the file has a JPG or JPEG extension (case insensitive)
        if filename.lower().endswith(('.png')):
            img_path = os.path.join(one_path, filename)
        try:
            # Open the image file using PIL
            img = Image.open(img_path)
            one_shot_images.append(img)
        except IOError:
            raise IOError(f"Warning: Could not open image {img_path}")
    
    return one_shot_images

def save_conversation(conversation_history, json_file):
    """Saves the conversation dynamically after each change."""
    with open(json_file, "w") as f:
        json.dump(conversation_history, f, indent=4)
