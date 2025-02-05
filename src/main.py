import torch
from PIL import Image
import argparse
from utils import *
import json


def get_args():
    """
    Argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id_A", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--model_id_B", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--quantization", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--logdir", type=str, required=False, default="../results/", help="Log directory path")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_rounds", type=int, default=10)
    parser.add_argument("--init_seed", type=int, default=0, help="Random seed")
    parser.add_argument("--structure_id", type=str, required=True)
    args = parser.parse_args()
    return args


def generate_response():
    #FILTER CONVERSATION

    # BUILD THE PROMPT USING THE PROCESSOR's CHAT TEMPLATE

    # MAKE IT MODULAR TO HAVE IMAGES OR NOT!!!!
    pass


def load_structure(structure_id):
    # Where the rendered structures are
    gold_processed_path = "../data/structures/gold-processed"

    # Construct the path to the structure
    structure_path = os.path.join(gold_processed_path, structure_id)

    # Load the structure JSON
    try:
        json_path = os.path.join(structure_path, f"{structure_id}.json")
        s_json = json.load(open(json_path, "r"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Structure {structure_id} not found.")
    
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
    
    return s_json, s_images_list



if __name__ == "__main__":

    args = get_args()  # Get arguments from the parser
    mkdirs(args.logdir)  # Create directory to save log
    logger = set_logger(args)  # Set the logger
    set_seed(args.init_seed)  # Set seed

    # Arguments in plain English
    plain_args = (
        f"Architect: {args.model_id_A}, Builder: {args.model_id_B}\n"
        f"Device: {args.device}, Quantization: {args.quantization}\n"
        f"Max new tokens: {args.max_new_tokens}, Temperature: {args.temperature}\n"   
    )
    logger.info(plain_args)

    # Initialize models
    logger.info("Initializing models...")
    model_A, processor_A = initialize_model(args.model_id_A, args.device, args.quantization)
    model_B, processor_B = initialize_model(args.model_id_B, args.device, args.quantization)
    
    
    # LOAD IMAGES FROM STRUCTURE
    s_json, s_images_list = load_structure(args.structure_id)

    # Initialize conversation loop
    current_round = 0
    conversation_history = setup_roles()

    while current_round < args.max_rounds:
        logger.info(f"===== Round {current_round + 1} =====")

        # ----- Architect's Turn -----
        # For the first turn, pass the images; later turns might not require images.
        modelA_response = generate_response(
            model=model_A,
            processor=processor_A,
            conversation=conversation_history,
            role_name="Architect",
            images=s_images_list,
            max_new_tokens=args.max_new_tokens
        )

        logger.info("Architect:", modelA_response)

        # Append Architect's response to the conversation history.
        conversation_history.append({
            "role": "Architect",
            "content": [
                {"type": "text", "text": modelA_response}
            ]
        })

        # Check if Architect signaled to finish.
        if "[FINISH]" in modelA_response:
            logger.info("Finishing conversation as indicated by Architect.")
            break

        # ----- Builder's Turn -----
        modelB_response = generate_response(
            model=model_B,
            processor=processor_B,
            conversation=conversation_history,
            role_name="Builder",
            images=None,
            max_new_tokens=200
        )

        logger.info("Builder:", modelB_response)

        # Append Builder's response to the conversation history.
        conversation_history.append({
            "role": "Builder",
            "content": [
                {"type": "text", "text": modelB_response}
            ]
        })

        current_round += 1

    logger.info("Conversation ended.")
