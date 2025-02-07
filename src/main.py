import torch
import argparse
from utils import *
import json


def get_args():
    """
    Argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--quantization", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results_dir", type=str, required=False, default="../results/", help="Results directory path")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_rounds", type=int, default=5)
    parser.add_argument("--init_seed", type=int, default=0, help="Random seed")
    parser.add_argument("--structure_id", type=str, required=True)
    args = parser.parse_args()
    return args


def generate_response(model, processor, conversation, target_name, images=None, max_new_tokens=128):
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
        return_tensors="pt"
    ).to(model.device)
    
    generate_ids = model.generate(**inputs,
                                  do_sample=False,     # Deterministic generation
                                  max_new_tokens=max_new_tokens)
    output = processor.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    
    parsed = output[0].split("[/INST] ")[-1]

    # Since [INST] and [/INST] are NOT single tokens (i.e., they are processede like '['+'/'+'INST'+'])
    # the model picks it up and understands that when [/ is happening, there is a switch of role.
    # The model then tries to force the switch of role and complete the task on its own (lol).
    if "[/" in parsed:
        #Truncate at the point of the unwanted token
        parsed = parsed.split("[/")[0]

    return parsed


if __name__ == "__main__":

    # Get arguments from the parser
    args = get_args()

    # Create directory to save log, if needed
    mkdirs(args.results_dir)  

    # Load structure
    combo_id, s_json, s_images_list = load_structure(args.structure_id)

    # Set the logger and store the time
    log_time, logger = set_logger(args, combo_id)

    # Set random seed
    set_seed(args.init_seed)

    # Arguments in plain English
    plain_args = (
        f"Model: {args.model_id}, Quantization: {args.quantization}, Device: {args.device}, "
        f"Max new tokens: {args.max_new_tokens}, Max rounds: {args.max_rounds}"   
    )
    logger.info(plain_args)

    # Initialize models
    logger.info("Initializing models...")
    model_A, model_B, processor = initialize_model(args.model_id, args.device, args.quantization)
    logger.info("Models initialized")

    # Initialize conversation loop
    current_round = 0
    conversation_history = setup_roles()


    ########## Conversation Loop ##########
    while current_round < args.max_rounds:
        logger.info(f"===== Round {current_round + 1} =====")

        ########## Architect's Turn ##########
        modelA_response = generate_response(
            model=model_A,
            processor=processor,
            conversation=conversation_history,
            target_name="Architect",
            images=s_images_list if current_round == 0 else None,  # Pass images only in first turn
            max_new_tokens=args.max_new_tokens
        )
        print(f"[ARCHITECT]: {modelA_response}")

        # Append Architect's response to the conversation history.
        conversation_history.append({
            "role": "user",
            "content": [
                {"type": "text", "text": modelA_response}
            ]
        })

        # Check if Architect signaled to finish.
        if "[FINISH]" in modelA_response:
            logger.info("Finishing conversation as indicated by Architect.")
            break


        ########## Builder's Turn ##########
        modelB_response = generate_response(
            model=model_B,
            processor=processor,
            conversation=conversation_history,
            target_name="Builder",
            images=None,
            max_new_tokens=args.max_new_tokens
        )
        print(f"[BUILDER]: {modelB_response}")

        # Append Builder's response to the conversation history.
        conversation_history.append({
            "role": "user",
            "content": [
                {"type": "text", "text": modelB_response}
            ]
        })

        current_round += 1

    ########## End Conversation ##########
    logger.info("Conversation ended.")

    # Save conversation into JSON file
    with open(f"../results/{log_time}_{combo_id}.json", "w") as file:
        json.dump(conversation_history, file, indent=4)  # indent=4 makes it pretty printed (optional)
    