import torch
import argparse
from utils import *
import json
import sys


def get_args():
    """
    Argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--n_models", type=int, default=2, choices=[1,2])
    parser.add_argument("-q,", "--quantization", type=int, default=4, choices=[4,8])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results_dir", type=str, required=False, default="../results/", help="Results directory path")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_rounds", type=int, default=5)
    parser.add_argument("--init_seed", type=int, default=0, help="Random seed")
    parser.add_argument("--structure_id", type=str, required=True)
    parser.add_argument("-rp", "--repetition_penalty", type=float, default=1.1)

    parser.add_argument("--img", type=bool, default=False)
    parser.add_argument('--use_img', dest='img', action='store_true')
    parser.add_argument("--json", type=bool, default=False)
    parser.add_argument('--use_json', dest='json', action='store_true')
    parser.add_argument("--shot", type=bool, default=False)
    parser.add_argument('--oneshot', dest='shot', action='store_true')
    args = parser.parse_args()
    return args


def generate_response(model, processor, conversation, target_name, images, max_new_tokens, rep_penalty):
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


if __name__ == "__main__":

    # Get arguments from the parser
    args = get_args()

    # Check that at least one input is chosen
    if not args.img and not args.json:
        sys.exit("At least one between the images and the JSON text has to be chosen.\nAdd --use_img or --use_json to your input.")

    # Create directory to save log, if needed
    mkdirs(args.results_dir)  

    # Load structure
    combo_id, json_text, images_list = load_structure(args.structure_id)

    # Load one shot images
    if args.shot:
        one_shot_images = load_one_shot()
        images_list.extend(one_shot_images)

    # Set the logger and store the time
    log_time, logger = set_logger(args, combo_id)

    # Set random seed
    set_seed(args.init_seed)

    # Arguments in plain English
    logger.info(
        f"Model: {args.model_id}, Quantization: {args.quantization}-bit, "
        f"Device: {args.device}, Number of models: {args.n_models}, "
        f"Max new tokens: {args.max_new_tokens}, Max rounds: {args.max_rounds}, "
        f"Input Image: {args.img}, Input JSON: {args.json}"
        )

    ### Initialize models ###
    logger.info("Initializing model(s)...")
    processor = LlavaNextProcessor.from_pretrained(args.model_id, use_fast=True, padding_side="left")
    model_A = initialize_model(args.model_id, args.device, args.quantization)
    # The models are separated
    if args.n_models == 2: model_B = initialize_model(args.model_id, args.device, args.quantization)
    # The models are the SAME model
    else: model_B = model_A
    logger.info(f"{args.n_models} Model(s) initialized")

    ########## Conversation Loop ##########
    # Conversation with the selected input(s) and one or zero-shot
    conversation_history = setup_roles(args.img, args.json, args.shot, json_text)
    current_round = 0

    while current_round < args.max_rounds:
        logger.info(f"===== Round {current_round + 1} =====")
        print(f"===== Round {current_round + 1} =====")

        ########## Architect's Turn ##########
        modelA_response = generate_response(
            model=model_A,
            processor=processor,
            conversation=conversation_history,
            target_name="Architect",
            images=images_list if current_round == 0 else None,  # Pass images only in first turn
            max_new_tokens=args.max_new_tokens,
            rep_penalty=args.repetition_penalty
        )
        
        # Ensure at least 2 rounds
        if "[FINISH]" in modelA_response and current_round < 2:
            modelA_response = modelA_response.replace("[FINISH]", "")
        
        print(f"##### ARCHITECT #####\n{modelA_response}\n")

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
            images=one_shot_images if current_round == 0 and args.shot else None,
            max_new_tokens=args.max_new_tokens,
            rep_penalty=args.repetition_penalty
        )
        print(f"##### BUILDER #####\n{modelB_response}\n")

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
    