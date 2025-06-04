import os
import json
from pathlib import Path
import re
import shutil
from itertools import groupby
from operator import itemgetter
from utils import main_path

def extract_and_copy_dialogues(new_results_path):
    structures_id = ""
    file_type = ".json"
    for dir_name in os.listdir(new_results_path):
        if os.path.isdir(os.path.join(new_results_path, dir_name)):
            structures_id = dir_name + file_type
        for subf in os.listdir(os.path.join(new_results_path, dir_name)):
            if subf == "img_only":
                dialogue_file = [
                    file for file in os.listdir(os.path.join(new_results_path, dir_name, subf))
                    if os.path.isfile(os.path.join(new_results_path, dir_name, subf, file)) and file == "builder_history.json"
                ]
                dialogue_file_path = os.path.join(new_results_path, dir_name, subf, dialogue_file[0])
                src_path = Path(dialogue_file_path)
                dst_path = Path(os.path.join(main_path, "results", "dialogues_og", "img_only"))
                new_path = dst_path / structures_id
                shutil.copy2(src_path, new_path)
            elif subf == "json_only":
                dialogue_file = [
                    file for file in os.listdir(os.path.join(new_results_path, dir_name, subf))
                    if os.path.isfile(os.path.join(new_results_path, dir_name, subf, file)) and file == "builder_history.json"
                ]
                dialogue_file_path = os.path.join(new_results_path, dir_name, subf, dialogue_file[0])
                src_path = Path(dialogue_file_path)
                dst_path = Path(os.path.join(main_path, "results", "dialogues_og", "json_only"))
                new_path = dst_path / structures_id
                shutil.copy2(src_path, new_path)
            elif subf == "img_json":
                dialogue_file = [
                    file for file in os.listdir(os.path.join(new_results_path, dir_name, subf))
                    if os.path.isfile(os.path.join(new_results_path, dir_name, subf, file)) and file == "builder_history.json"
                ]
                dialogue_file_path = os.path.join(new_results_path, dir_name, subf, dialogue_file[0])
                src_path = Path(dialogue_file_path)
                dst_path = Path(os.path.join(main_path, "results", "dialogues_og", "img_json"))
                new_path = dst_path / structures_id
                shutil.copy2(src_path, new_path)

#new_results_path = os.path.join(main_path, "results", "new_results")
#extract_and_copy_dialogues(new_results_path)

def parse_dialogues(dialogues_path):
    #parsed_dialogues = []
    for input_type_dir in os.listdir(dialogues_path):
        if os.path.isdir(os.path.join(dialogues_path, input_type_dir)) and input_type_dir == "img_only":
            for file in os.listdir(os.path.join(dialogues_path, input_type_dir)):
                with open(os.path.join(dialogues_path, input_type_dir, file), "r") as f:
                    data = json.load(f)
                    # parsing
                    parsed_dialogue = []
                    for turn in data:
                        role = turn.get("role")
                        content = turn.get("content")
                        if role == "system":
                            continue # skipe system message (i.e. prompt)
                        if role == "user":
                            if isinstance(content, list) and content and isinstance(content[0], dict):
                                text = content[0].get("text", "")
                                text = re.sub(r'Your current inventory: \{.*?\}', '', text).strip()
                                parsed_dialogue.append({"Architect": text})
                        elif role == "assistant":
                            if isinstance(content, str):
                                try:
                                    json_match = re.search(r'{.*}', content, re.DOTALL)
                                    if json_match:
                                        action_json = json.loads(json_match.group())
                                        communication = action_json.get("communication", "").strip()
                                        parsed_dialogue.append({"Builder": communication})
                                except json.JSONDecodeError:
                                    pass
                    output_path = os.path.join(main_path, "results", "dialogues_parsed", "img_only", file)
                    with open(output_path, "w") as f_out:
                        json.dump(parsed_dialogue, f_out, indent = 2)

        elif os.path.isdir(os.path.join(dialogues_path, input_type_dir)) and input_type_dir == "json_only":
            for file in os.listdir(os.path.join(dialogues_path, input_type_dir)):
                with open(os.path.join(dialogues_path, input_type_dir, file), "r") as f:
                    data = json.load(f)
                    # parsing
                    parsed_dialogue = []
                    for turn in data:
                        role = turn.get("role")
                        content = turn.get("content")
                        if role == "system":
                            continue
                        if role == "user":
                            if isinstance(content, list) and content and isinstance(content[0], dict):
                                text = content[0].get("text", "")
                                text = re.sub(r'Your current inventory: \{.*?\}', '', text).strip()
                                parsed_dialogue.append({"Architect": text})
                        elif role == "assistant":
                            if isinstance(content, str):
                                try:
                                    json_match = re.search(r'{.*}', content, re.DOTALL)
                                    if json_match:
                                        action_json = json.loads(json_match.group())
                                        communication = action_json.get("communication", "").strip()
                                        parsed_dialogue.append({"Builder": communication})
                                except json.JSONDecodeError:
                                    pass
                    output_path = os.path.join(main_path, "results", "dialogues_parsed", "json_only", file)
                    with open(output_path, "w") as f_out:
                        json.dump(parsed_dialogue, f_out, indent = 2)
        elif os.path.isdir(os.path.join(dialogues_path, input_type_dir)) and input_type_dir == "img_json":
            for file in os.listdir(os.path.join(dialogues_path, input_type_dir)):
                with open(os.path.join(dialogues_path, input_type_dir, file), "r") as f:
                    data = json.load(f)
                    # parsing
                    parsed_dialogue = []
                    for turn in data:
                        role = turn.get("role")
                        content = turn.get("content")
                        if role == "system":
                            continue
                        if role == "user":
                            if isinstance(content, list) and content and isinstance(content[0], dict):
                                text = content[0].get("text", "")
                                text = re.sub(r'Your current inventory: \{.*?\}', '', text).strip()
                                parsed_dialogue.append({"Architect": text})
                        elif role == "assistant":
                            if isinstance(content, str):
                                try:
                                    json_match = re.search(r'{.*}', content, re.DOTALL)
                                    if json_match:
                                        action_json = json.loads(json_match.group())
                                        communication = action_json.get("communication", "").strip()
                                        parsed_dialogue.append({"Builder": communication})
                                except json.JSONDecodeError:
                                    pass
                    output_path = os.path.join(main_path, "results", "dialogues_parsed", "img_json", file)
                    with open(output_path, "w") as f_out:
                        json.dump(parsed_dialogue, f_out, indent = 2)

#dialogues_path = os.path.join(main_path, "results", "dialogues_og")
#parse_dialogues(dialogues_path)

def add_condition(dialogues_path):
    for input_condition in os.listdir(dialogues_path):
        if os.path.isdir(os.path.join(dialogues_path, input_condition)) and input_condition == "img_only":
            for file in os.listdir(os.path.join(dialogues_path, input_condition)):
                if os.path.isfile(os.path.join(dialogues_path, input_condition, file)):
                    with open(os.path.join(dialogues_path, input_condition, file), "r") as f:
                        dialogue = json.load(f)
                        condition_entry ={
                            "condition": "img_only",
                            "dialogue": dialogue
                        }
                        with open(os.path.join(dialogues_path, input_condition, file), "w") as f:
                            json.dump(condition_entry, f, indent=2)
        elif os.path.isdir(os.path.join(dialogues_path, input_condition)) and input_condition == "json_only":
            for file in os.listdir(os.path.join(dialogues_path, input_condition)):
                if os.path.isfile(os.path.join(dialogues_path, input_condition, file)):
                    with open(os.path.join(dialogues_path, input_condition, file), "r") as f:
                        dialogue = json.load(f)
                        condition_entry ={
                            "condition": "json_only",
                            "dialogue": dialogue
                        }
                        with open(os.path.join(dialogues_path, input_condition, file), "w") as f:
                            json.dump(condition_entry, f, indent=2)
        elif os.path.isdir(os.path.join(dialogues_path, input_condition)) and input_condition == "img_json":
            for file in os.listdir(os.path.join(dialogues_path, input_condition)):
                if os.path.isfile(os.path.join(dialogues_path, input_condition, file)):
                    with open(os.path.join(dialogues_path, input_condition, file), "r") as f:
                        dialogue = json.load(f)
                        condition_entry ={
                            "condition": "img_json",
                            "dialogue": dialogue
                        }
                        with open(os.path.join(dialogues_path, input_condition, file), "w") as f:
                            json.dump(condition_entry, f, indent=2)

def add_structure_id(dialogues_path):
    structure_id = ""
    for input_condition in os.listdir(dialogues_path):
        if os.path.isdir(os.path.join(dialogues_path, input_condition)):
            for file in os.listdir(os.path.join(dialogues_path, input_condition)):
                if os.path.isfile(os.path.join(dialogues_path, input_condition, file)):
                    structure_id = file[:-5]
                    with open(os.path.join(dialogues_path, input_condition, file), "r") as f:
                        dialogue = json.load(f)
                    new_data = {"structure_id": structure_id}
                    for key, value in dialogue.items():
                        new_data[key] = value
                    with open(os.path.join(dialogues_path, input_condition, file), "w") as f:
                        json.dump(new_data, f, indent=2)

#dialogues_path = os.path.join(main_path, "results", "parsed_dialogues")
#add_structure_id(dialogues_path)


def get_architect_messages_from_reference(reference_path):
    with open(reference_path, "r") as f:
        data = json.load(f)
        architect_messages = []
        for turn in data:
            if turn.get("role") == "assistant":
                content = turn.get("content")
                if isinstance(content, str):
                    # Try to extract from JSON if present
                    try:
                        match = re.search(r'{.*}', content, re.DOTALL)
                        if match:
                            action_json = json.loads(match.group())
                            communication = action_json.get("communication", "").strip()
                            if communication:
                                architect_messages.append(communication)
                                continue
                    except json.JSONDecodeError:
                        pass
                    # Fallback: use raw string content if no JSON or bad format
                    content = content.strip()
                    if content:
                        architect_messages.append(content)
        return architect_messages


def append_missing_architect_turns(parsed_dialogue_path, reference_dialogue_path):
    with open(parsed_dialogue_path, "r") as f:
        parsed_data = json.load(f)

    parsed_turns = parsed_data["dialogue"]
    existing_architect_msgs = [turn["Architect"] for turn in parsed_turns if "Architect" in turn]

    reference_architect_msgs = get_architect_messages_from_reference(reference_dialogue_path)

    print(f"\n→ File: {os.path.basename(parsed_dialogue_path)}")
    print(f"  Architect turns in parsed:   {len(existing_architect_msgs)}")
    print(f"  Architect turns in reference: {len(reference_architect_msgs)}")

    # Matching logic
    num_matching = 0
    for a, b in zip(existing_architect_msgs, reference_architect_msgs):
        if a.strip() == b.strip():
            num_matching += 1
        else:
            break

    print(f"  Matching Architect turns: {num_matching}")
    missing = reference_architect_msgs[num_matching:]
    print(f"  Architect turns to append: {len(missing)}")

    for msg in missing:
        parsed_turns.append({"Architect": msg})

    # Remove [FINISH] token from the last Architect message, if present
    for i in reversed(range(len(parsed_turns))):
        if "Architect" in parsed_turns[i]:
            parsed_turns[i]["Architect"] = parsed_turns[i]["Architect"].replace("[FINISH]", "").strip()
            break

    with open(parsed_dialogue_path, "w") as f:
        json.dump(parsed_data, f, indent=2)


def process_all_dialogues(parsed_root, architect_root):
    for condition in ["img_only", "json_only", "img_json"]:
        parsed_dir = os.path.join(parsed_root, condition)
        reference_dir = os.path.join(architect_root, condition)

        if not os.path.exists(parsed_dir):
            continue

        for filename in os.listdir(parsed_dir):
            parsed_file = os.path.join(parsed_dir, filename)
            reference_file = os.path.join(reference_dir, filename)

            if os.path.exists(reference_file):
                append_missing_architect_turns(parsed_file, reference_file)
                print(f"Updated: {parsed_file}")
            else:
                print(f"Missing reference for: {parsed_file}")


#parsed_root_dir = os.path.join(main_path, "results", "dialogues_parsed")
#architect_root_dir = os.path.join(main_path, "results", "dialogues_architect")
#process_all_dialogues(parsed_root_dir, architect_root_dir)


def combine_parsed_dialgoues(dialogues_path):
    combined_dialogues = []
    for input_condition in os.listdir(dialogues_path):
        if os.path.isdir(os.path.join(dialogues_path, input_condition)) and input_condition == "img_only":
            for file in os.listdir(os.path.join(dialogues_path, input_condition)):
                with open(os.path.join(dialogues_path, input_condition, file), "r") as f:
                    data = json.load(f)
                    combined_dialogues.append(data)
        elif os.path.isdir(os.path.join(dialogues_path, input_condition)) and input_condition == "json_only":
            for file in os.listdir(os.path.join(dialogues_path, input_condition)):
                with open(os.path.join(dialogues_path, input_condition, file), "r") as f:
                    data = json.load(f)
                    combined_dialogues.append(data)
        elif os.path.isdir(os.path.join(dialogues_path, input_condition)) and input_condition == "img_json":
            for file in os.listdir(os.path.join(dialogues_path, input_condition)):
                with open(os.path.join(dialogues_path, input_condition, file), "r") as f:
                    data = json.load(f)
                    combined_dialogues.append(data)

    output_path = os.path.join(main_path, "results", "all_dialogues_parsed", "all_dialogues_parsed.json")
    with open(output_path, "w") as f_out:
        json.dump(combined_dialogues, f_out, indent=2)

#dialogues_path = os.path.join(main_path, "results", "final_world_states")
#combine_parsed_dialgoues(dialogues_path)

def extract_number(structure_id):
    match = re.match(r"C(\d+)_", structure_id)
    return int(match.group(1)) if match else float('inf')

def reorder_dialogues(dialogues_combined_path):
    # Load your JSON list
    with open(dialogues_combined_path, "r") as f:
        data = json.load(f)  # a list of dicts

    # Group by condition (preserves the original order of conditions)
    data_sorted = []
    for condition, group_items in groupby(data, key=itemgetter("condition")):
        group_list = list(group_items)
        # Sort each condition group by structure_id number
        sorted_group = sorted(group_list, key=lambda x: extract_number(x["structure_id"]))
        data_sorted.extend(sorted_group)

    # Save back the sorted list
    with open(dialogues_combined_path, "w") as f:
        json.dump(data_sorted, f, indent=2)

#reorder_dialogues(os.path.join(main_path, "results", "all_final_world_states", "all_final_world_states.json"))
                        
#reorder_dialogues(os.path.join(main_path, "results", "all_dialogues_parsed", "all_dialogues_parsed_img_only.json"))
#reorder_dialogues(os.path.join(main_path, "results", "all_dialogues_parsed", "all_dialogues_parsed_json_only.json"))
#reorder_dialogues(os.path.join(main_path, "results", "all_dialogues_parsed", "all_dialogues_parsed_img_json.json"))

def combine_parsed_dialogues_single_inputs(dialogues_path, condition):
    combined_dialogues = []
    for input_condition in os.listdir(dialogues_path):
        if os.path.isdir(os.path.join(dialogues_path, input_condition)) and input_condition == condition:
            for file in os.listdir(os.path.join(dialogues_path, input_condition)):
                with open(os.path.join(dialogues_path, input_condition, file), "r") as f:
                    data = json.load(f)
                    combined_dialogues.append(data)
    output_path = os.path.join(main_path, "results", "all_dialogues_parsed", f"all_dialogues_parsed_{condition}.json")
    with open(output_path, "w") as f_out:
        json.dump(combined_dialogues, f_out, indent=2)

#dialogues_path = os.path.join(main_path, "results", "dialogues_parsed")
#combine_parsed_dialogues_single_inputs(dialogues_path, "img_only")
#combine_parsed_dialogues_single_inputs(dialogues_path, "json_only")
#combine_parsed_dialogues_single_inputs(dialogues_path, "img_json")


'''# Load your original nested JSON file
with open("/home/xmakaco/cimec/MLLMs-construction-company-main-new/results/all_dialogues_parsed/all_dialogues_parsed.json", "r") as infile:
    raw_data = json.load(infile)

# Prepare the flattened list
flat_data = []

for entry in raw_data:
    structure_id = entry.get("structure_id", "unknown_id")
    dialogue = entry.get("dialogue", [])

    # Turn the list of dialogue turns into a single string
    dialogue_lines = []
    for turn in dialogue:
        for speaker, utterance in turn.items():
            dialogue_lines.append(f"{speaker}: {utterance.strip()}")

    dialogue_text = "\n".join(dialogue_lines)

    # Build the flat item
    flat_item = {
        "structure_id": structure_id,
        "dialogue_id": f"{structure_id}_0",
        "dialogue_text": dialogue_text
    }

    flat_data.append(flat_item)

# Save to a new JSON file ready for Label Studio
with open("labelstudio_ready.json", "w") as outfile:
    json.dump(flat_data, outfile, indent=2)

#print("Conversion complete. Saved to 'labelstudio_ready.json'.")'''


def extract_unevaluated_dialogues(input_path, output_path):
    with open(input_path, "r") as infile:
        data = json.load(infile)

    unevaluated = []
    for item in data:
        if item.get("judge_evaluation") == "Undefined":
            unevaluated.append({
                "structure_id": item["structure_id"],
                "condition": item["condition"],
                "dialogue": item["dialogue"]
            })

    with open(output_path, "w") as outfile:
        json.dump(unevaluated, outfile, indent=2)

# Example usage:
input_json = "/home/xmakaco/cimec/MLLMs-construction-company-main-new/results/all_dialogues_judged/more_unevaluated_dialogues_judged.json"
output_json = "last_unevaluated_dialogue.json"
extract_unevaluated_dialogues(input_json, output_json)





                
            
    



    