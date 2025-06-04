from utils import main_path
import os
import pandas as pd
import json
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import re


def get_system_prompt(json_file, cmd):
    with open(json_file, "r", encoding = "utf-8") as file:
        all_prompts = json.load(file)
        
    prompt = ""
    for key, value in all_prompts.items():
        if key == cmd:
            prompt = value
            prompt = str(prompt).split(":")[1]
            prompt = prompt[:-1]
            break
    return prompt

def get_system_examples(json_file, cmd):
    with open(json_file, "r", encoding="utf-8") as file:
        all_examples = json.load(file)

    examples = all_examples.get(cmd, "")

    if isinstance(examples, list):
        return "\n\n".join(json.dumps(ex, indent=2) if isinstance(ex, dict) else str(ex) for ex in examples)
    elif isinstance(examples, dict):
        return json.dumps(examples, indent=2)
    else:
        return str(examples)

def ask_judge(evaluation_type, conversation):
  
  # Load the apikey
  '''env_path = os.path.join(main_path, '.env')
  load_dotenv(dotenv_path=env_path)'
  deepseek_api = os.environ.get('DEEPSEEK_API_KEY')'''

  # Load files
  judge_data_path = os.path.join(main_path, "data", "judge_data")
  prompts_file = os.path.join(judge_data_path, "system_prompts.json")
  examples_file = os.path.join(judge_data_path, "system_examples.json")

  # Load prompt and example give the evaluation type
  evaluation_prompt = get_system_prompt(prompts_file, evaluation_type)
  evaluation_examples = get_system_examples(examples_file, evaluation_type)

  # Load Deepseek
  client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
  )
  
  # Create request
  completion = client.chat.completions.create(
    model="deepseek/deepseek-r1:free",
    messages=[
      {
          "role": "system",
          "content": evaluation_prompt,
      },
      {
          "role": "system",
          "content": "Your examples are: " + evaluation_examples,
      },
      {
        "role": "user",
        "content": conversation,
      }
    ]
  )
  try:
      return completion.choices[0].message.content
  except TypeError:
      return "[Empty answer]"

def dialogue_to_text(dialogue):
    """
    Converts a list of messages (each a dict with "Architect" or "Builder") into a single string
    suitable for DeepSeek input.
    """
    lines = []
    for turn in dialogue:
        if "Architect" in turn:
            lines.append(f"Architect: {turn['Architect']}")
        elif "Builder" in turn:
            lines.append(f"Builder: {turn['Builder']}")
    return "\n".join(lines)

dialogue_path = os.path.join(main_path, "results", "all_dialogues_parsed", "all_dialogues_parsed_json_only.json")
with open(dialogue_path, 'r', encoding='utf-8') as f:
    dialogue = json.load(f)
for d in dialogue:
    print(f"Structure: {d['structure_id']}")
    print(dialogue_to_text(d["dialogue"]))
    print("-" * 40) # For inspection

def run_judge(json_path, command, output_path=None):
    # Load the big JSON with many dialogues
    with open(json_path, 'r') as f:
        data = json.load(f)

    for dialogue_obj in tqdm(data):
        dialogue = dialogue_obj.get("dialogue", [])
        # Convert dialogue messages into a string for DeepSeek
        dialogue_text = dialogue_to_text(dialogue)

        # Call DeepSeek
        judge_output = ask_judge(command, dialogue_text)

        # Try to parse numeric score from output (last char)
        rating = "Undefined"
        try:
            rating = int(judge_output.strip()[-1])
        except Exception:
            pass

        # Insert judge evaluation BEFORE "dialogue" key:
        # Because dicts in Python 3.7+ preserve insertion order, recreate dict:
        new_dialogue_obj = {}
        # Insert keys before 'dialogue'
        for key in dialogue_obj:
            if key == "dialogue":
                new_dialogue_obj["judge_evaluation"] = str(rating)
                new_dialogue_obj["dialogue"] = dialogue_obj["dialogue"]
            else:
                new_dialogue_obj[key] = dialogue_obj[key]

        # Update the original dict (important to keep reference)
        dialogue_obj.clear()
        dialogue_obj.update(new_dialogue_obj)

    save_path = output_path if output_path is not None else json_path
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)

