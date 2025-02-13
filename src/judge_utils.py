from utils import main_path
import os
import pandas as pd
import json
from datetime import datetime
from openai import OpenAI
import sys
from tqdm import tqdm

results_path = os.path.join(main_path, "results")


def load_all_results(results_path):
    """Load all results into a DataFrame with additional round statistics."""
    data = []
    
    for f in os.listdir(results_path):
        if f.lower().endswith('.log'):
            file_path = os.path.join(results_path, f)
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            if not lines:
                print(f"The file {file_path} is empty!")
                continue  # skip empty files

            # --- Extract info from the first line ---
            # Example first line:
            # 02-11 18:29 INFO     Model: llava-hf/llava-v1.6-mistral-7b-hf, Quantization: 4-bit, Device: cuda, ...
            first_line = lines[0].strip()
            if "INFO" in first_line:
                # Get everything after "INFO"
                header = first_line.split("INFO", 1)[1].strip()
            else:
                header = first_line
            
            # --- Extract extra information from the filename ---
            run_time_str = f[:18]            # "2025-02-11-1829-56"
            structure_name = f[19:-4]          # everything after the underscore and before ".log"
            json_name = f[:-4] + ".json"

            # Setup dictionary
            info_dict = {}
            info_dict["run_time"] = run_time_str
            info_dict["structure_id"] = structure_name

            for pair in header.split(','):
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    info_dict[key.strip()] = value.strip()
            
            # --- Process the entire log to collect round timings and check for finishing marker ---
            round_timestamps = []
            finished_by_architect = False

            for line in lines:
                line = line.strip()
                # Check if the line indicates a round (e.g., "===== Round 1 =====")
                if "==== Round" in line:
                    # Extract timestamp: assume the first two tokens form the timestamp.
                    timpestamp = line[:11]
                    try:
                        dt = datetime.strptime(timpestamp, "%m-%d %H:%M")
                        round_timestamps.append(dt)
                    except ValueError as e:
                        print(f"Timestamp parsing error in file {f}: {e}")
                
                # Check for the finishing message
                if "Finishing conversation as indicated by Architect." in line:
                    finished_by_architect = True
            
            # --- Compute round statistics ---
            num_rounds = len(round_timestamps)
            total_round_time = 0  # in seconds
            
            if num_rounds > 1:
                # Total time is the difference between the first and last round timestamps
                total_time_delta = round_timestamps[-1] - round_timestamps[0]
                total_round_time = total_time_delta.total_seconds()
            elif num_rounds == 1:
                # Only one round logged; we set average round time to 0
                total_round_time = 0

            info_dict["num_rounds"] = num_rounds
            info_dict["total_time_min"] = (total_round_time / 60) if total_round_time != 0 else 1
            info_dict["finished_by_architect"] = finished_by_architect
            info_dict["json_file"] = json_name
            
            data.append(info_dict)

    df = pd.DataFrame(data)
    df['shot'] = df['shot'].fillna("zero-shot")

    return df


def parse_conv(conversation):
      final = ""
      for t in conversation:
        final += f"{t['speaker']}: {t['utterance']}\n"
      return final


def extract_conversation_data(json_path):
    """
    Extracts a combined conversation from the JSON file.
    
    For each message in the conversation:
      - If the speaker is "Architect": keep the full text.
      - If the speaker is "Builder": attempt to parse the text as a JSON dictionary and:
          - Retain only the 'feedback' (or use "[no_utterance]" if empty).
          - Also retain the 'add' and 'remove' actions, as well as the 'confidence' value.
    
    Returns:
      A list of dictionaries where each dictionary represents a turn in the conversation.
      For Builder turns, the dictionary has keys:
         - 'speaker': "Builder"
         - 'utterance': the builder feedback (or "[no_utterance]")
         - 'actions': a dictionary with keys "add" and "remove"
         - 'confidence': the confidence value (or None)
      For Architect turns, the dictionary has:
         - 'speaker': "Architect"
         - 'utterance': the full text from the Architect.
    """
    conversation_data = []

    try:
        with open(json_path, 'r') as f:
            conversation = json.load(f)
    except:
        print(f"COULD NOT FIND {json_path} FILE")
        return

    for msg in conversation:
        speaker = msg.get("speaker", "")
        # Collect all text parts from the content list.
        texts = []
        for part in msg.get("content", []):
            if part.get("type") == "text":
                texts.append(part.get("text", "").strip())
        combined_text = "\n".join(texts)

        if speaker == "Architect":
            # For the Architect, simply keep the full text.
            conversation_data.append({
                "speaker": speaker,
                "utterance": combined_text
            })
        elif speaker == "Builder":
            # For Builder, try to parse the text as JSON.
            try:
                builder_response = json.loads(combined_text)
                # Extract feedback; if it's missing or empty, use "[no_utterance]".
                feedback = builder_response.get("feedback", "").strip()
                if not feedback:
                    feedback = "[no_utterance]"
                # Extract actions if provided.
                actions = {
                    "add": builder_response.get("add", []),
                    "remove": builder_response.get("remove", [])
                }
                confidence = builder_response.get("confidence", None)

                conversation_data.append({
                    "speaker": speaker,
                    "utterance": feedback,
                    "actions": actions,
                    "confidence": confidence
                })
            except json.JSONDecodeError:
                # If parsing fails, use the raw text (or default feedback if empty)
                feedback = combined_text.strip() if combined_text.strip() else "[no_utterance]"
                conversation_data.append({
                    "speaker": speaker,
                    "utterance": feedback,
                    "actions": {},
                    "confidence": None
                })
        else:
            # Skip any other speaker (system).
            continue
        
    b_actions_list = [t["actions"] for t in conversation_data if t.get("speaker") == "Builder"]
    parsed_conversation = parse_conv(conversation_data)

    return parsed_conversation, b_actions_list


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
    #api_key="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",
    api_key="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",
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


def run_judge(command, results_df):

  # Name of the output JSON file
  output_file = os.path.join(main_path, "analysis", f"judge_analysis_{command}.json")

  # Check if output JSON file exists; if so, load existing results, else create an empty list.
  if os.path.exists(output_file):
    with open(output_file, "r") as f:
      results = json.load(f)
  else:
    results = []

  # Iterate over each row in the DataFrame
  total_rows = len(results_df)
  for index, row in tqdm(results_df.iterrows(), total=len(results_df)):

# Check if this index is already present in the JSON results.
    if any(item.get('index') == index for item in results):
        tqdm.write(f"Conversation {index} is already judged. Skipping...")
        continue

    tqdm.write(f"Processing row {index} of {total_rows-1}")
    
    # Open and load the JSON file
    json_path = os.path.join(results_path, row["json_file"])
    try:
        parsed_conversation, _ = extract_conversation_data(json_path)
    except:
        continue
    
    # Process the JSON data with ask_judge function
    judge_output = ask_judge(command, parsed_conversation)
    rating = "Undefined"

    try:
      rating = int(judge_output[-1])
      tqdm.write(f"Rating: {rating}")
    except:
      tqdm.write(f"Rating undefined - check json.")
      pass
    
    # Create a dictionary with desired columns and the judge result
    result = {
        'index': index,
        'structure_id': row['structure_id'],
        'num_rounds': row['num_rounds'],
        'total_time_min': row['total_time_min'],
        'finished_by_architect': row['finished_by_architect'],
        'use_img': row['use_img'],
        'use_json': row['use_json'],
        'shot': row['shot'],
        'judge': judge_output,
        'rating': rating,
    }
    
    # Append the result to our list
    results.append(result)
    
    # Update the JSON file with the new results
    with open(output_file, "w") as f:
      json.dump(results, f, indent=4)
