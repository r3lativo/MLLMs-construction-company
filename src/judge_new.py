from utils import main_path
from judge_utils_new import *
import json
import os

'''if __name__ == "__main__":
    input_path = os.path.join(main_path, "results", "all_dialogues_parsed", "all_dialogues_parsed.json")
    output_path = os.path.join(main_path, "results", "all_dialogues_judged", "all_dialogues_judged.json")
    
    with open(input_path, "r") as f:
        dialogues = json.load(f)

    judged_dialogues = run_judge(dialogues, command="BASE")

    with open(output_path, "w") as f:
        json.dump(judged_dialogues, f, indent=2)'''

if __name__ == "__main__":
    input_path = os.path.join(main_path, "results", "all_dialogues_judged", "last_unevaluated_dialogue.json")
    output_path = os.path.join(main_path, "results", "all_dialogues_judged", "last_unevaluated_dialogue_judged.json")

    run_judge(input_path, command="BASE", output_path=output_path)

