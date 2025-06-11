from utils import main_path
from judge_utils_new import *
import json
import os


if __name__ == "__main__":
    input_path = os.path.join(main_path, "results", "all_dialogues_parsed", "all_dialogues_parsed.json")
    output_path = os.path.join(main_path, "results", "all_dialogues_judged", "all_dialogues_judged.json")

    run_judge(input_path, command="BASE", output_path=output_path)

