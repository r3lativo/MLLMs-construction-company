"""This script was used to take in input .txt dialogues files from the Minecraft Dialogue Corpus, 
partially adjust their formatting, and convert them to .json files.
"""

import json
import sys


def parse_and_convert(txt_file, json_file):
    dialogue = []
    
    with open(txt_file, 'r', encoding='utf-8') as file:
        content = file.read()

    mod_content = content.replace("<", "").replace(">", ":")
    for line in mod_content.split("\n"):
        line = line.strip()
        if not line:
            continue
        
        if ":" in line:
            speaker, text = line.split(":", 1)
            dialogue.append({"speaker": speaker.strip(), "text": text.strip(), "annotation": []})
           
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump({"dialogue": dialogue}, file, indent=4, ensure_ascii=False)  # Convert txt to JSON

    print("Dialogue parsed and converted!")

parse_and_convert(sys.argv[1], sys.argv[2])


