import json
import shutil
import sys

def parse_and_convert(txt_file, copy_file, json_file):
    shutil.copyfile(txt_file, copy_file)
    dialogue = []
    
    with open(copy_file, 'r', encoding='utf-8') as file:
        content = file.read()
        mod_content = content.replace("<", "").replace(">", ":") # Modify text file
        with open(copy_file, "w", encoding = "utf-8") as new:
            new.write(mod_content)

    print(f"Mod completed!")
        
    #with open(txt_file, 'r', encoding='utf-8') as file:
    for line in mod_content.split("\n"):
            print(line)
            line = line.strip()
            if not line:
                continue 
            
            if ":" in line:
                speaker, text = line.split(":", 1)
                dialogue.append({"speaker": speaker.strip(), "text": text.strip(), "annotation": []})
            
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump({"dialogue": dialogue}, file, indent=4, ensure_ascii=False) # Convert txt to JSON

    print(f"Dialogue converted!")

parse_and_convert(sys.argv[1], sys.argv[2], sys.argv[3])


