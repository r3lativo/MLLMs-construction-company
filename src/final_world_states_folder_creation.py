import os
from pathlib import Path
import re
import shutil
from utils import main_path

'''
This function copies the final world state of each structure in new_results, 
renames them with the structure id and moves them to a new folder called final_world_states
'''

def create_final_world_states_folder(new_results_path):
    structures_id = ""
    file_type = ".json"
    for dir_name in os.listdir(new_results_path):
        if os.path.isdir(os.path.join(new_results_path, dir_name)):
                stuctures_id = dir_name + file_type
        json_files = [
                f for f in os.listdir(os.path.join(new_results_path, dir_name))
                if os.path.isfile(os.path.join(new_results_path, dir_name, f)) and f.endswith('.json')
            ]

        json_files.sort(key = lambda x: int(re.findall(r'\d+', x)[0]) if re.findall (r'\d+', x) else 0)
        final_world_state = json_files[-1]
        final_world_state_path = os.path.join(new_results_path, dir_name, final_world_state)
        src_path = Path(final_world_state_path)
        dst_path = Path(os.path.join(main_path, "results", "final_world_states"))
        new_path = dst_path / stuctures_id
        shutil.copy2(src_path, new_path)


new_results_path = os.path.join(main_path, "results", "new_results")
create_final_world_states_folder(new_results_path)
