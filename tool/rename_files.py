import pandas as pd
import os
import re
import shutil
from pathlib import Path

CONFIG = {
    "class" : "1班",
    "student_list": 'student_list.txt',
    "folder_path": "",
    "target_folder_path": ""
}

name_id_dict = {}

def read_txt_file(file_path):
    data = pd.read_csv(file_path, delimiter='\t', header=None)
    result = {}

    for index, row in data.iterrows():
        student_id = row[0]
        str(student_id)
        name = row[1]
        result[name] = student_id

    return result

def match_by_name(file_name:str, name:str):
    is_match = re.search(name, file_name)
    return is_match

def rename_file(folder_path, dict):
    try:
        if not os.path.exists(CONFIG["target_folder_path"]):
            os.mkdir(CONFIG["target_folder_path"])
            print(f'create new folder: {CONFIG["target_folder_path"]}')

        for file_name in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, file_name)):
                for name in name_id_dict:
                    match = match_by_name(file_name, name)
                    if match:
                        shuffix = Path(file_name).suffix
                        re_name_to = os.path.join(
                            CONFIG["target_folder_path"],
                            CONFIG["class"] + str(dict[name])[-3:] + name + shuffix
                        )
                        shutil.copy(file_name, re_name_to)
        print(f'finish, new folder is in {CONFIG["target_folder_path"]}')

    except Exception as e:
        print(str(e))


if __name__ == '__main__':
    name_id_dict = read_txt_file(CONFIG["student_list"])
    rename_file(CONFIG["folder_path"], name_id_dict)





