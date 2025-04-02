import os
import re

for file in os.listdir("."):
    if os.path.isfile(file):
        if re.match(r"20\d\d\d\d\d\d", file):
            new_name = file[2:]
            os.rename(file, new_name)
            print(f"Renamed: {file} -> {new_name}")
