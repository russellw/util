import os
import re

for file in os.listdir("."):
    if os.path.isfile(file):
        m = re.match(r"20(\d\d)-(\d\d)-(\d\d) - (.+)", file)
        if m:
            new = f"{m[1]}{m[2]}{m[3]} {m[4]}"
            os.rename(file, new)
            print(f"Renamed: {file} -> {new}")
