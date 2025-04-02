import os
from pathlib import Path

for root,dirs,_ in os.walk(".", topdown=False):
    for d in dirs:
        try:
            os.rmdir(Path(root,d))
            print(Path(root,d))
        except OSError:
            pass
