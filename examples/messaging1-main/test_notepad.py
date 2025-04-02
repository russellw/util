import subprocess
import time
from pywinauto.application import Application

# Step 1: Open Notepad
subprocess.Popen('notepad.exe')

# Step 2: Connect to the window
time.sleep(1)  # Allow some time for the application to start
app = Application(backend="uia").connect(title="Untitled - Notepad")
notepad = app.window(title='Untitled - Notepad')

# Step 3: Output the window handle
print('Window found:', notepad)

# Step 4: Enumerate and print all child windows
for child in notepad.descendants():
    print(f"Control - Type: {child.friendly_class_name()} , Name: {child.window_text()}")
