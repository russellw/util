import subprocess
import time
from pywinauto.application import Application

app = Application(backend="uia").connect(title="WhatsApp")
notepad = app.window(title='WhatsApp')

# Step 3: Output the window handle
print('Window found:', notepad)

# Step 4: Enumerate and print all child windows
for child in notepad.descendants():
    print(f"Control - Type: {child.friendly_class_name()} , Name: {child.window_text()}")
