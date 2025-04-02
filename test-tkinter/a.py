import tkinter as tk
from tkinter import filedialog, Text

def open_file_dialog():
    folder_selected = filedialog.askdirectory()
    cv_folder_entry.delete(0, tk.END)
    cv_folder_entry.insert(0, folder_selected)

root = tk.Tk()
root.title("Input Form")

# CV Folder
tk.Label(root, text="CV Folder:").grid(row=0, column=0, sticky='w')
cv_folder_entry = tk.Entry(root, width=50)
cv_folder_entry.grid(row=0, column=1)
#tk.Button(root, text="Browse...", command=open_file_dialog).grid(row=0, column=2)

# API Key
tk.Label(root, text="API Key:").grid(row=1, column=0, sticky='w')
api_key_entry = tk.Entry(root, width=50)
api_key_entry.grid(row=1, column=1, columnspan=2)

# Job Description
tk.Label(root, text="Job Description:").grid(row=2, column=0, sticky='nw')
job_description_text = Text(root, height=5, width=38)
job_description_text.grid(row=2, column=1, columnspan=2)

root.mainloop()
