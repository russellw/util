import tkinter as tk
from tkinter import filedialog, Text

def open_file_dialog():
    folder_selected = filedialog.askdirectory()
    cv_folder_entry.delete(0, tk.END)
    cv_folder_entry.insert(0, folder_selected)

def on_ok():
    print("CV Folder:", cv_folder_entry.get())
    print("API Key:", api_key_entry.get())
    print("Job Description:", job_description_text.get("1.0", tk.END))
    root.destroy()

def on_cancel():
    root.destroy()

root = tk.Tk()
root.title("Input Form")

# CV Folder
tk.Label(root, text="CV Folder:").grid(row=0, column=0, sticky='w', padx=10, pady=5)
cv_folder_entry = tk.Entry(root, width=50)
cv_folder_entry.grid(row=0, column=1, padx=10, pady=5)
tk.Button(root, text="Browse...", command=open_file_dialog).grid(row=0, column=2, padx=10, pady=5)

# API Key
tk.Label(root, text="API Key:").grid(row=1, column=0, sticky='w', padx=10, pady=5)
api_key_entry = tk.Entry(root, width=50)
api_key_entry.grid(row=1, column=1, columnspan=2, padx=10, pady=5)

# Job Description
tk.Label(root, text="Job Description:").grid(row=2, column=0, sticky='nw', padx=10, pady=5)
job_description_text = Text(root, height=5, width=38)
job_description_text.grid(row=2, column=1, columnspan=2, padx=10, pady=5)

# Buttons
tk.Button(root, text="OK", command=on_ok).grid(row=3, column=1, sticky='e', padx=10, pady=10)
tk.Button(root, text="Cancel", command=on_cancel).grid(row=3, column=2, sticky='w', padx=10, pady=10)

root.mainloop()
