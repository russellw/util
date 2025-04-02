import tkinter as tk
from tkinter import Menu


def update_menu():
    # Check if text is selected for Cut/Copy
    try:
        # Returns an empty string if no text is selected
        selected_text = text.get(tk.SEL_FIRST, tk.SEL_LAST)
        edit_menu.entryconfig("Cut", state=tk.NORMAL)
        edit_menu.entryconfig("Copy", state=tk.NORMAL)
    except tk.TclError:
        edit_menu.entryconfig("Cut", state=tk.DISABLED)
        edit_menu.entryconfig("Copy", state=tk.DISABLED)

    # Check clipboard for Paste functionality
    try:
        # This will raise an exception if the clipboard is empty or not text
        if root.clipboard_get():
            edit_menu.entryconfig("Paste", state=tk.NORMAL)
    except tk.TclError:
        edit_menu.entryconfig("Paste", state=tk.DISABLED)


def cut():
    text.event_generate("<<Cut>>")
    update_menu()


def copy():
    text.event_generate("<<Copy>>")
    update_menu()


def paste():
    text.event_generate("<<Paste>>")
    update_menu()


root = tk.Tk()
root.title("Text Editor")

text = tk.Text(root, width=40, height=10)
text.pack(padx=10, pady=10)

# Create a menu bar with an Edit menu
menu_bar = Menu(root)
root.config(menu=menu_bar)

edit_menu = Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Edit", menu=edit_menu)

# Add menu items with corresponding commands
edit_menu.add_command(label="Cut", command=cut)
edit_menu.add_command(label="Copy", command=copy)
edit_menu.add_command(label="Paste", command=paste)

# Initially update the menu items based on the current state
update_menu()

# Bind to update menu items whenever there is a change in the Text widget
text.bind("<<Selection>>", lambda event: update_menu())
text.bind("<<Modified>>", lambda event: update_menu())

root.mainloop()
