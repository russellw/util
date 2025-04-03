import tkinter as tk
from tkinter import Menu
import etc

etc.init_db()

def create_window():
    # Create the main window
    window = tk.Tk()
    window.title("Sample Menu Window")
    window.geometry("400x200")

    # Create a menu bar
    menubar = Menu(window)
    window.config(menu=menubar)

    # Add menu items
    file_menu = Menu(menubar, tearoff=0)
    file_menu.add_command(label="New")
    file_menu.add_command(label="Open...")
    file_menu.add_command(label="Save")
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=window.quit)
    menubar.add_cascade(label="File", menu=file_menu)

    # Add another menu
    edit_menu = Menu(menubar, tearoff=0)
    edit_menu.add_command(label="Undo")
    edit_menu.add_command(label="Redo")
    menubar.add_cascade(label="Edit", menu=edit_menu)

    return window


root = create_window()
root.mainloop()
