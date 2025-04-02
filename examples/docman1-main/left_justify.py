import tkinter as tk
from tkinter import Frame, Menu


def create_window():
    # Create the main window
    window = tk.Tk()
    window.title("Sample Menu and Sidebar Window")
    window.geometry("500x300")
    window.config(bg="light gray")  # Setting the background color of the main window

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

    # Create a sidebar
    sidebar = Frame(window, bg="light gray", borderwidth=2, relief="flat")
    sidebar.pack(expand=False, fill="y", side="left", anchor="nw")

    # Vertical line to separate the sidebar from the main area
    line = Frame(sidebar, bg="black", width=1)
    line.pack(side="right", fill="y")

    # Add buttons to the sidebar that look like hyperlinks and are left-justified
    button1 = tk.Button(
        sidebar,
        text="Button 1",
        relief="flat",
        fg="blue",
        cursor="hand2",
        command=lambda: print("Button 1 clicked"),
    )
    button1.pack(side="top", anchor="w", padx=(10, 0), pady=5)
    button2 = tk.Button(
        sidebar,
        text="Button 2",
        relief="flat",
        fg="blue",
        cursor="hand2",
        command=lambda: print("Button 2 clicked"),
    )
    button2.pack(side="top", anchor="w", padx=(10, 0), pady=5)
    button3 = tk.Button(
        sidebar,
        text="Button 33333333",
        relief="flat",
        fg="blue",
        cursor="hand2",
        command=lambda: print("Button 3 clicked"),
    )
    button3.pack(side="top", anchor="w", padx=(10, 0), pady=5)

    return window


# Create and run the window
if __name__ == "__main__":
    root = create_window()
    root.mainloop()
