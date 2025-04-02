import ctypes
import tkinter as tk
from io import BytesIO
from tkinter import (Button, Canvas, Frame, Label, Menu, Scrollbar, filedialog,
                     simpledialog)

import etc

# Set DPI awareness
ctypes.windll.shcore.SetProcessDpiAwareness(2)

documents = []
with etc.conn:
    cursor = etc.conn.cursor()
    try:
        cursor.execute("SELECT id, name, icon256 FROM documents")
        rs = cursor.fetchall()
        for r in rs:
            doc = etc.Document()
            doc.ID = r[0]
            doc.name = r[1]
            doc.icon = r[2]
            documents.append(doc)
    finally:
        cursor.close()

# Create the main window
root = tk.Tk()
root.title("Document Management System")
root.geometry("900x600")
root.state("zoomed")

# Create a menu bar
menubar = Menu(root)
root.config(menu=menubar)

# File
file_menu = Menu(menubar, tearoff=0)


def new_folder():
    name = simpledialog.askstring(title="New folder", prompt="Folder name:")
    if name:
        with etc.conn:
            cursor = etc.conn.cursor()
            try:
                cursor.execute("insert into folders(name) values(?)", (name,))
            finally:
                cursor.close()


def import_file():
    file = etc.norm(filedialog.askopenfilename())
    if file:
        with etc.conn:
            cursor = etc.conn.cursor()
            try:
                doc = etc.import_file(file, cursor)
                documents.append(doc)
                refresh()
                canvas.update_idletasks()
            finally:
                cursor.close()


file_menu.add_command(label="New folder", underline=0, command=new_folder)
file_menu.add_command(label="Import", underline=0, command=import_file)
file_menu.add_separator()
file_menu.add_command(label="Exit", underline=1, command=root.quit)
menubar.add_cascade(label="File", menu=file_menu)

# Edit
edit_menu = Menu(menubar, tearoff=0)
edit_menu.add_command(label="Undo")
edit_menu.add_command(label="Redo")
menubar.add_cascade(label="Edit", menu=edit_menu)

# Search
search_menu = Menu(menubar, tearoff=0)
menubar.add_cascade(label="Search", menu=search_menu)

# Create a sidebar
sidebar = Frame(root)
sidebar.pack(fill="y", side="left")

# Vertical line to separate the sidebar from the main area
line = Frame(sidebar, bg="dark gray")
line.pack(side="right", fill="y")

# Add buttons to the sidebar that look like hyperlinks
button = Button(
    sidebar,
    text="Documents",
    command=lambda: print("Button 1 clicked"),
    cursor="hand2",
    relief="flat",
    padx=10,
    pady=10,
    anchor="w",
)
button.pack(fill="x")

button = Button(
    sidebar,
    text="Messages",
    command=lambda: print("Button 2 clicked"),
    cursor="hand2",
    relief="flat",
    padx=10,
    pady=10,
    anchor="w",
)
button.pack(fill="x")

# Setup the scrolling canvas for the grid
def on_mouse_wheel(event):
    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


def refresh(event=None):
    for widget in scrollable_frame.winfo_children():
        widget.destroy()

    cell_width = etc.icon_size
    text_height = 72
    cell_height = cell_width + text_height
    padding = 10
    columns = max(1, canvas.winfo_width() // (cell_width + 2 * padding))

    for i, doc in enumerate(documents):
        row = i // columns
        col = i % columns

        frame = Frame(scrollable_frame, height=cell_height, width=cell_width)
        frame.grid(row=row, column=col, padx=padding, pady=padding)
        frame.pack_propagate(False)

        # Image
        img_label = Label(frame, image=doc.photo, height=cell_width)
        img_label.pack(side="top", fill="x")

        # Canvas for text
        text_canvas = Canvas(frame, width=cell_width, height=text_height)
        text_canvas.pack(side="top", fill="x")

        # Create text with a slight padding from the left edge to avoid clipping
        padding = 2
        text_id = text_canvas.create_text(
            padding, 0, text=doc.name, anchor="nw", width=cell_width - 2 * padding
        )

        # Optional: Dynamically adjust the canvas height to clip text instead of showing all
        def adjust_canvas_height(event):
            text_canvas.config(
                height=min(72, event.height)
            )  # Example fixed max height of 72

        text_canvas.bind("<Configure>", adjust_canvas_height)


canvas = Canvas(root)
scrollbar = Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = Frame(canvas)

scrollable_frame.bind(
    "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

canvas.bind_all("<MouseWheel>", on_mouse_wheel)
canvas.bind("<Configure>", refresh)

root.mainloop()
