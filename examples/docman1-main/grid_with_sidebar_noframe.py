import tkinter as tk
from tkinter import Canvas, Frame, Label, Menu, Scrollbar, Toplevel

from PIL import Image, ImageTk


class DynamicIconGridApp:
    def __init__(self, root, items, padding=10, cell_width=200, cell_height=100):
        self.root = root
        self.items = items
        self.padding = padding
        self.cell_width = cell_width
        self.cell_height = cell_height

        # Initialize menu
        self.menu = Menu(self.root)
        self.root.config(menu=self.menu)
        self.file_menu = Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="New", command=self.new_file)
        self.file_menu.add_command(label="Open", command=self.open_file)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.root.quit)

        # Setup the sidebar
        self.sidebar = Frame(root, width=200, bg="gray")
        self.sidebar.pack(side="left", fill="y")

        # Add some widgets to the sidebar
        label = Label(self.sidebar, text="Sidebar", bg="gray")
        label.pack(padx=10, pady=10)

        # Setup the scrolling canvas for the grid
        self.canvas = Canvas(root)
        self.scrollbar = Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.load_icons()  # Load icons
        self.root.after(100, self.populate_grid)
        self.root.bind("<Configure>", self.on_resize)

    def new_file(self):
        print("New File Created")  # Placeholder function

    def open_file(self):
        print("File Opened")  # Placeholder function

    def load_icons(self):
        for item in self.items:
            if "icon_path" in item:
                image = Image.open(item["icon_path"])
                image = image.resize(
                    (self.cell_width - 2 * self.padding, int(self.cell_height * 0.6)),
                    Image.Resampling.LANCZOS,
                )
                item["icon"] = ImageTk.PhotoImage(image)

    def calculate_columns(self):
        current_width = self.canvas.winfo_width()
        columns = max(1, current_width // (self.cell_width + 2 * self.padding))
        return columns

    def populate_grid(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        row, col = 0, 0
        columns = self.calculate_columns()

        for index, item in enumerate(self.items):
            if index % columns == 0 and index != 0:
                row += 1
                col = 0

            label = Label(
                self.scrollable_frame,
                text=item["name"],
                image=item.get("icon", None),
                compound="top",
                height=self.cell_height,
                width=self.cell_width,
                wraplength=150,
            )
            label.grid(
                row=row, column=col, sticky="nsew", padx=self.padding, pady=self.padding
            )

            col += 1

    def on_resize(self, event=None):
        if event and event.widget is self.root:
            self.populate_grid()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Dynamic Icon Grid with Menu and Sidebar")
    root.geometry("1000x600")  # Set initial size

    # Example items: Specify correct paths to your icon images
    items = [
        {"name": "Item 1", "icon_path": "/t/a.png"},
        {"name": "Item 2", "icon_path": "/t/a.png"},
        {
            "name": "Item 3 Using a Frame as a container provides a layer of abstraction and control that is beneficial for more complex or visually structured interfaces.",
            "icon_path": "/t/a.png",
        },
        {"name": "Item 4", "icon_path": "/t/a.png"},
        {"name": "Item 5", "icon_path": "/t/a.png"},
        {"name": "Item 6", "icon_path": "/t/a.png"},
        {"name": "Item 7", "icon_path": "/t/a.png"},
        {"name": "Item 8", "icon_path": "/t/a.png"},
        {"name": "Item 9", "icon_path": "/t/a.png"},
    ]

    app = DynamicIconGridApp(root, items)
    root.mainloop()
