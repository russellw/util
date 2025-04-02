import tkinter as tk
from tkinter import Canvas, Frame, Label, Scrollbar

from PIL import Image, ImageTk


class DynamicIconGridApp:
    def __init__(self, root, items, padding=10, cell_width=200, cell_height=100):
        self.root = root
        self.items = items
        self.padding = padding
        self.cell_width = cell_width
        self.cell_height = cell_height

        # Setup the scrolling canvas
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

    def load_icons(self):
        for item in self.items:
            if "icon_path" in item:
                image = Image.open(item["icon_path"])
                image = image.resize(
                    (self.cell_width - 2 * self.padding, int(self.cell_height * 0.6)),
                    Image.Resampling.LANCZOS,
                )  # Updated resampling method
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

            frame = Frame(
                self.scrollable_frame, height=self.cell_height, width=self.cell_width
            )
            frame.grid(row=row, column=col, padx=self.padding, pady=self.padding)
            frame.pack_propagate(False)
            label = Label(
                frame, text=item["name"], image=item.get("icon", None), compound="top"
            )
            label.pack(fill="both", expand=True)

            col += 1

    def on_resize(self, event=None):
        if event and event.widget is self.root:
            self.populate_grid()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Dynamic Icon Grid")
    root.geometry("800x600")  # Set initial size

    # Example items: Specify correct paths to your icon images
    items = [
        {"name": "Item 1", "icon_path": "/t/a.png"},
        {"name": "Item 2", "icon_path": "/t/a.png"},
        {"name": "Item 3", "icon_path": "/t/a.png"},
        {"name": "Item 4", "icon_path": "/t/a.png"},
    ]

    app = DynamicIconGridApp(root, items)
    root.mainloop()
