import tkinter as tk
from tkinter import Canvas, Frame, Label, Scrollbar


class DynamicGridApp:
    def __init__(self, root, items, padding=10, cell_width=200):
        self.root = root
        self.items = items
        self.padding = padding
        self.cell_width = cell_width

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

        self.root.after(100, self.populate_grid)
        self.root.bind("<Configure>", self.on_resize)

    def calculate_columns(self):
        # Ensure the width used for calculation is greater than the cell width
        current_width = (
            self.canvas.winfo_width()
        )  # Use the canvas width which is more reliable for grid layout
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

            frame = Frame(self.scrollable_frame, height=50, width=self.cell_width)
            frame.grid(row=row, column=col, padx=self.padding, pady=self.padding)
            frame.pack_propagate(False)
            label = Label(frame, text=item["name"])
            label.pack(fill="both", expand=True)

            col += 1

    def on_resize(self, event=None):
        if event and event.widget is self.root:
            self.populate_grid()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Dynamic Grid of Names")
    root.geometry("600x400")  # Set initial size

    items = [
        {"name": "Item 1"},
        {"name": "Item 2"},
        {"name": "Item 3"},
        {"name": "Item 4"},
    ]

    app = DynamicGridApp(root, items)
    root.mainloop()
