import tkinter as tk
from tkinter import messagebox


def show_message(message):
    messagebox.showinfo("Message", message)


def setup_sidebar(parent):
    sidebar = tk.Frame(parent, bg="lightgrey", relief="sunken", borderwidth=2)
    sidebar.pack(expand=False, fill="y", side="left", anchor="nw")

    btn1 = tk.Button(
        sidebar,
        text="Show Message 1",
        anchor="w",
        command=lambda: show_message("Button 1 clicked!"),
    )
    btn1.pack(pady=10, padx=10, fill="x")

    btn2 = tk.Button(
        sidebar,
        text="Show Message 2",
        anchor="w",
        command=lambda: show_message("Button 2 clicked!"),
    )
    btn2.pack(pady=10, padx=10, fill="x")

    btn3 = tk.Button(
        sidebar,
        text="This is a much longer button label",
        anchor="w",
        command=lambda: show_message("Button 3 clicked!"),
    )
    btn3.pack(pady=10, padx=10, fill="x")


def main():
    root = tk.Tk()
    root.title("Tkinter GUI with Sidebar")
    root.geometry("400x500")

    setup_sidebar(root)

    root.mainloop()


if __name__ == "__main__":
    main()
