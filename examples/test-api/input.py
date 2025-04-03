import tkinter as tk
from tkinter import scrolledtext

def main():
    # Create the main window
    root = tk.Tk()
    root.title("Paragraph Input Box")

    # Create a label
    label = tk.Label(root, text="Please enter a paragraph of text:")
    label.pack(pady=10)

    # Create a scrolled text widget
    text_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=10)
    text_box.pack(pady=10)

    # Define the submit function
    def submit(event=None):
        paragraph = text_box.get("1.0", tk.END).strip()
        print("User entered paragraph:", paragraph)
        # Copy text to clipboard
        root.clipboard_clear()
        root.clipboard_append(paragraph)
        root.update()  # Keep the clipboard contents after the window is closed
        print("Text copied to clipboard")

    # Create a submit button
    submit_button = tk.Button(root, text="Submit", command=submit)
    submit_button.pack(pady=10)

    # Bind the F12 key to the submit function
    root.bind('<F12>', submit)

    # Run the application
    root.mainloop()

if __name__ == "__main__":
    main()
