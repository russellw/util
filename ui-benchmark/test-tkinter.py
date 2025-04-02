import tkinter as tk
import time

def insert_alphabet():
    text_field.delete(1.0, tk.END)  # Clear the text field before inserting
    start_time = time.time()
    for _ in range(10):  # Repeat the insertion process 10 times
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            text_field.insert(tk.END, letter + ' ')
            text_field.update()  # Update the GUI to reflect the changes
            show_suggestion_box()  # Show suggestion box after each insertion
    end_time = time.time()
    elapsed_time = end_time - start_time
    result_label.config(text=f"Total time taken: {elapsed_time:.4f} seconds")

def show_suggestion_box(event=None):
    # Get the position of the cursor
    cursor_index = text_field.index(tk.INSERT)
    x, y, _, _ = text_field.bbox(cursor_index)
    
    # Convert the text box coordinates to root window coordinates
    x_root = x + text_field.winfo_rootx()
    y_root = y + text_field.winfo_rooty()

    # Create the suggestion box if it doesn't exist
    if not hasattr(show_suggestion_box, 'suggestion_box') or not show_suggestion_box.suggestion_box.winfo_exists():
        show_suggestion_box.suggestion_box = tk.Toplevel(root)
        show_suggestion_box.suggestion_box.wm_overrideredirect(True)
        show_suggestion_box.suggestion_box.wm_geometry(f"+{x_root}+{y_root + 20}")  # Offset below the cursor position
        
        suggestions = ["1. Lenovo", "2. HP", "3. Dell", "4. Apple", "5. Asus"]
        for suggestion in suggestions:
            label = tk.Label(show_suggestion_box.suggestion_box, text=suggestion, anchor="w")
            label.pack(fill=tk.BOTH)

    else:
        # If the suggestion box already exists, just update its position
        show_suggestion_box.suggestion_box.wm_geometry(f"+{x_root}+{y_root + 20}")

# Create the main window
root = tk.Tk()
root.title("Typing Assist Program Benchmark")

# Create a larger Text widget
text_field = tk.Text(root, height=20, width=80, wrap=tk.WORD)
text_field.pack(pady=10)

# Bind the key release event to show the suggestion box
text_field.bind("<KeyRelease>", show_suggestion_box)

# Create a button to start the insertion
start_button = tk.Button(root, text="Start Insertion", command=insert_alphabet)
start_button.pack(pady=10)

# Create a label to display the result
result_label = tk.Label(root, text="")
result_label.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()

