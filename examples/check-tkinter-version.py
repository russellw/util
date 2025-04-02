import tkinter as tk

# Create a Tk root widget, which is a window with a title bar and other decoration provided by the OS
root = tk.Tk()

# Print the version of Tkinter
print("Tkinter version:", tk.TkVersion)

# Print the version of the underlying Tcl interpreter
print("Tcl version:", tk.TclVersion)

# Destroy the root window
root.destroy()
