import tkinter as tk

root = tk.Tk()

tk.Label(root, text="Single:").grid(row=0, column=0, sticky='w')
cv_folder_entry = tk.Entry(root, width=50)
cv_folder_entry.grid(row=0, column=1)

tk.Label(root, text="Multi:").grid(row=1, column=0, sticky='nw')
job_description_text = tk.Text(root, height=5, width=38)
job_description_text.grid(row=1, column=1)

root.mainloop()
