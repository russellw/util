import tkinter as tk

class ScrollingConsole:
    def __init__(self, master):
        self.master = master
        self.text = tk.Text(master, height=15, width=50)
        self.scrollbar = tk.Scrollbar(master, command=self.text.yview)
        self.text.configure(yscrollcommand=self.scrollbar.set)

        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def add_text(self, new_text):
        self.text.insert(tk.END, new_text + '\n')
        self.text.see(tk.END)  # Scroll to the bottom

def main():
    root = tk.Tk()
    console = ScrollingConsole(root)
    for i in range(100):  # Example: Adding multiple lines of text
        console.add_text(f'Line {i + 1}')
    root.mainloop()

if __name__ == '__main__':
    main()
