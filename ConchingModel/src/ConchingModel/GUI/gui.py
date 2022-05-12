import tkinter as tk

from ConchingModel.GUI import gui_lib

root = tk.Tk()
root.title("Experiment Setup")

mainframe = gui_lib.MainFrame(master=root)

root.mainloop()
