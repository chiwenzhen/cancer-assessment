from Tkinter import *
from ttk import *

root = Tk()

master = Frame(root) # create Frame in "root"
master.pack(fill=BOTH) # fill both sides of the parent


nb = Notebook(master) # create Notebook in "master"
nb.pack(fill=BOTH) # fill "master" but pad sides

master_foo = Frame(nb)
nb.add(master_foo, text="foo") # add tab to Notebook

master_bar = Frame(master)
nb.add(master_bar, text="bar")

# start the app
if __name__ == "__main__":
    master.mainloop() # call master's Frame.mainloop() method.