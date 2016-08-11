from Tkinter import *

def onclick():
   pass

root = Tk()
text = Text(root)
text.insert(INSERT, "Hello.....")
text.insert(END, "Bye Bye.....")
text.pack()

text.tag_add("here", "1.0", "1.4")
text.tag_config("here", background="yellow", foreground="blue")
text.tag_add("here", "1.8", "1.13")
text.tag_config("here", background="black", foreground="green")


text.tag_config("tt", background="white", foreground="red")
start_pos = text.index(END)
text.insert(END, "\nI am new line.")
print("start: " + start_pos)
stop_pos = text.index(INSERT)
print("stop: " + stop_pos)
text.tag_add("tt", start_pos, stop_pos)



root.mainloop()