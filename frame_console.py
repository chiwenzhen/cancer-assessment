# coding=utf-8
import Tkinter as Tk
import time
import numpy as np
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2, f_classif


class ConsoleFrame(Tk.Frame):
    def __init__(self, master):
        Tk.Frame.__init__(self, master)

        scrollbar = Tk.Scrollbar(self)
        scrollbar.pack(side=Tk.RIGHT, fill=Tk.Y)

        self.console = Tk.Text(self, width=800, wrap='none', yscrollcommand=scrollbar.set)
        self.console.pack()

        scrollbar.config(command=self.console.yview)

        self.console.tag_config("kw", background="white", foreground="red")

    def output(self, kw, text):
        # write kw
        start_pos = self.console.index(Tk.END)
        self.console.insert(Tk.END, "\n")
        self.console.insert(Tk.END, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        self.console.insert(Tk.END, " " + kw)

        # colour the kw
        stop_pos = self.console.index(Tk.INSERT)
        self.console.tag_add("kw", start_pos, stop_pos)

        # write text
        self.console.insert(Tk.END, "\n")
        self.console.insert(Tk.END, text)
