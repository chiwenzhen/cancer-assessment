# coding=utf-8
import Tkinter as Tk
import numpy as np
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2, f_classif


class FeaturesPairFrame(Tk.Frame):
    def __init__(self, master, x_train, y_train, x_test, y_test, evaluator, df, console):
        Tk.Frame.__init__(self, master)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.evaluator = evaluator
        self.df = df
        self.console = console

        frame_train = Tk.Frame(self)
        frame_train.pack(fill=Tk.BOTH, expand=1, padx=15, pady=15)
        plt.figure(figsize=(12, 20))
        plt.subplot(111)
        # k best feature's names
        selection = SelectKBest(f_classif, k=3)
        selection.fit(self.x_train, self.y_train)
        feature_names = df.columns.values
        feature_names = feature_names[feature_names != "NSP"]
        kbest_feature_indexes = selection.get_support()
        kbest_feature_names = feature_names[kbest_feature_indexes]

        _ = sns.pairplot(self.df[:200], vars=kbest_feature_names, hue="NSP", size=2.8)
        plt.title("Cardiotocography Relations between Feature Pair (3x3) and Label", loc='center')
        self.attach_figure(plt.gcf(), frame_train)

    # 将figure放到frame上
    @staticmethod
    def attach_figure(figure, frame):
        canvas = FigureCanvasTkAgg(figure, master=frame)  # 内嵌散点图到UI
        canvas.show()
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        toolbar = NavigationToolbar2TkAgg(canvas, frame)  # 内嵌散点图工具栏到UI
        toolbar.update()
        canvas.tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
