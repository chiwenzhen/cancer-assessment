# coding=utf-8
import Tkinter as Tk
import numpy as np
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2, f_classif


class FeaturesCorrFrame(Tk.Frame):
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

        # 背景色白色
        sns.set(style="white")
        # 特征关联矩阵（矩阵里不仅包含特征，还包括类别）
        corr = df.corr()
        # 隐藏矩阵的上三角
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        # 画图
        f, ax = plt.subplots(figsize=(11, 11))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
        plt.xticks(rotation=-90)
        plt.yticks(rotation=0)
        plt.title("Cardiotocography \"Feature-Feature\" & \"Feature-Label\" Correlations")
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
