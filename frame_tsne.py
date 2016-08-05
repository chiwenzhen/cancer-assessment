# coding=utf-8
import numpy as np
import Tkinter as Tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


class TSNEFrame(Tk.Frame):
    def __init__(self, master, x_train, y_train, x_test, y_test, evaluator):
        Tk.Frame.__init__(self, master)
        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train, y_train)  # 经过中心化和归一化的训练数据
        y_train_s = y_train
        tsne = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        x_train_s = tsne.fit_transform(x_train_s)
        frame_tsne = Tk.Frame(self)
        frame_tsne.pack(fill='x', expand=1, padx=15, pady=15)
        figure_tsne = Figure(figsize=(6, 6), dpi=100)
        subplot_tsne = figure_tsne.add_subplot(111)
        subplot_tsne.scatter(x_train_s[:, 0], x_train_s[:, 1], c=y_train_s, cmap=plt.cm.get_cmap("Paired"))
        subplot_tsne.set_title("t-SNE on breast cancer data")
        self.attach_figure(figure_tsne, frame_tsne)

    @staticmethod
    def attach_figure(figure, frame):
        canvas = FigureCanvasTkAgg(figure, master=frame)  # 内嵌散点图到UI
        canvas.show()
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        toolbar = NavigationToolbar2TkAgg(canvas, frame)  # 内嵌散点图工具栏到UI
        toolbar.update()
        canvas.tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
