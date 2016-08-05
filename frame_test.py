# coding=utf-8
import numpy as np
import Tkinter as Tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score


class TestFrame(Tk.Frame):
    def __init__(self, master, x_train, y_train, x_test, y_test, evaluator):
        Tk.Frame.__init__(self, master)
        self.evaluator = evaluator
        self.evaluator.load_data(x_train, y_train, x_test, y_test)
        self.evaluator.train()
        x_test_r = self.evaluator.reduce(x_test)  # 特征降维

        frame_test = Tk.Frame(self)
        frame_test.pack(fill='x', expand=1, padx=15, pady=15)
        figure_test = Figure(figsize=(4, 4), dpi=100)
        subplot_test = figure_test.add_subplot(111)
        subplot_test.set_title('Breast Cancer Testing')
        figure_test.tight_layout()

        h = .02  # step size in the mesh
        x1_min, x1_max = x_test_r[:, 0].min() - 1, x_test_r[:, 0].max() + 1
        x2_min, x2_max = x_test_r[:, 1].min() - 1, x_test_r[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
        yy = self.evaluator.clf.predict(np.c_[xx1.ravel(), xx2.ravel()])
        yy = yy.reshape(xx1.shape)
        subplot_test.contourf(xx1, xx2, yy, cmap=plt.cm.get_cmap("Paired"), alpha=0.8)
        subplot_test.scatter(x_test_r[:, 0], x_test_r[:, 1], c=y_test, cmap=plt.cm.get_cmap("Paired"))
        self.attach_figure(figure_test, frame_test)

        # 第5.1页 测试性能指标 precision recall f_value
        y_pred = self.evaluator.pipeline.predict(x_test)
        frame_matrix = Tk.Frame(self)
        frame_matrix.pack(side=Tk.LEFT, fill='x', expand=1, padx=15, pady=15)
        figure_matrix = Figure(figsize=(4, 4), dpi=100)
        subplot_matrix = figure_matrix.add_subplot(111)

        confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
        subplot_matrix.matshow(confmat, cmap=plt.cm.get_cmap("Blues"), alpha=0.3)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                subplot_matrix.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

        subplot_matrix.set_xlabel('predicted label')
        subplot_matrix.set_ylabel('true label')
        self.attach_figure(figure_matrix, frame_matrix)

        frame_result = Tk.Frame(self)
        frame_result.pack(side=Tk.LEFT, fill='x', expand=1, padx=15, pady=15)
        Tk.Label(frame_result, text="Accuracy: ").grid(row=0, column=0, sticky=Tk.W)
        Tk.Label(frame_result, text=str(self.evaluator.pipeline.score(x_test, y_test))).grid(row=0, column=1,
                                                                                             sticky=Tk.W)
        Tk.Label(frame_result, text="Precision: ").grid(row=1, column=0, sticky=Tk.W)
        Tk.Label(frame_result, text=str(precision_score(y_true=y_test, y_pred=y_pred))).grid(row=1, column=1,
                                                                                             sticky=Tk.W)
        Tk.Label(frame_result, text="Recall: ").grid(row=2, column=0, sticky=Tk.W)
        Tk.Label(frame_result, text=str(recall_score(y_true=y_test, y_pred=y_pred))).grid(row=2, column=1,
                                                                                          sticky=Tk.W)
        Tk.Label(frame_result, text="F-value: ").grid(row=3, column=0, sticky=Tk.W)
        Tk.Label(frame_result, text=str(f1_score(y_true=y_test, y_pred=y_pred))).grid(row=3, column=1, sticky=Tk.W)

    @staticmethod
    def attach_figure(figure, frame):
        canvas = FigureCanvasTkAgg(figure, master=frame)  # 内嵌散点图到UI
        canvas.show()
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        toolbar = NavigationToolbar2TkAgg(canvas, frame)  # 内嵌散点图工具栏到UI
        toolbar.update()
        canvas.tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
