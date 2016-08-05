# coding=utf-8
import numpy as np
import Tkinter as Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from sklearn.learning_curve import learning_curve


class LearningCurveFrame(Tk.Frame):
    def __init__(self, master, x_train, y_train, x_test, y_test, evaluator):
        Tk.Frame.__init__(self, master)
        train_sizes, train_scores, test_scores = learning_curve(estimator=evaluator.pipeline,
                                                                X=x_train,
                                                                y=y_train,
                                                                train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        frame_lcurve = Tk.Frame(self)
        frame_lcurve.pack(fill="x", expand=1, padx=15, pady=15)
        figure_lcurve = Figure(figsize=(6, 6), dpi=100)
        subplot_lcurve = figure_lcurve.add_subplot(111)
        subplot_lcurve.plot(train_sizes, train_mean, color="blue", marker='o', markersize=5,
                            label="training accuracy")
        subplot_lcurve.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15,
                                    color="blue")
        subplot_lcurve.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5,
                            label="cross-validation accuracy")
        subplot_lcurve.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15,
                                    color="green")
        subplot_lcurve.grid()
        subplot_lcurve.set_xlabel("Number of training samples")
        subplot_lcurve.set_ylabel("Accuracy")
        subplot_lcurve.legend(loc="lower right")
        subplot_lcurve.set_ylim([0.8, 1.0])
        self.attach_figure(figure_lcurve, frame_lcurve)

    @staticmethod
    def attach_figure(figure, frame):
        canvas = FigureCanvasTkAgg(figure, master=frame)  # 内嵌散点图到UI
        canvas.show()
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        toolbar = NavigationToolbar2TkAgg(canvas, frame)  # 内嵌散点图工具栏到UI
        toolbar.update()
        canvas.tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)