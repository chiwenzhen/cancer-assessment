# coding=utf-8
import numpy as np
import Tkinter as Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from sklearn.learning_curve import validation_curve


class ValidationCurveFrame(Tk.Frame):
    def __init__(self, master, x_train, y_train, x_test, y_test, evaluator):
        Tk.Frame.__init__(self, master)

        param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        train_scores, test_scores = validation_curve(estimator=evaluator.pipeline,
                                                     X=x_train, y=y_train,
                                                     param_name='clf__gamma',
                                                     param_range=param_range, cv=10)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        frame_vcurve = Tk.Frame(self)
        frame_vcurve.pack(fill='x', expand=1, padx=15, pady=15)
        figure_vcurve = Figure(figsize=(6, 6), dpi=100)
        subplot_vcurve = figure_vcurve.add_subplot(111)

        subplot_vcurve.plot(param_range, train_mean, color='blue', marker='o', markersize=5,
                            label='training accuracy')
        subplot_vcurve.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15,
                                    color='blue')
        subplot_vcurve.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5,
                            label='cross-validation accuracy')
        subplot_vcurve.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15,
                                    color='green')

        subplot_vcurve.grid()
        subplot_vcurve.set_xscale('log')
        subplot_vcurve.legend(loc='lower right')
        subplot_vcurve.set_xlabel('Parameter C')
        subplot_vcurve.set_ylabel('Accuracy')
        subplot_vcurve.set_ylim([0.91, 1.0])
        self.attach_figure(figure_vcurve, frame_vcurve)

    @staticmethod
    def attach_figure(figure, frame):
        canvas = FigureCanvasTkAgg(figure, master=frame)  # 内嵌散点图到UI
        canvas.show()
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        toolbar = NavigationToolbar2TkAgg(canvas, frame)  # 内嵌散点图工具栏到UI
        toolbar.update()
        canvas.tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
