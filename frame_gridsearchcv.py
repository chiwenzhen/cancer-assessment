# coding=utf-8
import numpy as np
import Tkinter as Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from tabulate import tabulate


class GridSearchCVFrame(Tk.Frame):
    def __init__(self, master, x_train, y_train, x_test, y_test, evaluator):
        Tk.Frame.__init__(self, master)

        evaluator_gs = evaluator
        evaluator_gs.pipeline.named_steps['clf'] = SVC(random_state=1)
        frame_linear_param = Tk.Frame(self)
        frame_linear_param.pack(fill='x', expand=1, padx=15, pady=15)
        figure_gs = Figure(figsize=(6, 4), dpi=100)
        subplot_linear_param = figure_gs.add_subplot(111)
        figure_gs.tight_layout()
        subplot_linear_param.set_xscale('log')
        subplot_linear_param.set_ylim([0.5, 1.0])
        subplot_linear_param.set_xlabel("C")
        subplot_linear_param.set_ylabel("Accuracy")
        subplot_linear_param.set_title("GridSearchCV on parameter C in SVM with linear-kernel")

        param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']},
                      {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}]
        gs = GridSearchCV(estimator=evaluator_gs.pipeline, param_grid=param_grid, scoring='accuracy', cv=10,
                          n_jobs=-1)
        gs = gs.fit(x_train, y_train)
        y_true, y_pred = y_test, gs.predict(x_test)

        lieanr_end = len(param_range)
        subplot_linear_param.plot(map(lambda e: e[0]['clf__C'], gs.grid_scores_[0:lieanr_end]),
                                  map(lambda e: e[1], gs.grid_scores_[0:lieanr_end]),
                                  linewidth=1, label='svm_linear', color="blue", marker='o', markersize=5)
        subplot_linear_param.grid()
        self.attach_figure(figure_gs, frame_linear_param)

        frame_rbf_param = Tk.Frame(self)
        frame_rbf_param.pack(fill='x', expand=1, padx=15, pady=15)
        scrollbar = Tk.Scrollbar(frame_rbf_param)
        scrollbar.pack(side=Tk.RIGHT, fill=Tk.Y)
        text_rbf_param = Tk.Text(frame_rbf_param, width=800, wrap='none', yscrollcommand=scrollbar.set)
        text_rbf_param.insert(Tk.END, "1. Best parameter: " + str(gs.best_params_) + "\n\n")
        text_rbf_param.insert(Tk.END, "2. Best parameter performance on testing data.\n\n ")
        text_rbf_param.insert(Tk.END, classification_report(y_true, y_pred))
        text_rbf_param.insert(Tk.END, "\n\n")
        text_rbf_param.insert(Tk.END, "3. All parameter searched by GridSearchCV.\n\n ")
        log = []
        for params, mean_score, scores in gs.grid_scores_:
            log.append(
                ["%0.3f" % (mean_score), "(+/-%0.03f)" % (scores.std() * 2), params["clf__C"],
                 params.has_key("clf__gamma") and params["clf__gamma"] or "",
                 params["clf__kernel"]])
        text_rbf_param.insert(Tk.END, tabulate(log, headers=["Accuracy", "SD", "C", "gamma", "type"]))
        text_rbf_param.pack()
        scrollbar.config(command=text_rbf_param.yview)

    @staticmethod
    def attach_figure(figure, frame):
        canvas = FigureCanvasTkAgg(figure, master=frame)  # 内嵌散点图到UI
        canvas.show()
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        toolbar = NavigationToolbar2TkAgg(canvas, frame)  # 内嵌散点图工具栏到UI
        toolbar.update()
        canvas.tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
