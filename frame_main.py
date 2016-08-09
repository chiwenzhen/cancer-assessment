# coding=utf-8
import numpy as np
import Tkinter as Tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
import tkMessageBox as MsbBox


class MainFrame(Tk.Frame):
    def __init__(self, master, x_train, y_train, x_test, y_test, evaluator):
        Tk.Frame.__init__(self, master)
        self.evaluator = evaluator
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.best_estimator = None
        self.evaluator.load_data(x_train, y_train, x_test, y_test)
        self.evaluator.train()
        x_train_r = self.evaluator.reduce(x_train)  # 特征降维

        # 0. 优化按钮
        button_opt = Tk.Button(self, text="优化", command=self.optimize_parameter)
        button_opt.pack(side=Tk.TOP, anchor=Tk.E)

        # 1. 散点图
        frame_train = Tk.Frame(self)
        frame_train.pack(fill=Tk.BOTH, expand=1, padx=15, pady=15)
        self.figure_train = Figure(figsize=(5, 4), dpi=100)
        self.subplot_train = self.figure_train.add_subplot(111)
        self.subplot_train.set_title('Breast Cancer Evaluation Model')
        self.figure_train.tight_layout()  # 一定要放在add_subplot函数之后，否则崩溃
        self.last_line = None

        h = .02  # step size in the mesh
        x1_min, x1_max = x_train_r[:, 0].min() - 1, x_train_r[:, 0].max() + 1
        x2_min, x2_max = x_train_r[:, 1].min() - 1, x_train_r[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
        yy = self.evaluator.clf.predict(np.c_[xx1.ravel(), xx2.ravel()])
        yy = yy.reshape(xx1.shape)
        self.subplot_train.contourf(xx1, xx2, yy, cmap=plt.cm.get_cmap("Paired"), alpha=0.8)
        self.subplot_train.scatter(x_train_r[:, 0], x_train_r[:, 1], c=y_train, cmap=plt.cm.get_cmap("Paired"))
        self.attach_figure(self.figure_train, frame_train)

        # 2. 概率输出框
        frame_prob = Tk.Frame(self)
        frame_prob.pack(fill=Tk.BOTH, expand=1, padx=5, pady=5)
        self.strvar_prob = Tk.StringVar()
        Tk.Label(frame_prob, text="malignant prob").pack(side=Tk.LEFT)
        Tk.Entry(frame_prob, textvariable=self.strvar_prob, bd=5).pack(side=Tk.LEFT, padx=5, pady=5)

        # 3. 滑动条
        frame_slides = Tk.Frame(self)
        frame_slides.pack(fill=Tk.BOTH, expand=1, padx=5, pady=5)
        canv = Tk.Canvas(frame_slides, relief=Tk.SUNKEN)
        vbar = Tk.Scrollbar(frame_slides, command=canv.yview)
        canv.config(scrollregion=(0, 0, 300, 1500))
        canv.config(yscrollcommand=vbar.set)
        vbar.pack(side=Tk.RIGHT, fill=Tk.Y)
        canv.pack(side=Tk.LEFT, expand=Tk.YES, fill=Tk.BOTH)
        feature_num = x_train.shape[1]
        self.slides = [None] * feature_num  # 滑动条个数为特征个数
        feature_names = ["radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave",
                         "symmetry", "fractal",
                         "radius SE", "texture SE", "perimeter SE", "area SE", "smoothness SE", "compactness SE",
                         "concavity SE", "concave SE",
                         "symmetry SE", "fractal SE",
                         "radius MAX", "texture MAX", "perimeter MAX", "area MAX", "smoothness MAX", "compactness MAX",
                         "concavity MAX", "concave MAX",
                         "symmetry MAX", "fractal MAX"]
        for i in range(feature_num):
            canv.create_window(60, (i + 1) * 40, window=Tk.Label(canv, text=str(i + 1) + ". " + feature_names[i]))
            min_x = np.min(x_train[:, i])
            max_x = np.max(x_train[:, i])
            self.slides[i] = Tk.Scale(canv, from_=min_x, to=max_x, resolution=(max_x - min_x) / 100.0,
                                      orient=Tk.HORIZONTAL, command=self.predict)
            canv.create_window(200, (i + 1) * 40, window=self.slides[i])

    # 根据即特征值，计算归属类别的概率
    def predict(self, trivial):
        x = np.arange(30, dtype='f').reshape((1, 30))
        for i in range(30):
            x[0, i] = float(self.slides[i].get())
        result = self.evaluator.predict(x)
        self.strvar_prob.set("%.2f%%" % (result[0, 1] * 100))  # 恶性肿瘤的概率
        self.plot_point(self.subplot_train, self.evaluator.reduce(x))
        self.figure_train.canvas.draw()

    # 重绘点
    def plot_point(self, subplot, x):
        if self.last_line is not None:
            self.last_line.remove()
            del self.last_line
        lines = subplot.plot(x[:, 0], x[:, 1], "ro", label="case")
        self.last_line = lines.pop(0)
        subplot.legend(loc='lower right')

    # 将figure放到frame上
    @staticmethod
    def attach_figure(figure, frame):
        canvas = FigureCanvasTkAgg(figure, master=frame)  # 内嵌散点图到UI
        canvas.show()
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        toolbar = NavigationToolbar2TkAgg(canvas, frame)  # 内嵌散点图工具栏到UI
        toolbar.update()
        canvas.tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

    # 搜索最优参数
    def optimize_parameter(self):
        param_C_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0,
                         self.evaluator.pipeline.named_steps["clf"].C]
        param_gamma_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0,
                             self.evaluator.pipeline.named_steps["clf"].gamma]
        param_grid = [{'clf__C': param_C_range, 'clf__gamma': param_gamma_range, 'clf__kernel': ['rbf']}]
        gs = GridSearchCV(estimator=self.evaluator.pipeline, param_grid=param_grid, scoring='accuracy', cv=10,
                          n_jobs=-1)
        gs = gs.fit(self.x_train, self.y_train)
        self.best_estimator = gs.best_estimator_
        new_accuracy = gs.best_score_

        #########################################################
        print("Best parameters set found on development set:")
        print()
        print(gs.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in gs.grid_scores_:
            print("%f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = self.y_test, gs.predict(self.x_test)
        print(classification_report(y_true, y_pred))
        print()
        #########################################################

        for params, mean_score, scores in gs.grid_scores_:
            if params["clf__kernel"] == self.evaluator.pipeline.named_steps["clf"].kernel \
                    and params["clf__C"] == self.evaluator.pipeline.named_steps["clf"].C \
                    and params["clf__gamma"] == self.evaluator.pipeline.named_steps["clf"].gamma:
                old_accuracy = mean_score

        if self.best_estimator.named_steps["clf"].kernel == self.evaluator.pipeline.named_steps["clf"].kernel \
                and self.best_estimator.named_steps["clf"].C == self.evaluator.pipeline.named_steps["clf"].C \
                and self.best_estimator.named_steps["clf"].gamma == self.evaluator.pipeline.named_steps["clf"].gamma:
            MsbBox.askquestion("Sorry", "Sorry, no more better models.")
        elif new_accuracy > old_accuracy:
            MsbBox.askquestion("Congrats", "A better model has been found:\nAccuracy increase from %f to %f" % (
                old_accuracy, new_accuracy))
        else:
            MsbBox.askquestion("Notificatoin",
                               "A argumentative model has been found:\nAccuracy decrease from %f to %f" % (
                                   old_accuracy, new_accuracy))
