# -*- coding: utf-8 -*-
"""
chiwenzhen
2016-07-20
@ruijie

"""
import numpy as np
from Tkinter import *
from ttk import Notebook
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from model import *
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve


class App:
    def __init__(self, root):
        # 分类器训练
        self.dataset = DataSet()
        self.dataset.load_data(data_path="wdbc.data")
        x, y = self.dataset.get_data()

        self.evaluator = LREvaluator()
        self.evaluator.load_data(x, y)
        self.evaluator.train()
        x, y = self.evaluator.get_train_data()
        x_r = self.evaluator.reduce(x)  # 特征降维

        # 初始化UI
        # 1.菜单和标签页
        menubar = Menu(root)  # 添加菜单
        root.config(menu=menubar)
        filemenu = Menu(menubar)
        filemenu.add_command(label="Exit", command=sys.exit)
        menubar.add_cascade(label="File", menu=filemenu)

        notebook = Notebook(root)  # 添加标签页
        notebook.pack(fill=BOTH)

        first_page = Frame(notebook)
        notebook.add(first_page, text="主页")

        second_page = Frame(notebook)
        notebook.add(second_page, text="学习曲线")

        third_page = Frame(notebook)
        notebook.add(third_page, text="验证曲线")

        fourth_page = Frame(notebook)
        notebook.add(fourth_page, text="ROC")

        fifth_page = Frame(notebook)
        notebook.add(fifth_page, text="测试结果")

        # first_page 1.matplotlib绘制
        frame_x_y = Frame(first_page)
        frame_x_y.pack(fill=BOTH, expand=1, padx=15, pady=15)
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.subplot = self.figure.add_subplot(111)
        self.plot_subplot(self.subplot, x_r, y)  # 绘制数据散点
        minx = np.min(x_r[:, 0])
        maxx = np.max(x_r[:, 0])
        self.plot_hyperplane(self.subplot, self.evaluator.get_clf(), minx, maxx)  # 绘制超平面
        self.last_line = None
        canvas = FigureCanvasTkAgg(self.figure, master=frame_x_y)  # 内嵌散点图到UI
        canvas.show()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        toolbar = NavigationToolbar2TkAgg(canvas, frame_x_y)  # 内嵌散点图工具栏到UI
        toolbar.update()
        canvas.tkcanvas.pack(side=TOP, fill=BOTH, expand=1)

        # first_page 2.概率输出框
        frame_output = Frame(first_page)
        frame_output.pack(fill=BOTH, expand=1, padx=5, pady=5)
        self.malig_prob = StringVar()
        Label(frame_output, text="恶性肿瘤概率").pack(side=LEFT)
        Entry(frame_output, textvariable=self.malig_prob, bd=5).pack(side=LEFT, padx=5, pady=5)

        # first_page 3.滑动条
        frame_scale = Frame(first_page)
        frame_scale.pack(fill=BOTH, expand=1, padx=5, pady=5)
        canv = Canvas(frame_scale, relief=SUNKEN)
        vbar = Scrollbar(frame_scale, command=canv.yview)
        canv.config(scrollregion=(0, 0, 300, 1500))
        canv.config(yscrollcommand=vbar.set)
        vbar.pack(side=RIGHT, fill=Y)
        canv.pack(side=LEFT, expand=YES, fill=BOTH)
        feature_num = x.shape[1]
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
            canv.create_window(60, (i + 1) * 40, window=Label(canv, text=feature_names[i]))
            min_x = np.min(x[:, i])
            max_x = np.max(x[:, i])
            self.slides[i] = Scale(canv, from_=min_x, to=max_x, resolution=(max_x - min_x) / 100.0,
                                   orient=HORIZONTAL, command=self.predict)
            canv.create_window(200, (i + 1) * 40, window=self.slides[i])

        # second_page 1.学习曲线
        evaluator_lcurve = LREvaluator()
        train_sizes, train_scores, test_scores = learning_curve(estimator=evaluator_lcurve.get_pipeline(),
                                                                X=self.dataset.get_data_x(),
                                                                y=self.dataset.get_data_y(),
                                                                train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        frame_lcurve = Frame(second_page)
        frame_lcurve.pack(fill='x', expand=1, padx=15, pady=15)
        self.figure_lcurve = Figure(figsize=(6, 6), dpi=100)
        self.subplot_lcurve = self.figure_lcurve.add_subplot(111)
        self.subplot_lcurve.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5,
                                 label='training accuracy')
        self.subplot_lcurve.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15,
                                         color='blue')
        self.subplot_lcurve.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5,
                                 label='validation accuracy')
        self.subplot_lcurve.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15,
                                         color='green')
        self.subplot_lcurve.grid()
        self.subplot_lcurve.set_xlabel('Number of training samples')
        self.subplot_lcurve.set_ylabel('Accuracy')
        self.subplot_lcurve.legend(loc='lower right')
        self.subplot_lcurve.set_ylim([0.8, 1.0])
        canvas = FigureCanvasTkAgg(self.figure_lcurve, master=frame_lcurve)  # 内嵌散点图到UI
        canvas.show()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        toolbar = NavigationToolbar2TkAgg(canvas, frame_lcurve)  # 内嵌散点图工具栏到UI
        toolbar.update()
        canvas.tkcanvas.pack(side=TOP, fill=BOTH, expand=1)

        # third_page 验证曲线
        evaluator_vcurve = LREvaluator()
        param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        train_scores, test_scores = validation_curve(estimator=evaluator_vcurve.get_pipeline(),
                                                     X=self.dataset.get_data_x(), y=self.dataset.get_data_y(),
                                                     param_name='clf__C',
                                                     param_range=param_range, cv=10)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        frame_vcurve = Frame(third_page)
        frame_vcurve.pack(fill='x', expand=1, padx=15, pady=15)
        self.figure_vcurve = Figure(figsize=(6, 6), dpi=100)
        self.subplot_vcurve = self.figure_vcurve.add_subplot(111)

        self.subplot_vcurve.plot(param_range, train_mean, color='blue', marker='o', markersize=5,
                                 label='training accuracy')
        self.subplot_vcurve.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15,
                                         color='blue')
        self.subplot_vcurve.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5,
                                 label='validation accuracy')
        self.subplot_vcurve.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15,
                                         color='green')

        self.subplot_vcurve.grid()
        self.subplot_vcurve.set_xscale('log')
        self.subplot_vcurve.legend(loc='lower right')
        self.subplot_vcurve.set_xlabel('Parameter C')
        self.subplot_vcurve.set_ylabel('Accuracy')
        self.subplot_vcurve.set_ylim([0.91, 1.0])
        canvas = FigureCanvasTkAgg(self.figure_vcurve, master=frame_vcurve)  # 内嵌散点图到UI
        canvas.show()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        toolbar = NavigationToolbar2TkAgg(canvas, frame_vcurve)  # 内嵌散点图工具栏到UI
        toolbar.update()
        canvas.tkcanvas.pack(side=TOP, fill=BOTH, expand=1)

    # 根据x,y，绘制散点图
    @staticmethod
    def plot_subplot(subplot, x, y):
        x_zero = x[y == 0]
        x_one = x[y == 1]
        subplot.plot(x_zero[:, 0], x_zero[:, 1], "g.", label="benign")
        subplot.plot(x_one[:, 0], x_one[:, 1], "k.", label="malignant")
        subplot.set_title('Model Illustration')
        subplot.set_xlabel('x1')
        subplot.set_ylabel('x2')

    # 删除上一个点，再根据坐标X=(x1 ,x2)重绘绘制一个点，以实现点的移动
    def plot_point(self, subplot, x):
        if self.last_line is not None:
            self.last_line.remove()
            del self.last_line
        lines = subplot.plot(x[:, 0], x[:, 1], "ro", label="case")
        self.last_line = lines.pop(0)

    # 根据clf给出的系数，绘制超平面
    @staticmethod
    def plot_hyperplane(subplot, clf, min_x, max_x):
        w = clf.coef_[0]
        xx = np.linspace(min_x, max_x)
        yy = -(w[0] * xx + clf.intercept_[0]) / w[1]
        subplot.plot(xx, yy, "black")

    # 根据即特征值，计算归属类别的概率
    def predict(self, trivial):
        x = np.arange(30, dtype='f').reshape((1, 30))
        for i in range(30):
            x[0, i] = float(self.slides[i].get())
        result = self.evaluator.predict(x)
        self.malig_prob.set("%.2f%%" % (result[0, 1] * 100))  # 恶性肿瘤的概率
        self.plot_point(self.subplot, self.evaluator.reduce(x))
        self.figure.canvas.draw()


if __name__ == "__main__":
    master = Tk()
    master.wm_title("Breast Cancer Evaluation Platform")
    master.geometry('900x700')
    master.iconbitmap("cancer2.ico")
    app = App(master)
    master.mainloop()
