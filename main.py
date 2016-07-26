# -*- coding: utf-8 -*-
"""
chiwenzhen
2016-07-20
@ruijie

"""
import numpy as np
from Tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from model import *


class App:
    def __init__(self):
        # 分类器训练
        self.evaluator = BreastCancerEvaluator()
        self.evaluator.load_data()
        self.evaluator.train()
        x_origin, y = self.evaluator.get_train_data()
        x = self.evaluator.reduce(x_origin)

        # 构建UI
        self.root = Tk()
        self.root.wm_title("Breast Cancer Evaluation Platform")

        # 1.菜单
        menubar = Menu(self.root)  # 添加菜单
        self.root.config(menu=menubar)
        filemenu = Menu(menubar)
        filemenu.add_command(label="Exit", command=sys.exit)
        menubar.add_cascade(label="File", menu=filemenu)

        # 2.matplotlib绘制和内嵌
        frame1 = Frame(self.root)
        frame1.pack(fill=BOTH, expand=1, padx=15, pady=15)
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.subplot = self.figure.add_subplot(111)
        self.plot_subplot(self.subplot, x, y)
        minx = np.min(x[:, 0])
        maxx = np.max(x[:, 0])
        self.plot_hyperplane(self.subplot, self.evaluator.get_clf(), minx, maxx)
        self.last_line = None
        canvas = FigureCanvasTkAgg(self.figure, master=frame1)
        canvas.show()
        canvas.get_tk_widget().grid(row=0, column=0)
        toolbar = NavigationToolbar2TkAgg(canvas, frame1)
        toolbar.update()
        canvas.tkcanvas.pack(side=TOP, fill=BOTH, expand=1)

        # 3.滑动条
        frame2 = Frame(self.root)
        frame2.pack(fill=BOTH, expand=1, padx=5, pady=5)
        canv = Canvas(frame2, relief=SUNKEN)
        vbar = Scrollbar(frame2, command=canv.yview)
        canv.config(scrollregion=(0, 0, 300, 1500))
        canv.config(yscrollcommand=vbar.set)
        vbar.pack(side=RIGHT, fill=Y)
        canv.pack(side=LEFT, expand=YES, fill=BOTH)
        feature_num = x_origin.shape[1]
        self.slides = [None] * feature_num
        feature_names = ["radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave",
                         "symmetry", "fractal",
                         "radius SE", "texture SE", "perimeter SE", "area SE", "smoothness SE", "compactness SE",
                         "concavity SE", "concave SE",
                         "symmetry SE", "fractal SE",
                         "radius MAX", "texture MAX", "perimeter MAX", "area MAX", "smoothness MAX", "compactness MAX",
                         "concavity MAX", "concave MAX",
                         "symmetry MAX", "fractal MAX"]
        for i in range(feature_num):
            canv.create_window(50, (i+1) * 50, window=Label(canv, text=feature_names[i]))
            min_x = np.min(x_origin[:, i])
            max_x = np.max(x_origin[:, i])
            self.slides[i] = Scale(canv, from_=min_x, to=max_x, resolution=(max_x - min_x) / 100.0,
                                   orient=HORIZONTAL, command=self.predict)
            canv.create_window(200, (i+1) * 50, window=self.slides[i])

        # 4.概率输出框
        self.malig_prob = StringVar()
        Label(self.root, text="恶性肿瘤概率").pack(side=LEFT)
        Entry(self.root, textvariable=self.malig_prob, bd=5).pack(side=LEFT, padx=5, pady=5)

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

    # 根据坐标X=(x1 ,x2)，绘制一个点
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
        str = "%.2f%%" % (result[0, 1] * 100)
        self.malig_prob.set(str)  # 恶性肿瘤的概率
        self.plot_point(self.subplot, self.evaluator.reduce(x))
        self.figure.canvas.draw()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = App()
    app.run()
