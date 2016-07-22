# -*- coding: utf-8 -*-
"""
chiwenzhen
2016-07-20
@ruijie

"""
import numpy as np
from Tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
        menubar = Menu(self.root)  # 添加菜单
        self.root.config(menu=menubar)
        filemenu = Menu(menubar)
        filemenu.add_command(label="Exit", command=sys.exit)
        menubar.add_cascade(label="File", menu=filemenu)

        self.figure = Figure(figsize=(5, 4), dpi=100)
        canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        canvas.show()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        canvas.tkcanvas.pack(side=TOP, fill=BOTH, expand=1)

        # 添加两个输入框，用于输入细胞特征
        label1 = Label(self.root, text="radius")
        label1.pack(side=LEFT)
        self.slide1 = Scale(self.root, from_=np.min(x_origin[:, 0]), to=np.max(x_origin[:, 0]), orient=HORIZONTAL,
                            command=self.predict)
        self.slide1.pack(side=LEFT)

        label2 = Label(self.root, text="texture")
        label2.pack(side=LEFT)
        self.slide2 = Scale(self.root, from_=np.min(x_origin[:, 1]), to=np.max(x_origin[:, 1]), orient=HORIZONTAL,
                            command=self.predict)
        self.slide2.pack(side=LEFT)

        self.malig_prob = StringVar()
        label3 = Label(self.root, text="malignant prob")
        label3.pack(side=LEFT)
        self.entry3 = Entry(self.root, textvariable=self.malig_prob, bd=5)
        self.entry3.pack(side=LEFT)

        # matplotlib绘制图
        self.subplot = self.figure.add_subplot(111)
        self.plot_subplot(self.subplot, x, y)
        minx = np.min(x[:, 0])
        maxx = np.max(x[:, 0])
        plot_hyperplane(self.subplot, self.evaluator.get_clf(), minx, maxx)
        self.last_line = None

    @staticmethod
    def plot_subplot(subplot, x, y):
        x_zero = x[y == 0]
        x = x[y == 1]
        subplot.plot(x_zero[:, 0], x_zero[:, 1], "g.", label="benign")
        subplot.plot(x[:, 0], x[:, 1], "k.", label="malignant")
        subplot.set_title('Model Illustration')
        subplot.set_xlabel('x1')
        subplot.set_ylabel('x2')

    def plot_point(self, subplot, x):
        if self.last_line is not None:
            self.last_line.remove()
            del self.last_line
        lines = subplot.plot(x[:, 0], x[:, 1], "ro", label="case")
        self.last_line = lines.pop(0)

    def predict(self, trival):
        x = np.array([[float(self.slide1.get()), float(self.slide2.get()), 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471,
                       0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003,
                       0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]])
        result = self.evaluator.predict(x)
        self.malig_prob.set(result[0, 1])  # 恶性肿瘤的概率
        self.plot_point(self.subplot, self.evaluator.reduce(x))
        self.figure.canvas.draw()

    def run(self):
        self.root.mainloop()


def plot_hyperplane(subplot, clf, min_x, max_x):
    # get the separating hyperplane
    w = clf.coef_[0]
    xx = np.linspace(min_x, max_x)
    yy = -(w[0] * xx + clf.intercept_[0]) / w[1]
    subplot.plot(xx, yy, "black")


if __name__ == "__main__":
    app = App()
    app.run()
