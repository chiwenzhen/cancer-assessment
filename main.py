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

        # 1.菜单
        menubar = Menu(self.root)  # 添加菜单
        self.root.config(menu=menubar)
        filemenu = Menu(menubar)
        filemenu.add_command(label="Exit", command=sys.exit)
        menubar.add_cascade(label="File", menu=filemenu)

        # 2.matplotlib绘制和内嵌

        self.frame1 = Frame(self.root)
        self.frame1.pack(fill=BOTH, expand=1,padx=5, pady=5)
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.subplot = self.figure.add_subplot(111)
        self.plot_subplot(self.subplot, x, y)
        minx = np.min(x[:, 0])
        maxx = np.max(x[:, 0])
        self.plot_hyperplane(self.subplot, self.evaluator.get_clf(), minx, maxx)
        self.last_line = None
        canvas = FigureCanvasTkAgg(self.figure, master=self.frame1)
        canvas.show()
        canvas.get_tk_widget().grid(row=0, column=0)
        canvas.tkcanvas.pack(side=TOP, fill=BOTH, expand=1)

        # 3.滑动条

        self.frame2 = Frame(self.root)
        self.frame2.pack(fill=BOTH, expand=1, padx=5, pady=5)

        canv = Canvas(self.frame2, relief=SUNKEN)
        canv.config(width=300, height=200)
        canv.config(scrollregion=(0,0,300, 1500))
        canv.config(highlightthickness=0)
        sbar = Scrollbar(self.frame2)
        sbar.config(command=canv.yview)
        canv.config(yscrollcommand=sbar.set)
        sbar.pack(side=RIGHT, fill=Y)
        canv.pack(side=LEFT, expand=YES, fill=BOTH)
        self.canvas = canv
        feature_num = x_origin.shape[1]
        self.slides = [None] * feature_num
        for i in range(feature_num):
            self.canvas.create_window(50, i*50, window=Label(self.canvas, text="feature " + str(i)))
            min_x = np.min(x_origin[:, i])
            max_x = np.max(x_origin[:, i])
            self.slides[i] = Scale(self.canvas, from_=min_x, to=max_x, resolution=(max_x - min_x) / 100.0,
                                   orient=HORIZONTAL, command=self.predict)
            self.canvas.create_window(200, i*50, window=self.slides[i])

        # 4.概率输出框
        self.malig_prob = StringVar()
        label3 = Label(self.root, text="恶性肿瘤概率")
        label3.pack(side=LEFT)
        self.entry3 = Entry(self.root, textvariable=self.malig_prob, bd=5)
        self.entry3.pack(side=LEFT, padx=5, pady=5)

    # 根据x，y绘制散点图
    @staticmethod
    def plot_subplot(subplot, x, y):
        x_zero = x[y == 0]
        x = x[y == 1]
        subplot.plot(x_zero[:, 0], x_zero[:, 1], "g.", label="benign")
        subplot.plot(x[:, 0], x[:, 1], "k.", label="malignant")
        subplot.set_title('Model Illustration')
        subplot.set_xlabel('x1')
        subplot.set_ylabel('x2')

    # 绘制一个点，坐标为X=(x1 ,x2)
    def plot_point(self, subplot, x):
        if self.last_line is not None:
            self.last_line.remove()
            del self.last_line
        lines = subplot.plot(x[:, 0], x[:, 1], "ro", label="case")
        self.last_line = lines.pop(0)

    # 绘制超平面
    @staticmethod
    def plot_hyperplane(subplot, clf, min_x, max_x):
        w = clf.coef_[0]
        xx = np.linspace(min_x, max_x)
        yy = -(w[0] * xx + clf.intercept_[0]) / w[1]
        subplot.plot(xx, yy, "black")

    # 根据滑动条数据（即特征数据）计算归属类别的概率
    def predict(self, trivial):
        x = np.arange(30, dtype='f').reshape((1, 30))
        for i in range(30):
            x[0, i] = float(self.slides[i].get())
        result = self.evaluator.predict(x)
        str = "%.2f%%" % (result[0, 1]*100)
        self.malig_prob.set(str)  # 恶性肿瘤的概率
        self.plot_point(self.subplot, self.evaluator.reduce(x))
        self.figure.canvas.draw()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = App()
    app.run()
