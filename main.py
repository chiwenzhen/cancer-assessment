# -*- coding: utf-8 -*-
"""
chiwenzhen
2016-07-20
@ruijie

"""

import matplotlib
from Tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from model import *

class App:
    def __init__(self):
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
        canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)

        # 添加两个输入框，用于输入细胞特征
        self.V1 = StringVar()
        self.V1.set("18.0")
        L1 = Label(self.root, text="radius")
        L1.pack(side=LEFT)
        E1 = Entry(self.root, textvariable=self.V1, bd=5)
        E1.pack(side=LEFT)

        self.V2 = StringVar()
        self.V2.set("10.0")
        L2 = Label(self.root, text="texture")
        L2.pack(side=LEFT)
        E2 = Entry(self.root, textvariable=self.V2, bd=5)
        E2.pack(side=LEFT)

        self.V3 = StringVar()
        L3 = Label(self.root, text="malignant prob")
        L3.pack(side=LEFT)
        self.E3 = Entry(self.root, textvariable=self.V3, bd=5)
        self.E3.pack(side=LEFT)

        # 分类器训练
        self.evaluator = BreastCancerEvaluator()
        self.evaluator.load_data()
        self.evaluator.train()
        X, y = self.evaluator.get_train_data_r()

        # matplotlib绘制图
        self.subplot = self.figure.add_subplot(111)
        self.plot_subplot(self.subplot, X, y)
        minx = np.min(X[:, 0])
        maxx = np.max(X[:, 0])
        self.plot_hyperplane(self.subplot, self.evaluator.get_clf(), minx, maxx)
        self.last_line = None

        # 添加一个按钮，用于评估具体案例中肿瘤属于良性/恶性
        button = Button(master=self.root, text='Predict', command=self.predict)
        button.pack(side=BOTTOM)

    def plot_hyperplane(self, subplot, clf, min_x, max_x):
        # get the separating hyperplane
        w = clf.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(min_x, max_x)
        yy = a * xx - (clf.intercept_[0]) / w[1]
        subplot.plot(xx, yy, "black")

    def plot_subplot(self, subplot, X, y):
        X_zero = X[y == 0]
        X_one = X[y == 1]
        subplot.plot(X_zero[:, 0], X_zero[:, 1], "g.", label="benign")
        subplot.plot(X_one[:, 0], X_one[:, 1], "k.", label="malignant")
        subplot.set_title('Model Illustration')
        subplot.set_xlabel('x1')
        subplot.set_ylabel('x2')

    def plot_point(self, subplot, X):
        if self.last_line != None:
            self.last_line.remove()
            del self.last_line
        lines = subplot.plot(X[:, 0], X[:, 1], "ro", label="case")
        self.last_line = lines.pop(0)

    def predict(self):
        X = np.array([[float(self.V1.get()),float(self.V2.get()),122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]])
        result, newX = self.evaluator.predict(X)
        self.V3.set(result[0, 1]) # 恶性肿瘤的概率
        self.plot_point(self.subplot, self.evaluator.reduce(X))
        self.figure.canvas.draw()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = App()
    app.run()

