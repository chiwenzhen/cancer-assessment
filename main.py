# -*- coding: utf-8 -*-
"""
chiwenzhen
2016-07-20
@ruijie

"""
import numpy as np
import sys

import Tkinter as TK
from ttk import Notebook
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from model import CancerEvaluator, DataSet
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from scipy import interp


class App:
    def __init__(self, root):
        # 数据载入和分类器训练
        self.dataset = DataSet(data_path="wdbc.data")
        x = self.dataset.x
        y = self.dataset.y
        x_train = self.dataset.x_train
        y_train = self.dataset.y_train
        x_test = self.dataset.x_test
        y_test = self.dataset.y_test

        self.evaluator = CancerEvaluator()
        self.evaluator.load_data(x, y)
        self.evaluator.train()
        x_train_r = self.evaluator.reduce(x_train)  # 特征降维
        x_test_r = self.evaluator.reduce(x_test)  # 特征降维

        # 初始化UI
        # 1.菜单和标签页
        menubar = TK.Menu(root)  # 添加菜单
        root.config(menu=menubar)
        filemenu = TK.Menu(menubar)
        filemenu.add_command(label="Exit", command=sys.exit)
        menubar.add_cascade(label="File", menu=filemenu)

        notebook = Notebook(root)  # 添加标签页
        notebook.pack(fill=TK.BOTH)

        first_page = TK.Frame(notebook)
        notebook.add(first_page, text="Main")

        second_page = TK.Frame(notebook)
        notebook.add(second_page, text="Learning Curve")

        third_page = TK.Frame(notebook)
        notebook.add(third_page, text="Validation Curve")

        fourth_page = TK.Frame(notebook)
        notebook.add(fourth_page, text="ROC & AUC")

        fifth_page = TK.Frame(notebook)
        notebook.add(fifth_page, text="Testing Result")

        # first_page 1.matplotlib绘制
        frame_x_y = TK.Frame(first_page)
        frame_x_y.pack(fill=TK.BOTH, expand=1, padx=15, pady=15)
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.subplot = self.figure.add_subplot(111)
        self.subplot.set_title('Breast Cancer Evaluation Model')
        self.plot_subplot(self.subplot, x_train_r, y_train)  # 绘制数据散点
        minx = np.min(x_train_r[:, 0])
        maxx = np.max(x_train_r[:, 0])
        self.plot_hyperplane(self.subplot, self.evaluator.get_clf(), minx, maxx)  # 绘制超平面
        self.last_line = None
        canvas = FigureCanvasTkAgg(self.figure, master=frame_x_y)  # 内嵌散点图到UI
        self.figure.tight_layout()
        canvas.show()
        canvas.get_tk_widget().pack(side=TK.TOP, fill=TK.BOTH, expand=1)
        toolbar = NavigationToolbar2TkAgg(canvas, frame_x_y)  # 内嵌散点图工具栏到UI
        toolbar.update()
        canvas.tkcanvas.pack(side=TK.TOP, fill=TK.BOTH, expand=1)

        # first_page 2.概率输出框
        frame_output = TK.Frame(first_page)
        frame_output.pack(fill=TK.BOTH, expand=1, padx=5, pady=5)
        self.malig_prob = TK.StringVar()
        TK.Label(frame_output, text="malignant prob").pack(side=TK.LEFT)
        TK.Entry(frame_output, textvariable=self.malig_prob, bd=5).pack(side=TK.LEFT, padx=5, pady=5)
        TK.Label(frame_output, text="hyperplane： %.4f*x1 + %.4f*x2 + %.4f = 0" % (
        self.evaluator.get_clf().coef_[0, 0], self.evaluator.get_clf().coef_[0, 1],
        self.evaluator.get_clf().intercept_)).pack(side=TK.RIGHT)

        # first_page 3.滑动条
        frame_scale = TK.Frame(first_page)
        frame_scale.pack(fill=TK.BOTH, expand=1, padx=5, pady=5)
        canv = TK.Canvas(frame_scale, relief=TK.SUNKEN)
        vbar = TK.Scrollbar(frame_scale, command=canv.yview)
        canv.config(scrollregion=(0, 0, 300, 1500))
        canv.config(yscrollcommand=vbar.set)
        vbar.pack(side=TK.RIGHT, fill=TK.Y)
        canv.pack(side=TK.LEFT, expand=TK.YES, fill=TK.BOTH)
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
            canv.create_window(60, (i + 1) * 40, window=TK.Label(canv, text=str(i+1) + ". " + feature_names[i]))
            min_x = np.min(x_train[:, i])
            max_x = np.max(x_train[:, i])
            self.slides[i] = TK.Scale(canv, from_=min_x, to=max_x, resolution=(max_x - min_x) / 100.0,
                                   orient=TK.HORIZONTAL, command=self.predict)
            canv.create_window(200, (i + 1) * 40, window=self.slides[i])

        # second_page 1.学习曲线
        evaluator_lcurve = CancerEvaluator()
        train_sizes, train_scores, test_scores = learning_curve(estimator=evaluator_lcurve.get_pipeline(),
                                                                X=x_train,
                                                                y=y_train,
                                                                train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        frame_lcurve = TK.Frame(second_page)
        frame_lcurve.pack(fill='x', expand=1, padx=15, pady=15)
        self.figure_lcurve = Figure(figsize=(6, 6), dpi=100)
        self.subplot_lcurve = self.figure_lcurve.add_subplot(111)
        self.subplot_lcurve.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5,
                                 label='training accuracy')
        self.subplot_lcurve.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15,
                                         color='blue')
        self.subplot_lcurve.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5,
                                 label='cross-validation accuracy')
        self.subplot_lcurve.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15,
                                         color='green')
        self.subplot_lcurve.grid()
        self.subplot_lcurve.set_xlabel('Number of training samples')
        self.subplot_lcurve.set_ylabel('Accuracy')
        self.subplot_lcurve.legend(loc='lower right')
        self.subplot_lcurve.set_ylim([0.8, 1.0])
        canvas = FigureCanvasTkAgg(self.figure_lcurve, master=frame_lcurve)  # 内嵌散点图到UI
        
        canvas.show()
        canvas.get_tk_widget().pack(side=TK.TOP, fill=TK.BOTH, expand=1)
        toolbar = NavigationToolbar2TkAgg(canvas, frame_lcurve)  # 内嵌散点图工具栏到UI
        toolbar.update()
        canvas.tkcanvas.pack(side=TK.TOP, fill=TK.BOTH, expand=1)

        # third_page 验证曲线
        evaluator_vcurve = CancerEvaluator()
        param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        train_scores, test_scores = validation_curve(estimator=evaluator_vcurve.get_pipeline(),
                                                     X=x_train, y=y_train,
                                                     param_name='clf__C',
                                                     param_range=param_range, cv=10)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        frame_vcurve = TK.Frame(third_page)
        frame_vcurve.pack(fill='x', expand=1, padx=15, pady=15)
        self.figure_vcurve = Figure(figsize=(6, 6), dpi=100)
        self.subplot_vcurve = self.figure_vcurve.add_subplot(111)

        self.subplot_vcurve.plot(param_range, train_mean, color='blue', marker='o', markersize=5,
                                 label='training accuracy')
        self.subplot_vcurve.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15,
                                         color='blue')
        self.subplot_vcurve.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5,
                                 label='cross-validation accuracy')
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
        canvas.get_tk_widget().pack(side=TK.TOP, fill=TK.BOTH, expand=1)
        toolbar = NavigationToolbar2TkAgg(canvas, frame_vcurve)  # 内嵌散点图工具栏到UI
        toolbar.update()
        canvas.tkcanvas.pack(side=TK.TOP, fill=TK.BOTH, expand=1)

        # fourth_page ROC&AUC
        evaluator_roc = CancerEvaluator()
        frame_roc = TK.Frame(fourth_page)
        frame_roc.pack(fill='x', expand=1, padx=15, pady=15)
        cv = StratifiedKFold(y_train, n_folds=3, random_state=1)
        self.figure_roc = Figure(figsize=(6, 6), dpi=100)
        self.subplot_roc = self.figure_roc.add_subplot(111)

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)

        for i, (train, test) in enumerate(cv):
            evaluator_roc.load_data(x_train, y_train)
            probas = evaluator_roc.get_pipeline().fit(x_train[train], y_train[train]).predict_proba(x_train[test])
            fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            self.subplot_roc.plot(fpr, tpr, linewidth=1, label='ROC fold %d (area = %0.2f)' % (i + 1, roc_auc))

        self.subplot_roc.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='random guessing')

        mean_tpr /= len(cv)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        self.subplot_roc.plot(mean_fpr, mean_tpr, 'k--', label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
        self.subplot_roc.plot([0, 0, 1], [0, 1, 1], lw=2, linestyle=':', color='black', label='perfect performance')
        self.subplot_roc.set_xlim([-0.05, 1.05])
        self.subplot_roc.set_ylim([-0.05, 1.05])
        self.subplot_roc.set_xlabel('false positive rate')
        self.subplot_roc.set_ylabel('true positive rate')
        self.subplot_roc.set_title('Receiver Operator Characteristic')
        self.subplot_roc.legend(loc="lower right")

        canvas = FigureCanvasTkAgg(self.figure_roc, master=frame_roc)  # 内嵌散点图到UI
        canvas.show()
        canvas.get_tk_widget().pack(side=TK.TOP, fill=TK.BOTH, expand=1)
        toolbar = NavigationToolbar2TkAgg(canvas, frame_roc)  # 内嵌散点图工具栏到UI
        toolbar.update()
        canvas.tkcanvas.pack(side=TK.TOP, fill=TK.BOTH, expand=1)

        # ffifth_page 1.测试集展示
        frame_test = TK.Frame(fifth_page)
        frame_test.pack(fill='x', expand=1, padx=15, pady=15)
        self.figure_test = Figure(figsize=(4, 4), dpi=100)
        self.subplot_test = self.figure_test.add_subplot(111)
        self.subplot_test.set_title('Breast Cancer Testing')
        self.plot_subplot(self.subplot_test, x_test_r, y_test)  # 绘制数据散点
        minx = np.min(x_test_r[:, 0])
        maxx = np.max(x_test_r[:, 0])
        self.plot_hyperplane(self.subplot_test, self.evaluator.get_clf(), minx, maxx)  # 绘制超平面
        canvas = FigureCanvasTkAgg(self.figure_test, master=frame_test)  # 内嵌散点图到UI
        self.figure_test.tight_layout()
        canvas.show()
        canvas.get_tk_widget().pack(side=TK.TOP, fill=TK.BOTH, expand=1)
        toolbar = NavigationToolbar2TkAgg(canvas, frame_test)  # 内嵌散点图工具栏到UI
        toolbar.update()
        canvas.tkcanvas.pack(side=TK.TOP, fill=TK.BOTH, expand=1)

        # first_page 2.测试性能指标 precision recall f_value
        y_pred = self.evaluator.get_pipeline().predict(x_test)
        frame_matrix = TK.Frame(fifth_page)
        frame_matrix.pack(side=TK.LEFT, fill='x', expand=1, padx=15, pady=15)
        self.figure_matrix = Figure(figsize=(4, 4), dpi=100)
        self.subplot_matrix = self.figure_matrix.add_subplot(111)

        confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
        self.subplot_matrix.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                self.subplot_matrix.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

        self.subplot_matrix.set_xlabel('predicted label')
        self.subplot_matrix.set_ylabel('true label')

        canvas = FigureCanvasTkAgg(self.figure_matrix, master=frame_matrix)  # 内嵌散点图到UI
        self.figure_matrix.tight_layout()
        canvas.show()
        canvas.get_tk_widget().pack(side=TK.TOP, fill=TK.BOTH, expand=1)
        canvas.tkcanvas.pack(side=TK.TOP, fill=TK.BOTH, expand=1)

        frame_result = TK.Frame(fifth_page)
        frame_result.pack(side=TK.LEFT, fill='x', expand=1, padx=15, pady=15)
        TK.Label(frame_result, text="Accuracy: ").grid(row=0, column=0, sticky=TK.W)
        TK.Label(frame_result, text=str(self.evaluator.get_pipeline().score(x_test, y_test))).grid(row=0, column=1,
                                                                                                sticky=TK.W)
        TK.Label(frame_result, text="Precision: ").grid(row=1, column=0, sticky=TK.W)
        TK.Label(frame_result, text=str(precision_score(y_true=y_test, y_pred=y_pred))).grid(row=1, column=1, sticky=TK.W)
        TK.Label(frame_result, text="Recall: ").grid(row=2, column=0, sticky=TK.W)
        TK.Label(frame_result, text=str(recall_score(y_true=y_test, y_pred=y_pred))).grid(row=2, column=1, sticky=TK.W)
        TK.Label(frame_result, text="F-value: ").grid(row=3, column=0, sticky=TK.W)
        TK.Label(frame_result, text=str(f1_score(y_true=y_test, y_pred=y_pred))).grid(row=3, column=1, sticky=TK.W)

    # 根据x,y，绘制散点图
    @staticmethod
    def plot_subplot(subplot, x, y):
        x_zero = x[y == 0]
        x_one = x[y == 1]
        subplot.plot(x_zero[:, 0], x_zero[:, 1], "g.", label='benign')
        subplot.plot(x_one[:, 0], x_one[:, 1], "k.", label='malignant')
        subplot.set_xlabel('x1')
        subplot.set_ylabel('x2')
        subplot.legend(loc='lower right')

    # 删除上一个点，再根据坐标X=(x1 ,x2)重绘绘制一个点，以实现点的移动
    def plot_point(self, subplot, x):
        if self.last_line is not None:
            self.last_line.remove()
            del self.last_line
        lines = subplot.plot(x[:, 0], x[:, 1], "ro", label="case")
        self.last_line = lines.pop(0)
        subplot.legend(loc='lower right')

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
    master = TK.Tk()
    master.wm_title("Breast Cancer Evaluation Platform")
    master.geometry('900x750')
    master.iconbitmap("cancer.ico")
    app = App(master)
    master.mainloop()
