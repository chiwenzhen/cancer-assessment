# coding=utf-8
import Tkinter as Tk
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


class BreastCancerMainFrame(Tk.Frame):
    def __init__(self, master, x_train, y_train, x_test, y_test, evaluator, console):
        Tk.Frame.__init__(self, master)
        self.evaluator = evaluator
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.new_estimator = None
        self.console = console
        self.evaluator.load_data(x_train, y_train, x_test, y_test)
        self.evaluator.train()
        self.x_train_r = self.evaluator.reduce(x_train)  # 特征降维

        # 0. 优化按钮
        self.button_opt = Tk.Button(self, text="优化", command=self.optimize_parameter)
        self.button_opt.pack(side=Tk.TOP, anchor=Tk.E)
        self.label_tips = Tk.Label(self)
        self.label_tips.pack(side=Tk.TOP, anchor=Tk.E)

        # 1. 散点图
        frame_train = Tk.Frame(self)
        frame_train.pack(fill=Tk.BOTH, expand=1, padx=15, pady=15)
        self.figure_train = Figure(figsize=(5, 4), dpi=100)
        self.subplot_train = self.figure_train.add_subplot(111)
        self.subplot_train.set_title('Breast Cancer Evaluation Model (2-dim)')
        self.figure_train.tight_layout()  # 一定要放在add_subplot函数之后，否则崩溃
        self.last_line = None

        h = .02  # step size in the mesh
        x1_min, x1_max = self.x_train_r[:, 0].min() - 1, self.x_train_r[:, 0].max() + 1
        x2_min, x2_max = self.x_train_r[:, 1].min() - 1, self.x_train_r[:, 1].max() + 1
        self.xx1, self.xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
        self.yy = self.evaluator.clf.predict(np.c_[self.xx1.ravel(), self.xx2.ravel()])
        self.yy = self.yy.reshape(self.xx1.shape)
        self.subplot_train.contourf(self.xx1, self.xx2, self.yy, cmap=plt.cm.get_cmap("Paired"), alpha=0.8)
        self.subplot_train.scatter(self.x_train_r[:, 0], self.x_train_r[:, 1], c=y_train,
                                   cmap=plt.cm.get_cmap("Paired"))
        self.attach_figure(self.figure_train, frame_train)

        y_pred = self.evaluator.pipeline.predict(x_train)
        accuracy = accuracy_score(y_true=y_train, y_pred=y_pred)
        self.console.output("[BC] INIT MODEL: ", str(self.evaluator.pipeline.named_steps['clf']) + "\n")
        self.console.output("[BC] INIT ACCURACY: ", str(accuracy) + "\n")

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
        for i in range(feature_num):
            canv.create_window(60, (i + 1) * 40, window=Tk.Label(canv, text=str(i + 1) + ". "))
            min_x = np.min(x_train[:, i])
            max_x = np.max(x_train[:, i])
            self.slides[i] = Tk.Scale(canv, from_=min_x, to=max_x, resolution=(max_x - min_x) / 100.0,
                                      orient=Tk.HORIZONTAL, command=self.predict)
            canv.create_window(200, (i + 1) * 40, window=self.slides[i])

    # 根据即特征值，计算归属类别的概率
    def predict(self, trivial):
        feature_num = self.x_train.shape[1]
        x = np.arange(feature_num, dtype='f').reshape((1, feature_num))
        for i in range(feature_num):
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
        self.console.output("[BC] OPTIMIZATION START...", "\n")

        # 计算旧模型（即初始模型）的交叉验证精度
        old_scores = cross_validation.cross_val_score(estimator=self.evaluator.pipeline, X=self.x_train, y=self.y_train,
                                                      scoring='accuracy',
                                                      cv=10, n_jobs=-1)
        old_score = np.mean(old_scores)

        # 计算新模型们中最好的交叉验证精度
        new_score = -1.0
        self.new_estimator = None
        for clf, param_grid in ParameterSettings.possible_models:
            self.console.output("[BC] SEARCH MODEL:", str(clf))
            estimator = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=2)), ('clf', clf)])
            gs = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)
            gs = gs.fit(self.x_train, self.y_train)
            if new_score < gs.best_score_:
                new_score = gs.best_score_
                self.new_estimator = gs.best_estimator_

        if new_score > old_score:
            self.label_tips.config(
                text='Found a new model with improvement: %.2f%%' % (100.0 * (new_score - old_score) / old_score))
            self.button_opt.config(text='应用', command=self.apply_new_estimator)
        else:
            self.label_tips.config(text="No better model.")

        self.console.output("[BC] OPTIMIZATION COMPLETE!", "\n")
        self.console.output("[BC] RESULT: ", "old_model_accuracy=%f, new_model_accuracy=%f, improvement=%.2f%%\n" % (
        old_score, new_score, 100.0 * (new_score - old_score) / old_score))

    def apply_new_estimator(self):
        self.console.output("[BC] APPLY MODEL:",
                            "old_model=%s \n new_model=%s\n" % (self.evaluator.pipeline, self.new_estimator))
        self.label_tips.config(text="New model has been applied.")
        self.evaluator.pipeline = self.new_estimator
        self.last_line = None
        self.subplot_train.cla()
        self.predict(None)
        self.yy = self.evaluator.pipeline.named_steps['clf'].predict(np.c_[self.xx1.ravel(), self.xx2.ravel()])
        self.yy = self.yy.reshape(self.xx1.shape)
        self.subplot_train.contourf(self.xx1, self.xx2, self.yy, cmap=plt.cm.get_cmap("Paired"), alpha=0.8)
        self.subplot_train.scatter(self.x_train_r[:, 0], self.x_train_r[:, 1], c=self.y_train,
                                   cmap=plt.cm.get_cmap("Paired"))
        self.figure_train.canvas.draw()


# 各个分类器和对应的参数值列表
class ParameterSettings:
    def __init__(self):
        pass

    clf_lr = LogisticRegression()
    param_lr = {'clf__C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}

    clf_svm_linear = SVC(kernel="linear", probability=True)
    param_svm_linear = {'clf__C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                        'clf__kernel': ['linear']}

    clf_svm_poly = SVC(kernel="poly", probability=True)
    param_svm_poly = {'clf__C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                      'clf__degree': [2, 3, 4],
                      'clf__kernel': ['poly']}

    clf_svm_rbf = SVC(kernel="rbf", probability=True)
    param_svm_rbf = {'clf__C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                     'clf__gamma': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                     'clf__kernel': ['rbf']}

    clf_rf = RandomForestClassifier()
    param_rf = {'clf__n_estimators': [10, 20, 50, 100, 150, 200],
                'clf__criterion': ["gini", "entropy"],
                'clf__max_features': ["auto", "log2", None],
                'clf__max_depth': [None, 2, 3, 4, 5, 6],
                'clf__min_samples_split': [2, 3],
                'clf__min_samples_leaf': [1, 2, 3],
                'clf__min_weight_fraction_leaf': [0],
                'clf__max_leaf_nodes': [None]}

    possible_models = [(clf_lr, param_lr),
                       (clf_svm_linear, param_svm_linear),
                       (clf_svm_poly, param_svm_poly),
                       (clf_svm_rbf, param_svm_rbf),
                       (clf_rf, param_rf)]

    possible_models = [
        (clf_lr, param_lr),
        (clf_svm_rbf, param_svm_rbf)]
