# -*- coding: utf-8 -*-
"""
chiwenzhen
2016-07-20
@ruijie

"""
import numpy as np
import sys

import Tkinter as Tk
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
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


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
        menubar = Tk.Menu(root)  # 添加菜单
        root.config(menu=menubar)
        filemenu = Tk.Menu(menubar)
        filemenu.add_command(label="Exit", command=sys.exit)
        menubar.add_cascade(label="File", menu=filemenu)

        notebook = Notebook(root)  # 添加标签页
        notebook.pack(fill=Tk.BOTH)

        page_1 = Tk.Frame(notebook)
        notebook.add(page_1, text="Main")

        page_2 = Tk.Frame(notebook)
        notebook.add(page_2, text="Learning Curve")

        page_3 = Tk.Frame(notebook)
        notebook.add(page_3, text="Validation Curve")

        page_4 = Tk.Frame(notebook)
        notebook.add(page_4, text="ROC & AUC")

        page_5 = Tk.Frame(notebook)
        notebook.add(page_5, text="Testing Result")

        page_6 = Tk.Frame(notebook)
        notebook.add(page_6, text="GridSearchCV")

        page_6 = Tk.Frame(notebook)
        notebook.add(page_6, text="GridSearchCV")

        page_7 = Tk.Frame(notebook)
        notebook.add(page_7, text="t-SNE")

        # # 第1页 1.matplotlib绘制
        # frame_x_y = Tk.Frame(page_1)
        # frame_x_y.pack(fill=Tk.BOTH, expand=1, padx=15, pady=15)
        # self.figure = Figure(figsize=(5, 4), dpi=100)
        # self.subplot = self.figure.add_subplot(111)
        # self.figure.tight_layout()  # 一定要放在add_subplot函数之后，否则崩溃
        # self.subplot.set_title('Breast Cancer Evaluation Model')
        # self.last_line = None
        #
        # h = .02  # step size in the mesh
        # x1_min, x1_max = x_train_r[:, 0].min() - 1, x_train_r[:, 0].max() + 1
        # x2_min, x2_max = x_train_r[:, 1].min() - 1, x_train_r[:, 1].max() + 1
        # xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
        # yy = self.evaluator.clf.predict(np.c_[xx1.ravel(), xx2.ravel()])
        # yy = yy.reshape(xx1.shape)
        # self.subplot.contourf(xx1, xx2, yy, cmap=plt.cm.Paired, alpha=0.8)
        # self.subplot.scatter(x_train_r[:, 0], x_train_r[:, 1], c=y_train, cmap=plt.cm.Paired)
        # self.attach_figure(self.figure, frame_x_y)
        #
        # # 第1页 2.概率输出框
        # frame_output = Tk.Frame(page_1)
        # frame_output.pack(fill=Tk.BOTH, expand=1, padx=5, pady=5)
        # self.malig_prob = Tk.StringVar()
        # Tk.Label(frame_output, text="malignant prob").pack(side=Tk.LEFT)
        # Tk.Entry(frame_output, textvariable=self.malig_prob, bd=5).pack(side=Tk.LEFT, padx=5, pady=5)
        #
        # # 第1页 3.滑动条
        # frame_scale = Tk.Frame(page_1)
        # frame_scale.pack(fill=Tk.BOTH, expand=1, padx=5, pady=5)
        # canv = Tk.Canvas(frame_scale, relief=Tk.SUNKEN)
        # vbar = Tk.Scrollbar(frame_scale, command=canv.yview)
        # canv.config(scrollregion=(0, 0, 300, 1500))
        # canv.config(yscrollcommand=vbar.set)
        # vbar.pack(side=Tk.RIGHT, fill=Tk.Y)
        # canv.pack(side=Tk.LEFT, expand=Tk.YES, fill=Tk.BOTH)
        # feature_num = x_train.shape[1]
        # self.slides = [None] * feature_num  # 滑动条个数为特征个数
        # feature_names = ["radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave",
        #                  "symmetry", "fractal",
        #                  "radius SE", "texture SE", "perimeter SE", "area SE", "smoothness SE", "compactness SE",
        #                  "concavity SE", "concave SE",
        #                  "symmetry SE", "fractal SE",
        #                  "radius MAX", "texture MAX", "perimeter MAX", "area MAX", "smoothness MAX", "compactness MAX",
        #                  "concavity MAX", "concave MAX",
        #                  "symmetry MAX", "fractal MAX"]
        # for i in range(feature_num):
        #     canv.create_window(60, (i + 1) * 40, window=Tk.Label(canv, text=str(i + 1) + ". " + feature_names[i]))
        #     min_x = np.min(x_train[:, i])
        #     max_x = np.max(x_train[:, i])
        #     self.slides[i] = Tk.Scale(canv, from_=min_x, to=max_x, resolution=(max_x - min_x) / 100.0,
        #                               orient=Tk.HORIZONTAL, command=self.predict)
        #     canv.create_window(200, (i + 1) * 40, window=self.slides[i])
        #
        # # 第2页 1.学习曲线
        # evaluator_lcurve = CancerEvaluator()
        # train_sizes, train_scores, test_scores = learning_curve(estimator=evaluator_lcurve.pipeline,
        #                                                         X=x_train,
        #                                                         y=y_train,
        #                                                         train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)
        #
        # train_mean = np.mean(train_scores, axis=1)
        # train_std = np.std(train_scores, axis=1)
        # test_mean = np.mean(test_scores, axis=1)
        # test_std = np.std(test_scores, axis=1)
        # frame_lcurve = Tk.Frame(page_2)
        # frame_lcurve.pack(fill="x", expand=1, padx=15, pady=15)
        # figure_lcurve = Figure(figsize=(6, 6), dpi=100)
        # subplot_lcurve = figure_lcurve.add_subplot(111)
        # subplot_lcurve.plot(train_sizes, train_mean, color="blue", marker='o', markersize=5,
        #                     label="training accuracy")
        # subplot_lcurve.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15,
        #                             color="blue")
        # subplot_lcurve.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5,
        #                     label="cross-validation accuracy")
        # subplot_lcurve.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15,
        #                             color="green")
        # subplot_lcurve.grid()
        # subplot_lcurve.set_xlabel("Number of training samples")
        # subplot_lcurve.set_ylabel("Accuracy")
        # subplot_lcurve.legend(loc="lower right")
        # subplot_lcurve.set_ylim([0.8, 1.0])
        # self.attach_figure(figure_lcurve, frame_lcurve)
        #
        # # 第3页 验证曲线
        # evaluator_vcurve = CancerEvaluator()
        # param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        # train_scores, test_scores = validation_curve(estimator=evaluator_vcurve.pipeline,
        #                                              X=x_train, y=y_train,
        #                                              param_name='clf__gamma',
        #                                              param_range=param_range, cv=10)
        #
        # train_mean = np.mean(train_scores, axis=1)
        # train_std = np.std(train_scores, axis=1)
        # test_mean = np.mean(test_scores, axis=1)
        # test_std = np.std(test_scores, axis=1)
        #
        # frame_vcurve = Tk.Frame(page_3)
        # frame_vcurve.pack(fill='x', expand=1, padx=15, pady=15)
        # figure_vcurve = Figure(figsize=(6, 6), dpi=100)
        # subplot_vcurve = figure_vcurve.add_subplot(111)
        #
        # subplot_vcurve.plot(param_range, train_mean, color='blue', marker='o', markersize=5,
        #                     label='training accuracy')
        # subplot_vcurve.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15,
        #                             color='blue')
        # subplot_vcurve.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5,
        #                     label='cross-validation accuracy')
        # subplot_vcurve.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15,
        #                             color='green')
        #
        # subplot_vcurve.grid()
        # subplot_vcurve.set_xscale('log')
        # subplot_vcurve.legend(loc='lower right')
        # subplot_vcurve.set_xlabel('Parameter C')
        # subplot_vcurve.set_ylabel('Accuracy')
        # subplot_vcurve.set_ylim([0.91, 1.0])
        # self.attach_figure(figure_vcurve, frame_vcurve)
        #
        # # 第4页 ROC&AUC
        # evaluator_roc = CancerEvaluator()
        # frame_roc = Tk.Frame(page_4)
        # frame_roc.pack(fill='x', expand=1, padx=15, pady=15)
        # cv = StratifiedKFold(y_train, n_folds=3, random_state=1)
        # figure_roc = Figure(figsize=(6, 6), dpi=100)
        # subplot_roc = figure_roc.add_subplot(111)
        #
        # mean_tpr = 0.0
        # mean_fpr = np.linspace(0, 1, 100)
        #
        # for i, (train, test) in enumerate(cv):
        #     evaluator_roc.load_data(x_train, y_train)
        #     probas = evaluator_roc.pipeline.fit(x_train[train], y_train[train]).predict_proba(x_train[test])
        #     fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)
        #     mean_tpr += interp(mean_fpr, fpr, tpr)
        #     mean_tpr[0] = 0.0
        #     roc_auc = auc(fpr, tpr)
        #     subplot_roc.plot(fpr, tpr, linewidth=1, label='ROC fold %d (area = %0.2f)' % (i + 1, roc_auc))
        #
        # subplot_roc.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='random guessing')
        #
        # mean_tpr /= len(cv)
        # mean_tpr[-1] = 1.0
        # mean_auc = auc(mean_fpr, mean_tpr)
        # subplot_roc.plot(mean_fpr, mean_tpr, 'k--', label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
        # subplot_roc.plot([0, 0, 1], [0, 1, 1], lw=2, linestyle=':', color='black', label='perfect performance')
        # subplot_roc.set_xlim([-0.05, 1.05])
        # subplot_roc.set_ylim([-0.05, 1.05])
        # subplot_roc.set_xlabel('false positive rate')
        # subplot_roc.set_ylabel('true positive rate')
        # subplot_roc.set_title('Receiver Operator Characteristic')
        # subplot_roc.legend(loc="lower right")
        # self.attach_figure(figure_roc, frame_roc)
        #
        # # 第5页 1.测试集展示
        # frame_test = Tk.Frame(page_5)
        # frame_test.pack(fill='x', expand=1, padx=15, pady=15)
        # figure_test = Figure(figsize=(4, 4), dpi=100)
        # subplot_test = figure_test.add_subplot(111)
        # subplot_test.set_title('Breast Cancer Testing')
        # figure_test.tight_layout()
        #
        # x1_min, x1_max = x_test_r[:, 0].min() - 1, x_test_r[:, 0].max() + 1
        # x2_min, x2_max = x_test_r[:, 1].min() - 1, x_test_r[:, 1].max() + 1
        # xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
        # yy = self.evaluator.clf.predict(np.c_[xx1.ravel(), xx2.ravel()])
        # yy = yy.reshape(xx1.shape)
        # subplot_test.contourf(xx1, xx2, yy, cmap=plt.cm.Paired, alpha=0.8)
        # subplot_test.scatter(x_test_r[:, 0], x_test_r[:, 1], c=y_test, cmap=plt.cm.Paired)
        # self.attach_figure(figure_test, frame_test)
        #
        # # 第5页 2.测试性能指标 precision recall f_value
        # y_pred = self.evaluator.pipeline.predict(x_test)
        # frame_matrix = Tk.Frame(page_5)
        # frame_matrix.pack(side=Tk.LEFT, fill='x', expand=1, padx=15, pady=15)
        # figure_matrix = Figure(figsize=(4, 4), dpi=100)
        # subplot_matrix = figure_matrix.add_subplot(111)
        #
        # confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
        # subplot_matrix.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
        # for i in range(confmat.shape[0]):
        #     for j in range(confmat.shape[1]):
        #         subplot_matrix.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
        #
        # subplot_matrix.set_xlabel('predicted label')
        # subplot_matrix.set_ylabel('true label')
        # self.attach_figure(figure_matrix, frame_matrix)
        #
        # frame_result = Tk.Frame(page_5)
        # frame_result.pack(side=Tk.LEFT, fill='x', expand=1, padx=15, pady=15)
        # Tk.Label(frame_result, text="Accuracy: ").grid(row=0, column=0, sticky=Tk.W)
        # Tk.Label(frame_result, text=str(self.evaluator.pipeline.score(x_test, y_test))).grid(row=0, column=1,
        #                                                                                      sticky=Tk.W)
        # Tk.Label(frame_result, text="Precision: ").grid(row=1, column=0, sticky=Tk.W)
        # Tk.Label(frame_result, text=str(precision_score(y_true=y_test, y_pred=y_pred))).grid(row=1, column=1,
        #                                                                                      sticky=Tk.W)
        # Tk.Label(frame_result, text="Recall: ").grid(row=2, column=0, sticky=Tk.W)
        # Tk.Label(frame_result, text=str(recall_score(y_true=y_test, y_pred=y_pred))).grid(row=2, column=1, sticky=Tk.W)
        # Tk.Label(frame_result, text="F-value: ").grid(row=3, column=0, sticky=Tk.W)
        # Tk.Label(frame_result, text=str(f1_score(y_true=y_test, y_pred=y_pred))).grid(row=3, column=1, sticky=Tk.W)
        #
        # # 第6页，GridSearchCV
        # evaluator_gs = CancerEvaluator()
        # evaluator_gs.pipeline.named_steps['clf'] = SVC(random_state=1)
        # frame_linear_param = Tk.Frame(page_6)
        # frame_linear_param.pack(fill='x', expand=1, padx=15, pady=15)
        # figure_gs = Figure(figsize=(6, 4), dpi=100)
        # subplot_linear_param = figure_gs.add_subplot(111)
        # figure_gs.tight_layout()
        # subplot_linear_param.set_xscale('log')
        # subplot_linear_param.set_ylim([0.5, 1.0])
        # subplot_linear_param.set_xlabel("C")
        # subplot_linear_param.set_ylabel("Accuracy")
        # subplot_linear_param.set_title("GridSearchCV on parameter C in SVM with linear-kernel")
        #
        # param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        # param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']},
        #               {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}]
        # gs = GridSearchCV(estimator=evaluator_gs.pipeline, param_grid=param_grid, scoring='accuracy', cv=10,
        #                   n_jobs=-1)
        # gs = gs.fit(x_train, y_train)
        # y_true, y_pred = y_test, gs.predict(x_test)
        #
        # lieanr_end = len(param_range)
        # rbf_end = lieanr_end + len(param_range) ** 2
        # subplot_linear_param.plot(map(lambda e: e[0]['clf__C'], gs.grid_scores_[0:lieanr_end]),
        #                           map(lambda e: e[1], gs.grid_scores_[0:lieanr_end]),
        #                           linewidth=1, label='svm_linear', color="blue", marker='o', markersize=5)
        # subplot_linear_param.grid()
        # self.attach_figure(figure_gs, frame_linear_param)
        #
        # frame_rbf_param = Tk.Frame(page_6)
        # frame_rbf_param.pack(fill='x', expand=1, padx=15, pady=15)
        # scrollbar = Tk.Scrollbar(frame_rbf_param)
        # scrollbar.pack(side=Tk.RIGHT, fill=Tk.Y)
        # text_rbf_param = Tk.Text(frame_rbf_param, width=800, wrap='none', yscrollcommand=scrollbar.set)
        # text_rbf_param.insert(Tk.END, "1. Best parameter: " + str(gs.best_params_) + "\n\n")
        # text_rbf_param.insert(Tk.END, "2. Best parameter performance on testing data.\n\n ")
        # text_rbf_param.insert(Tk.END, classification_report(y_true, y_pred))
        # text_rbf_param.insert(Tk.END, "\n\n")
        # text_rbf_param.insert(Tk.END, "3. All parameter searched by GridSearchCV.\n\n ")
        # log = []
        # for params, mean_score, scores in gs.grid_scores_:
        #     log.append(
        #         ["%0.3f" % (mean_score), "(+/-%0.03f)" % (scores.std() * 2), params["clf__C"],
        #          params.has_key("clf__gamma") and params["clf__gamma"] or "",
        #          params["clf__kernel"]])
        # text_rbf_param.insert(Tk.END, tabulate(log, headers=["Accuracy","SD", "C", "gamma", "type"]))
        # text_rbf_param.pack()
        # scrollbar.config(command=text_rbf_param.yview)

        # 第7页 t-SNE
        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train, y_train) # 经过中心化和归一化的训练数据
        y_train_s = y_train
        tsne = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        x_train_s = tsne.fit_transform(x_train_s)
        frame_tsne = Tk.Frame(page_7)
        frame_tsne.pack(fill='x', expand=1, padx=15, pady=15)
        figure_tsne = Figure(figsize=(6, 6), dpi=100)
        subplot_tsne = figure_tsne.add_subplot(111)
        subplot_tsne.scatter(x_train_s[:, 0], x_train_s[:, 1], c=y_train_s, cmap=plt.cm.Paired)
        subplot_tsne.set_title("t-SNE on breast cancer data")
        self.attach_figure(figure_tsne, frame_tsne)

    # 重新绘制点
    def plot_point(self, subplot, x):
        if self.last_line is not None:
            self.last_line.remove()
            del self.last_line
        lines = subplot.plot(x[:, 0], x[:, 1], "ro", label="case")
        self.last_line = lines.pop(0)
        subplot.legend(loc='lower right')

    # 根据即特征值，计算归属类别的概率
    def predict(self, trivial):
        x = np.arange(30, dtype='f').reshape((1, 30))
        for i in range(30):
            x[0, i] = float(self.slides[i].get())
        result = self.evaluator.predict(x)
        self.malig_prob.set("%.2f%%" % (result[0, 1] * 100))  # 恶性肿瘤的概率
        self.plot_point(self.subplot, self.evaluator.reduce(x))
        self.figure.canvas.draw()

    @staticmethod
    def attach_figure(figure, frame):
        canvas = FigureCanvasTkAgg(figure, master=frame)  # 内嵌散点图到UI
        canvas.show()
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        toolbar = NavigationToolbar2TkAgg(canvas, frame)  # 内嵌散点图工具栏到UI
        toolbar.update()
        canvas.tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)


if __name__ == "__main__":
    master = Tk.Tk()
    master.wm_title("Breast Cancer Evaluation Platform")
    master.geometry('900x750')
    master.iconbitmap("cancer.ico")
    app = App(master)
    master.mainloop()
