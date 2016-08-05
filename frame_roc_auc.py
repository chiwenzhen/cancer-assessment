# coding=utf-8
import numpy as np
import Tkinter as Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from scipy import interp


class RocAucFrame(Tk.Frame):
    def __init__(self, master, x_train, y_train, x_test, y_test, evaluator):
        Tk.Frame.__init__(self, master)
        frame_roc = Tk.Frame(self)
        frame_roc.pack(fill='x', expand=1, padx=15, pady=15)
        cv = StratifiedKFold(y_train, n_folds=3, random_state=1)
        figure_roc = Figure(figsize=(6, 6), dpi=100)
        subplot_roc = figure_roc.add_subplot(111)

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)

        for i, (train, test) in enumerate(cv):
            probas = evaluator.pipeline.fit(x_train[train], y_train[train]).predict_proba(x_train[test])
            fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            subplot_roc.plot(fpr, tpr, linewidth=1, label='ROC fold %d (area = %0.2f)' % (i + 1, roc_auc))

        subplot_roc.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='random guessing')

        mean_tpr /= len(cv)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        subplot_roc.plot(mean_fpr, mean_tpr, 'k--', label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
        subplot_roc.plot([0, 0, 1], [0, 1, 1], lw=2, linestyle=':', color='black', label='perfect performance')
        subplot_roc.set_xlim([-0.05, 1.05])
        subplot_roc.set_ylim([-0.05, 1.05])
        subplot_roc.set_xlabel('false positive rate')
        subplot_roc.set_ylabel('true positive rate')
        subplot_roc.set_title('Receiver Operator Characteristic')
        subplot_roc.legend(loc="lower right")
        self.attach_figure(figure_roc, frame_roc)

    @staticmethod
    def attach_figure(figure, frame):
        canvas = FigureCanvasTkAgg(figure, master=frame)  # 内嵌散点图到UI
        canvas.show()
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        toolbar = NavigationToolbar2TkAgg(canvas, frame)  # 内嵌散点图工具栏到UI
        toolbar.update()
        canvas.tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
