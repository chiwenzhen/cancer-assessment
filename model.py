# -*- coding: utf-8 -*-
"""
chiwenzhen
2016-07-20
@ruijie

"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.svm import SVC


class Evaluator:
    def __init__(self, scaler=None, pca=None, clf=None):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = scaler
        self.pca = pca
        self.clf = clf
        self.estimators = []
        if self.scaler is not None:
            self.estimators.append(('scl', self.scaler))
        if self.pca is not None:
            self.estimators.append(('pca', self.pca))
        if self.clf is not None:
            self.estimators.append(('clf', self.clf))
        self.pipeline = Pipeline(self.estimators)  # 可以通过pipe.named_steps['pca']来访问PCA对象

    def load_data(self, x_train, y_train, x_test, y_test):
        self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test

    # 训练模型
    def train(self):
        self.pipeline.fit(self.x_train, self.y_train)
        pass
        pass

    # 评估模型
    def score(self):
        return self.pipeline.score(self.x_test, self.y_test)

    # 保存模型
    def dump_model(self):
        joblib.dump(self.pipeline, "train.model")

    # 获取分类器
    def get_clf(self):
        return self.clf

    # 輸入原始特征進行分類
    def predict(self, x):
        x = self.scaler.transform(x)
        x = self.pca.transform(x)
        return self.clf.predict_proba(x)

    # 输入降维后特征进行分类
    def predict_r(self, x):
        return self.clf.predict_proba(x), x

    # 特征降维
    def reduce(self, x):
        x = self.scaler.transform(x)
        x = self.pca.transform(x)
        return x

    # 获取训练数据
    def get_train_data(self):
        return self.x_train, self.y_train

    # 获取测试数据
    def get_test_data(self):
        return self.x_test, self.y_test

    # 获取流水线评估器
    def get_pipeline(self):
        return self.pipeline


# 载入乳腺肿瘤数据
class BreastCancerDataSet:
    def __init__(self, data_path="wdbc.data", test_percent=0.2):
        df = pd.read_csv(data_path, header=None)
        self.x = df.loc[:, 2:].values  # 训练集特征
        y = df.loc[:, 1].values  # 训练集标签
        le = LabelEncoder()
        self.y = le.fit_transform(y)  # 把字符串标签转换为整数，恶性M-1，良性B-0
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=test_percent,
                                                                                random_state=1)  # 拆分成训练集(80%)和测试集(20%)
        self.feature_num = le.classes_.shape[0]


# 载入胎心监护数据
class CardiotocographyDataSet:
    def __init__(self, data_path="CTG.xls", test_percent=0.2):
        df = pd.read_excel(data_path, header=1, sheetname=1)
        df = df[0:2125]                 # 选择0-2125行
        x = df.iloc[:, 10:31].values    # 10-30列为训练集特征
        y = df.iloc[:, 45].values       # 45列为训练集标签
        keeped_cols = range(10, 31) + [45]
        df = df.iloc[:, keeped_cols]
        le = LabelEncoder()
        y = le.fit_transform(y)  # 把字符串标签转换为整数，恶性M-1，良性B-0
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_percent,
                                                                                random_state=1)  # 拆分成训练集(80%)和测试集(20%)
        self.x, self.y = x, y
        self.feature_num = le.classes_.shape[0]
        self.df = df
