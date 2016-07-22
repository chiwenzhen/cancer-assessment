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
import matplotlib.pyplot as plt
import numpy as np


class BreastCancerEvaluator:

    # 载入数据集
    def load_data(self, train_data_path ="wdbc.data"):
        df = pd.read_csv(train_data_path, header=None)
        X = df.loc[:, 2:].values  # 训练集特征
        y = df.loc[:, 1].values  # 训练集标签
        le = LabelEncoder()
        y = le.fit_transform(y)  # 把字符串标签转换为整数，恶性M-1，良性B-0
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.20,
                                                                                random_state=1)  # 拆分成训练集(80%)和测试集(20%)

    # 训练模型
    def train(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.clf = LogisticRegression(random_state=1)
        estimators = [('scl', self.scaler), ('pca', self.pca), ('clf', self.clf)]
        self.pipeline = Pipeline(estimators)  # 可以通过pipe_lr.named_steps['pca']来访问PCA对象
        self.pipeline.fit(self.X_train, self.y_train)

    # 评估模型
    def score(self):
        return self.pipeline.score(self.X_test, self.y_test)

    # 保存模型
    def dump_model(self):
        joblib.dump(self.pipeline, "train.model")

    # 获取降维后的数据
    def get_train_data_r(self):
        X_train_decomp = self.scaler.transform(self.X_train)
        X_train_decomp = self.pca.transform(X_train_decomp)
        return X_train_decomp, self.y_train

    # 获取训练后的分类器参数
    def get_clf(self):
        return self.clf

    # 輸入原始特征進行分類
    def predict(self, X):
        X = self.scaler.transform(X)
        X = self.pca.transform(X)
        return self.clf.predict_proba(X), X

    # 输入降维后特征进行分类
    def predict_reduced(self, X):
        X = self.scaler.transform(X)
        X = self.pca.transform(X)
        return self.clf.predict_proba(X), X

    # 特征降维
    def reduce(self, X):
        X = self.scaler.transform(X)
        X = self.pca.transform(X)
        return X
