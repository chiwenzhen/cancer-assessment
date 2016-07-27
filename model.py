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


class LREvaluator:
    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.clf = LogisticRegression(penalty='l2', random_state=1)
        self.estimators = [('scl', self.scaler), ('pca', self.pca), ('clf', self.clf)]
        self.pipeline = Pipeline(self.estimators)  # 可以通过pipe.named_steps['pca']来访问PCA对象

    # 载入数据集
    def load_data(self, x, y):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.20,
                                                                                random_state=1)  # 拆分成训练集(80%)和测试集(20%)

    # 训练模型
    def train(self):
        self.pipeline.fit(self.x_train, self.y_train)

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
        x = self.scaler.transform(x)
        x = self.pca.transform(x)
        return self.clf.predict_proba(x), x

    # 特征降维
    def reduce(self, x):
        x = self.scaler.transform(x)
        x = self.pca.transform(x)
        return x

    # 获取训练数据
    def get_train_data(self):
        return self.x_train, self.y_train

    # 获取流水线评估器
    def get_pipeline(self):
        return self.pipeline


# 载入数据
class DataSet:
    def __init__(self):
        self.x = None
        self.y = None

    def load_data(self, data_path="wdbc.data"):
        df = pd.read_csv(data_path, header=None)
        self.x = df.loc[:, 2:].values  # 训练集特征
        y = df.loc[:, 1].values  # 训练集标签
        le = LabelEncoder()
        self.y = le.fit_transform(y)  # 把字符串标签转换为整数，恶性M-1，良性B-0

    def get_data(self):
        return self.x, self.y

    def get_data_x(self):
        return self.x

    def get_data_y(self):
        return self.y
