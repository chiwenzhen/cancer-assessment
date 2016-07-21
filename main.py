# -*- coding: utf-8 -*-
"""
chiwenzhen 
@ruijie 2016-07-16

"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

# 载入数据集
df = pd.read_csv('wdbc.data', header=None) 
X = df.loc[:, 2:].values # 训练集特征
y = df.loc[:, 1].values # 训练集标签
le = LabelEncoder()
y = le.fit_transform(y) # 把字符串标签转换为整数，恶性M-1，良性B-0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1) # 拆分成训练集(80%)和测试集(20%)


# 训练模型
scaler = StandardScaler()
pca = PCA(n_components=2)
lr = LogisticRegression(random_state=1)
estimators = [('scl', scaler), ('pca', pca), ('clf', lr)] 
pipe_lr = Pipeline(estimators) #可以通过pipe_lr.named_steps['pca']来访问PCA对象
pipe_lr.fit(X_train, y_train)

# 评估模型
print(pipe_lr.score(X_test, y_test))
print(lr.coef_)

# 保存模型
#from sklearn.externals import joblib
#joblib.dump(pipe_lr, "D:/pipelr.model")
