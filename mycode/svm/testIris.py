import numpy as np
import pandas as pd
import matplotlib as mlt
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

path="../.././teacher/svm/datas/iris.data"
data = pd.read_csv(path, header=None)
x, y = data[list(range(4))], data[4]
y = pd.Categorical(y).codes #把文本数据进行编码，比如a b c编码为 0 1 2
x = x[[0, 1]]
#提取x,y

#生成训练集 测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=28, train_size=0.3,test_size=0.3)

print(x_train)
print("=========================")
print(x_test)
print("=========================")
print(y_train)
print("=========================")
print(y_test)
print("=========================")
#创建SVM模型


## 数据SVM分类器构建
clf = svm.SVC(C=1,kernel='rbf',gamma=0.1)
#gamma值越大，训练集的拟合就越好，但是会造成过拟合，导致测试集拟合变差
#gamma值越小，模型的泛化能力越好，训练集和测试集的拟合相近，但是会导致训练集出现欠拟合问题，
#从而，准确率变低，导致测试集准确率也变低。
## 模型训练
clf.fit(x_train, y_train)


## 计算模型的准确率/精度
print (clf.score(x_train, y_train))
print ('训练集准确率：', accuracy_score(y_train, clf.predict(x_train)))
print (clf.score(x_test, y_test))
print ('测试集准确率：', accuracy_score(y_test, clf.predict(x_test)))

## 计算决策函数的结构值以及预测值(decision_function计算的是样本x到各个分割平面的距离<也就是决策函数的值>)
print ('decision_function:\n', clf.decision_function(x_train))
print ('\npredict:\n', clf.predict(x_train))



