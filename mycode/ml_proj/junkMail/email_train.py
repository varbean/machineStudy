import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

#读取需要训练的数据
from sklearn.naive_bayes import BernoulliNB

data=pd.read_csv("data/finish_result",encoding="utf-8")
Y=data["label"]
X=data.drop("label",1)

train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.2,random_state=0)

print("训练数据集大小:%d" % train_x.shape[0])
print("测试数据集大小:%d" % test_x.shape[0])

nb=BernoulliNB(alpha=1.0,binarize=0.0005)
modelNB=nb.fit(train_x,train_y)

pre_y=modelNB.predict(test_x)

print("准确率为:%.5f" % precision_score(test_y, pre_y))
print("召回率为:%.5f" % recall_score(test_y, pre_y))
print("F1值为:%.5f" % f1_score(test_y, pre_y))




