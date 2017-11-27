import time
import matplotlib as ml
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data=pd.read_csv("data/household_power_consumption_1000.txt",sep=";")
#清洗数据 (去掉异常数据)

data.replace("?",np.nan)
data.replace(" ",np.nan)
data.dropna(how="any",axis=1)

#获取x,y数据
dataX=data.iloc[:,2:4]
dataY=data["Global_intensity"]

#获得训练集和测试集
train_X,test_X,train_Y,test_Y=train_test_split(dataX,dataY,test_size=0.3,random_state=1)

#对集合进行标准化
ss=StandardScaler()
train_X=ss.fit_transform(train_X)
test_X=ss.transform(test_X)

#模型训练完毕
lr=LinearRegression()
lr.fit(train_X,train_Y)

#打印模型分数
print(lr.score(test_X,test_Y))
#获得预测集
predict_Y=lr.predict(test_X)

#利用图形来比较测试集和预测集的差别
ml.rcParams['font.sans-serif']=[u'simHei']
ml.rcParams['axes.unicode_minus']=False

x=np.arange(len(test_X))
plt.plot(x,predict_Y,color="black",label=u"预测值")
plt.plot(x,test_Y,color="r",label=u"真实值")


plt.title("功率与电流之间的预测值和真实值的比较")
plt.legend(loc="lower right")
plt.show()






