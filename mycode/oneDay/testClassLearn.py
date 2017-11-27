import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from  sklearn.preprocessing import StandardScaler

#格式化日期
def test(x):
    s=time.strptime("".join(x),"%d/%m/%Y%H:%M:%S")
    return (s.tm_year,s.tm_mon,s.tm_mday,s.tm_hour,s.tm_min,s.tm_sec)
#读入数据
r=pd.read_csv("data/household_power_consumption_1000.txt",sep=";") #sep 为分隔符
dataR=r.iloc[:,0:2] #提取全部行，前两列的数据
dataR=dataR.replace("?",np.nan).dropna(axis=0,how="any") #经过测试确认 先将?的值转成nan,再将含有nan的整行都删除


#生成好数据,
dataX=dataR.apply(lambda x:pd.Series(test(x)),axis=1) #处理时间
dataY=r["Global_active_power"]

#划分训练集，和测试集 (将数据拆分)
train_x,test_x,train_y,test_y=train_test_split(dataX,dataY,test_size=0.3,random_state=1)

#标准化 将数据准备充分
ss=StandardScaler()
train_x=ss.fit_transform(train_x)
test_x=ss.transform(test_x)

#构建模型
lr=LinearRegression()
lr.fit(train_x,train_y)

#保存模型 但是ss为啥要保存 还得试
# joblib.dump(ss,"data_ss.model")
# joblib.dump(lr,"data_lr.model")

#读取模型
# joblib.load("data_ss.model")
# joblib.load("data_lr.model")

#根据模型  生成测试数据 x 对应的预测y
predict_y=lr.predict(test_x)

print(lr.score(test_x,test_y))

#图形展示 预测值和实际值差别
t=np.arange(len(test_x))
plt.figure(facecolor='w')
plt.plot(t, test_y, 'r-', linewidth=2, label=u'真实值')
plt.plot(t, predict_y, 'g-', linewidth=2, label=u'预测值')
plt.legend(loc = 'lower right')
plt.title(u"线性回归预测时间和功率之间的关系", fontsize=20)
plt.grid(b=True)
plt.show()