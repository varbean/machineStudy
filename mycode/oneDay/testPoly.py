import sklearn
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sympy.functions.special.error_functions import Li


def test(x):
    a=time.strptime("".join(x),"%d/%m/%Y%H:%M:%S")
    return (a.tm_year,a.tm_mon,a.tm_mday,a.tm_hour,a.tm_min,a.tm_sec)

d=pd.read_csv("data/household_power_consumption_1000.txt",sep=";")

d=d.replace("?",np.nan).dropna(axis=0,how="any")
data=d.iloc[:,0:2]
dataX=data.apply(lambda x:pd.Series(test(x)),axis=1)
dataY=d["Voltage"]


test_X,train_X,test_Y,train_Y=train_test_split(dataX,dataY,test_size=0.2,random_state=0)


sta=StandardScaler()
sta.fit_transform(test_X)
sta.transform(train_X)

#前面与之前无益，后面增加了管道，供选择
models=[
    Pipeline([
        ("Poly",PolynomialFeatures()),
        ("Linear",LinearRegression(fit_intercept=False))
    ])
]

#选择第一种
model=models[0]

#4阶多项式，可以用循环遍历 选出分数最高的
model.set_params(Poly__degree=4)
model.fit(train_X,train_Y)

pre_Y=model.predict(test_X)
print(model.score(test_X,test_Y))

