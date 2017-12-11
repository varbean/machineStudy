import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from  sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
#解决画图乱码问题
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

#忽略警告信息
warnings.filterwarnings(action = 'ignore', category=ConvergenceWarning)



path = "breast-cancer-wisconsin.data"
names = ['id','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape',
         'Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei',
        'Bland Chromatin','Normal Nucleoli','Mitoses','Class']

df = pd.read_csv(path, header=None,names=names)

#数据清洗
datas = df.replace('?', np.nan).dropna(how = 'any')
datas.head(5)



X = datas[names[1:10]]
Y = datas[names[10]]

#获取训练集和测试集
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,random_state=0)



#数据标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train) ## 训练模型及标准化数据


#交叉验证,获得最优模型 这些参数?
lr = LogisticRegressionCV(fit_intercept=True, Cs=np.logspace(-2, 2, 20), cv=2, penalty='l2', solver='lbfgs', tol=0.01)
lr.fit(X_train, Y_train)



r = lr.score(X_train, Y_train)
print ("R值（准确率）：", r)
print ("稀疏化特征比率：%.2f%%" % (np.mean(lr.coef_.ravel() == 0) * 100))  #拉伸
print ("参数：",lr.coef_)
print ("截距：",lr.intercept_)



X_test = ss.transform(X_test)

Y_predict = lr.predict(X_test)


#对真实值和预测值画图比较
x_len = range(len(X_test))
plt.figure(figsize=(14,7), facecolor='w')
plt.ylim(0,6)
plt.plot(x_len, Y_test, 'ro',markersize = 8, zorder=3, label=u'真实值')
plt.plot(x_len, Y_predict, 'go', markersize = 14, zorder=2, label=u'预测值,$R^2$=%.3f' % lr.score(X_test, Y_test))
plt.legend(loc = 'upper left')
plt.xlabel(u'数据编号', fontsize=18)
plt.ylabel(u'乳腺癌类型', fontsize=18)
plt.title(u'Logistic回归算法对数据进行分类', fontsize=20)
plt.show()