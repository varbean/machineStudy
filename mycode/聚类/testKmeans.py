import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import sklearn.datasets as ds
from pandas import Series,DataFrame
import matplotlib.colors
from sklearn.cluster import KMeans

#处理图表中文乱码
mpl.rcParams["font.sans-serif"]=[u"SimHei"]
mpl.rcParams["axes.unicode_minus"]=False

N=1500
ct=4

data,Y=ds.make_blobs(n_samples=N,n_features=2,centers=ct,random_state=28)
# df=DataFrame(data)
# dy=DataFrame(Y)
# #清洗数据
# data=df.replace(" ",np.nan).dropna(how="any")
# Y=dy.replace(" ",np.nan).dropna(how="any")

#模型构建
km=KMeans(n_clusters=ct,random_state=28)
km.fit(data,Y)

pre_y=km.predict(data)

print ("所有样本距离聚簇中心点的总距离和:", km.inertia_)

def expandBorder(a, b):
    d = (b - a) * 0.1
    return a-d, b+d


#画图
cm = mpl.colors.ListedColormap(list('rgbmyc'))
plt.figure(figsize=(15,9),facecolor="w")
plt.subplot(241)
plt.scatter(data[:, 0], data[:, 1], c=Y, s=30, cmap=cm, edgecolors='none')

x1_min, x2_min = np.min(data, axis=0)
x1_max, x2_max = np.max(data, axis=0)
x1_min, x1_max = expandBorder(x1_min, x1_max)
x2_min, x2_max = expandBorder(x2_min, x2_max)
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.title(u'原始数据')
plt.grid(True)

plt.subplot(242)
plt.scatter(data[:, 0], data[:, 1], c=pre_y, s=30, cmap=cm, edgecolors='none')
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.title(u'K-Means算法聚类结果')
plt.grid(True)

plt.tight_layout(2, rect=(0, 0, 1, 0.97))
plt.suptitle(u'数据分布对KMeans聚类的影响', fontsize=18)
plt.show()
