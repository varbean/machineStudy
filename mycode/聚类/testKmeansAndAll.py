import numpy as np
import pandas as pd
import time
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import cluster
from sklearn import datasets as ds
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

mpl.rcParams["font.sans-serif"]=[u"SimHei"]
mpl.rcParams["axes.unicode_minus"]=False
warnings.filterwarnings(action='ignore', category=UserWarning)

#生成数据
n=1500
np.random.seed(0)
n_cir=ds.make_circles(n_samples=n,factor=.5,noise=.05) #圆形
n_moons=ds.make_moons(n_samples=n,noise=.05)#月牙形

n_blobs=ds.make_blobs(n_samples=n,cluster_std=0.5,random_state=0)#高斯分布
no_str=np.random.rand(n,2),None #不清楚什么形状数据

datas=[n_cir,n_moons,n_blobs,no_str] #数据集
clusters=[2,2,3,2]#分类数集

cluster_names=[
    "KMeans","MiniBatchKMeans","AC-ward","AC-average",
    "Birch","DBSCAN","SpectralClustering"
               ]#几种聚类算法

#开始画图
plt.figure(figsize=(len(cluster_names)*2+3,9.5),facecolor="w")

plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)

colors=np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors=np.hstack([colors] * 20)

plot_num=1
for i_dataset,(dataset,n_cluster) in enumerate(zip(datas,clusters)):
    X,y=dataset
    X=StandardScaler().fit_transform(X)

    connectivity=kneighbors_graph(X,n_neighbors=10,include_self=False)
    connectivity=0.5*(connectivity+connectivity.T)

    km=cluster.KMeans(n_clusters=n_cluster)#kmeans
    mbkm=cluster.MiniBatchKMeans(n_clusters=n_cluster)
    ward=cluster.AgglomerativeClustering(n_clusters=n_cluster,connectivity=connectivity,linkage="ward")
    average=cluster.AgglomerativeClustering(n_clusters=n_cluster,connectivity=connectivity,linkage="average")
    birch=cluster.Birch(n_clusters=n_cluster)
    dbscan=cluster.DBSCAN(eps=.2)
    spectral=cluster.SpectralClustering(n_clusters=n_cluster,eigen_solver="arpack",affinity="nearest_neighbors") #不清楚方法的含义

    clustering_algorithms = [km, mbkm, ward, average, birch, dbscan, spectral] #算法集
    for name,algorithm in zip(cluster_names,clustering_algorithms):
        t0=time.time()
        algorithm.fit(X)
        t1=time.time()
        if hasattr(algorithm, 'labels_'):#hasattr 是什么方法
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

    #开始画图
    plt.subplot(4,len(clustering_algorithms),plot_num)
    if i_dataset==0:
        plt.title(name,size=18)
    plt.scatter(X[:,0],X[:,1],color=colors[y_pred].tolist(), s=10)
    if hasattr(algorithm, 'cluster_centers_'):
        centers = algorithm.cluster_centers_
        center_colors = colors[:len(centers)]
        plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xticks(())
    plt.yticks(())
    plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
             transform=plt.gca().transAxes, size=15,
             horizontalalignment='right')
    plot_num += 1

plt.show()
