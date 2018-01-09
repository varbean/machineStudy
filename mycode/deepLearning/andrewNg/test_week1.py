import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes=load_dataset()

#print(train_set_x_orig[0])
#查看每张图片
#plt.imshow(train_set_x_orig[0])
#plt.show()

#获取训练集和测试集的量以及图片大小
m_train=train_set_x_orig.shape[0]
m_test=test_set_x_orig.shape[0]
num_px=train_set_x_orig.shape[1]

# print("训练集："+str(m_train))
# print("测试集："+str(m_test))
# print("图片大小为："+str(num_px))

#将测试、训练集 统一化 方便计算 1
train_set_x_flatten=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten=test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
# print(train_set_x_flatten.shape)
# print(test_set_x_flatten.shape)

#规范化数据集，方便计算2
"""
    为了表示颜色图像，必须为每个像素指定红色、绿色和蓝色通道(RGB)，因此像素值实际上是一个从0到255的三个数字的向量
    机器学习中一个常见的预处理步骤是对数据集进行集中和标准化，这意味着您可以从每个示例中派生出整个numpy数组的平均值，、
    然后根据整个numpy数组的标准偏差来划分每个实例。但是对于图片集来说，它更简单、更方便，而且几乎可以将数据集的
    每一行除以255(一个像素通道的最大值)。
    规范我们的数据集  虽然还是不懂原理 应该是避免数值相乘太大，为了运算方便吧。
"""
train_set_x=train_set_x_flatten/255
test_set_x=test_set_x_flatten/255

# print(train_set_x.shape)
# print(test_set_x.shape)


