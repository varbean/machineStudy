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

def sigmoid(x):
    y=1/(1+np.exp(-x) )
    return y
#初始化参数
def initialize_with_zeros(dim):
    w=np.zeros((dim,1))
    b=0

    assert(w.shape==(dim,1))
    return w,b
#向前传播
def propagate(w,b,X,Y):
    #激活函数
    A=sigmoid(np.dot(w.T,X)+b)
    #样本个数
    m=X.shape[1]

    J=-1/m * np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))

    #偏导
    dw=1/m * np.dot(X,((A-Y).T))
    db=1/m * np.sum(A-Y)

    assert(dw.shape==w.shape)
    assert(db.dtype==float)

    J=np.squeeze(J)
    assert(J.dtype==float)
    grads={
        "dw":dw,
        "db":db
    }
    return grads,J

#向后传播 更新参数
def optimize(w,b,X,Y,num_iterators,learning_rate,print_cost=False):
    costs=[]

    for i in range(num_iterators):

        grads,cost=propagate(w,b,X,Y)
        dw=grads["dw"]
        db=grads["db"]
        w=w-learning_rate*dw
        b=b-learning_rate*db
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params={
        "w":w,
        "b":b
    }
    grads={
        "dw":dw,
        "db":db
    }
    return params,grads,costs

w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
params,grads,costs=optimize(w,b,X,Y,num_iterators=100,learning_rate=0.009)
# print(params["w"])
# print(params["b"])
# print(grads["dw"])
# print(grads["db"])
# print(costs)

#获得预期
def predict(w,b,X):
    m=X.shape[1]
    #规定好形状
    w=w.reshape(X.shape[0],1)
    Y_predict=np.zeros((1,m))
    A=sigmoid(np.dot(w.T,X)+b)

    assert(A.shape==(1,m))
    for i in range(A.shape[1]):
        if A[0,i]>=0.5:
            Y_predict[0,i]=1
        else:
            Y_predict[0,i]=0
    assert (Y_predict.shape == (1, m))
    return Y_predict


print("pre:"+str(predict(w,b,X)))

#构建模型
def model(X_train,Y_train,X_test,Y_test,num_iterations=2000,learning_rate=0.5,print_cost=True):
    w,b=initialize_with_zeros(X_train.shape[0]) #初始化


    params,grads,costs=optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    #获得更新后的参数
    w=params["w"]
    b=params["b"]

    Y_train_predict = predict(w, b, X_train)
    Y_test_predict=predict(w,b,X_test)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_train_predict - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_test_predict - Y_test)) * 100))


    d={
        "costs": costs,
        "Y_prediction_test": Y_test_predict,
        "Y_prediction_train": Y_train_predict,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations
    }
    return d
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

index = 1
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[int(d["Y_prediction_test"][0,index])].decode("utf-8") +  "\" picture.")

costs=np.squeeze(d["costs"])
plt.plot(costs)
plt.ylabel("cost")
plt.xlabel("iterations (per hundreds)")
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


