import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases_v2 import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward
from dnn_app_utils_v2 import *

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


np.random.seed(1)

def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(1)
    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros((n_y,1))

    assert(W1.shape==(n_h,n_x))
    assert (b1.shape==(n_h,1))
    assert(W2.shape==(n_y,n_h))
    assert(b2.shape==(n_y,1))

    param = {
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }
    return param

# parameters = initialize_parameters(2,2,1)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)

    parameters={}
    l=len(layer_dims)
    for i in range(1,l):
        parameters["W"+str(i)]=np.random.randn(layer_dims[i],layer_dims[i-1])*0.01
        parameters["b"+str(i)]=np.zeros((layer_dims[i],1))

    return parameters

# parameters = initialize_parameters_deep([5,4,3])
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

#计算Z,并储存计算需要的参数
def linear_forward(A,W,b):

    Z=np.dot(W,A)+b

    assert (Z.shape==(W.shape[0],A.shape[1]))
    cache=(A,W,b)

    return Z,cache

# A, W, b = linear_forward_test_case()
#
# Z, linear_cache = linear_forward(A, W, b)
# print("Z = " + str(Z))
#计算A,激活函数的值,并储存计算需要的参数
def linear_activation_forward(A_prev,W,b,activation):
    if activation =="sigmoid":
        Z,linear_cache=linear_forward(A_prev,W,b)
        A,activation_cache=sigmoid(Z)
    if activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape ==(W.shape[0],A_prev.shape[1]))

    cache=(linear_cache,activation_cache)
    return A,cache

# A_prev, W, b = linear_activation_forward_test_case()
#
# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
# print("With sigmoid: A = " + str(A))
#
# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
# print("With ReLU: A = " + str(A))
#正向传播
def L_model_forward(X,parameters):
    caches=[]
    A=X
    L=len(parameters) // 2  # // 为向下取整

    for l in range(1,L):
        A_prev=A
        #计算激活函数
        A,cache=linear_activation_forward(A_prev,parameters["W"+str(l)],parameters["b"+str(l)],"relu")
        caches.append(cache)
    #计算最后一层激活函数获取结果
    Al,cache=linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],"sigmoid")
    caches.append(cache)

    assert (Al.shape==(1,X.shape[1]))
    return Al,caches

# X, parameters = L_model_forward_test_case()
# AL, caches = L_model_forward(X, parameters)
# print("AL = " + str(AL))
# print("Length of caches list = " + str(len(caches)))
#计算成本
def compute_cost(AL,Y):
    m=Y.shape[1]

    loga=np.multiply(Y,np.log(AL))+(np.multiply((1-Y),np.log(1-AL)))
    cost=-np.sum(loga)/m

    cost=np.squeeze(cost)
    assert (cost.shape==())

    return cost

# Y, AL = compute_cost_test_case()sz
#
# print("cost = " + str(compute_cost(AL, Y)))
#反向传播
def  linear_backward(dZ,cache):
    A_prev,W,b=cache
    m=cache[0].shape[1]

    dW=np.dot(dZ,A_prev.T)/m
    db=np.sum(dZ,axis=1,keepdims=True)/m
    dA=np.dot(W.T,dZ)

    assert (dA.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA,dW,db

# dZ, linear_cache = linear_backward_test_case()
#
# dA_prev, dW, db = linear_backward(dZ, linear_cache)
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db))
#反向传播激活函数
def linear_activation_backward(dA,cache,activation):
    l_cache,a_cache=cache

    if activation=="sigmoid":
        dZ=sigmoid_backward(dA,a_cache)
        dA_prev,dW,db=linear_backward(dZ,l_cache)
    if activation == "relu":
        dZ = relu_backward(dA, a_cache)
        dA_prev, dW, db = linear_backward(dZ, l_cache)

    return dA_prev,dW,db

# AL, linear_activation_cache = linear_activation_backward_test_case()
#
# dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "sigmoid")
# print ("sigmoid:")
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db) + "\n")
#
# dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "relu")
# print ("relu:")
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db))
#整合反向传播
def L_model_backward(AL,Y,caches):

    grads={}
    l=len(caches)
    m=AL.shape[1]
    Y=Y.reshape(AL.shape)


    dAL=- (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache=caches[l-1]
    grads["dA" + str(l)],grads["dW"+str(l)],grads["db"+str(l)]=linear_activation_backward(dAL,current_cache,"sigmoid")

    for i in reversed(range(l-1)):
        current_cache=caches[i]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(i+2)],current_cache,"relu")
        grads["dA" + str(i + 1)] = dA_prev_temp
        grads["dW" + str(i + 1)] = dW_temp
        grads["db" + str(i + 1)] = db_temp

    return grads

# AL, Y_assess, caches = L_model_backward_test_case()
# grads = L_model_backward(AL, Y_assess, caches)
# print ("dW1 = "+ str(grads["dW1"]))
# print ("db1 = "+ str(grads["db1"]))
# print ("dA1 = "+ str(grads["dA1"]))
#根据梯度更新参数
def update_parameters(parameters,grads,learning_rate):
    l=len(parameters) //2

    for i in range(l):
        parameters["W"+str(i+1)]-=grads["dW"+str(i+1)]*learning_rate
        parameters["b" + str(i + 1)] -= grads["db" + str(i + 1)] * learning_rate

    return parameters


# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads, 0.1)
#
# print ("W1 = "+ str(parameters["W1"]))
# print ("b1 = "+ str(parameters["b1"]))
# print ("W2 = "+ str(parameters["W2"]))
# print ("b2 = "+ str(parameters["b2"]))


def two_layer_model(X,Y,layers_dims,learning_rate=0.0075,num_iterators=3000,print_cost=False):
    costs=[]
    grads={}
    np.random.randn(1)

    m=X.shape[1]

    parameters=initialize_parameters(layers_dims[0],layers_dims[1],layers_dims[2])

    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2 = parameters["b2"]

    for i in range(1,num_iterators):
        #计算向前传播的参数
        A1,cache1=linear_activation_forward(X,W1,b1,"relu")
        A2,cache2=linear_activation_forward(A1,W2,b2,"sigmoid")

        #计算成本
        cost=compute_cost(A2,Y)

        #计算梯度值
        dA2= - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))


        dA1,dW2,db2=linear_activation_backward(dA2,cache2,"sigmoid")
        dA0,dW1,db1=linear_activation_backward(dA1,cache1,"relu")

        grads["dW1"]=dW1
        grads["dW2"] = dW2
        grads["db1"] = db1
        grads["db2"] = db2

        #更新函数
        parameters=update_parameters(parameters,grads,learning_rate)
        #加入成本函数


        if print_cost and i %100 ==0 :
            costs.append(cost)
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters

def many_layers_model(X,Y,layer_dims,learning_rate=0.0075,num_iterators=3000,print_cost=True):

    np.random.seed(1)

    costs=[]

    param=initialize_parameters_deep(layer_dims)#参数

    for i in range(0,num_iterators):
        #向前传播 存储必要的参数
        Al,caches=L_model_forward(X,param)

        #计算成本
        cost=compute_cost(Al,Y)

        #更新梯度
        grads=L_model_backward(Al,Y,caches)

        #更新参数
        param=update_parameters(param,grads,learning_rate)

        if print_cost and i%100 ==0 :
            costs.append(cost)
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return param


# train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
#
# train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
# test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
#
# # Standardize data to have feature values between 0 and 1.
# train_x = train_x_flatten/255.
# test_x = test_x_flatten/255.
# layers_dims=[12288,20,7,5,1]
# layer_dim=[12288,7,1]
#
# param=two_layer_model(train_x,train_y,layer_dim, learning_rate=0.0075,num_iterators = 2500, print_cost=True)
