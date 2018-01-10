import numpy as np
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
def model(X_train,Y_train,X_test,Y_test,num_iterations=2000,learning_rate=0.5,print_cost=False):
    w,b=initialize_with_zeros(X_train.shape[0]) #初始化


    params,grads,costs=optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost=True)
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




