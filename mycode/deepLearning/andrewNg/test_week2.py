import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1)

X,Y=load_planar_dataset()

# plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
# plt.show()

shape_X=X.shape
shape_Y=Y.shape
m=shape_X[1]

# print ('The shape of X is: ' + str(shape_X))
# print ('The shape of Y is: ' + str(shape_Y))
# print ('I have m = %d training examples!' % (m))

# lrc=sklearn.linear_model.LogisticRegressionCV()
# lrc.fit(X.T,Y.T)
#
# plot_decision_boundary(lambda x:lrc.predict(x),X,Y)
# plt.title("Logistic Regression")
#
# LR_predictions = lrc.predict(X.T)
# print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
#        '% ' + "(percentage of correctly labelled datapoints)")
#
# plt.show()

def layer_sizes(X,Y):
    n_x=X.shape[0]
    n_h=4
    n_y=Y.shape[0]
    # return (n_x,n_h,n_y)
    return (n_x,n_y)
# X_testcase,Y_testcase=layer_sizes_test_case()
# n_x,n_h,n_y=layer_sizes(X_testcase,Y_testcase)
#
# print("The size of the input layer is: n_x = " + str(n_x))
# print("The size of the hidden layer is: n_h = " + str(n_h))
# print("The size of the output layer is: n_y = " + str(n_y))

#初始化参数
def initialize_parameters(n_x,n_h,n_y):

    np.random.seed(2)
    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros((n_y,1))

    params={"W1":W1,"b1":b1,"W2":W2,"b2":b2}
    return params

#向前传播
def forward_propagation(X,parameters):
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]

    #print("W1 shape:"+str(W1.shape))
    Z1=np.dot(W1,X)

    #print("Z1 shape:"+str(Z1.shape))
    A1=np.tanh(Z1)

    # print("W2 shape:"+str(W2.shape))
    # print("A1 shape:" + str(A1.shape))
    Z2=np.dot(W2,A1)
    A2=sigmoid(Z2)

    assert (A2.shape == (1,X.shape[1]))

    cache={
        "Z1":Z1,
        "A1":A1,
        "Z2":Z2,
        "A2":A2
    }
    return A2,cache

# X_assess, parameters = forward_propagation_test_case()
# A2,cache=forward_propagation(X_assess,parameters)
# print(np.mean(cache['z1']) ,np.mean(cache['a1']),np.mean(cache['z2']),np.mean(cache['a2']))
#计算成本函数
def compute_cost(A2,Y,parameters):
    m=Y.shape[1]

    ll=np.multiply(np.log(A2),Y)+np.multiply(np.log(1-A2),1-Y)
    cost=-np.sum(ll) /m

    cost=np.squeeze(cost)
    assert (isinstance(cost,float))
    return cost

# A2, Y_assess, parameters = compute_cost_test_case()
# print(compute_cost(A2,Y_assess,parameters))
#向后传播
def backward_propagation(parameters,cache,X,Y):
    m=Y.shape[1]

    W1=parameters["W1"]
    W2=parameters["W2"]

    A1=cache["A1"]
    A2=cache["A2"]

    dZ2=A2-Y
    dW2=np.dot(dZ2,A1.T)/m
    db2=np.sum(dZ2,axis=1,keepdims=True)/m

    dZ1=np.dot(W2.T,dZ2) * (1-np.power(A1,2))
    dW1=np.dot(dZ1,X.T)/m
    db1=np.sum(dZ1,axis=1,keepdims=True)/m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads

# parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
#
# grads = backward_propagation(parameters, cache, X_assess, Y_assess)
# print ("dW1 = "+ str(grads["dW1"]))
# print ("db1 = "+ str(grads["db1"]))
# print ("dW2 = "+ str(grads["dW2"]))
# print ("db2 = "+ str(grads["db2"]))

#更新参数
def update_parameters(parameters,grads,learning_rate=1.2):
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2 = parameters["W2"]
    b2=parameters["b2"]

    dW1=grads["dW1"]
    dW2=grads["dW2"]
    db1 = grads["db1"]
    db2 = grads["db2"]

    W1-=learning_rate*dW1
    b1 -= learning_rate * db1
    W2-=learning_rate*dW2
    b2 -= learning_rate * db2

    parameters={
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }

    return parameters

# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads)
#
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

#创建模型（对上述方法进行整合）
def nn_model(X,Y,n_h,num_count=10000,print_cost=False):
    np.random.seed(3)
    n_x,n_y=layer_sizes(X,Y)

    params=initialize_parameters(n_x,n_h,n_y)

    for i in range(0,num_count):
        A2,cache=forward_propagation(X,params)
        cost=compute_cost(A2,Y,params)
        grads=backward_propagation(params,cache,X,Y)

        params=update_parameters(params,grads)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    return params

# X_assess, Y_assess = nn_model_test_case()
# parameters = nn_model(X_assess, Y_assess, 4, num_count=10000, print_cost=False)
# print(parameters)

def predict(params,X):
    A2,cache=forward_propagation(X,params)
    Y_pre=(A2>0.5)
    return Y_pre

# parameters, X_assess = predict_test_case()
#
# predictions = predict(parameters, X_assess)
# print("predictions mean = " + str(np.mean(predictions)))

# parameters = nn_model(X, Y, n_h = 4, num_count = 10000, print_cost=True)

# #Plot the decision boundary
# plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
# plt.title("Decision Boundary for hidden layer size " + str(4))
# plt.show()
# predictions = predict(parameters, X)
# print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')


# Datasets
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}

### START CODE HERE ### (choose your dataset)
dataset = "gaussian_quantiles"
### END CODE HERE ###

X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])

# make blobs binary
if dataset == "blobs":
    Y = Y%2

# Visualize the data
# plt.figure(figsize=(16, 32))
# hidden_layer_sizes = [1, 2, 3, 4, 5, 10, 20]
# for i, n_h in enumerate(hidden_layer_sizes):
n_h=4
plt.title('Hidden Layer of size %d' % n_h)
parameters = nn_model(X, Y, n_h, num_count = 5000)
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
predictions = predict(parameters, X)
accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
plt.show()