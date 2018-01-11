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

lrc=sklearn.linear_model.LogisticRegressionCV()
lrc.fit(X.T,Y.T)

plot_decision_boundary(lambda x:lrc.predict(x),X,Y)
plt.title("Logistic Regression")

LR_predictions = lrc.predict(X.T)
# print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
#        '% ' + "(percentage of correctly labelled datapoints)")
#
# plt.show()
