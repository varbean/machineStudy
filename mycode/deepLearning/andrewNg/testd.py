import numpy as np
def sigmoid(x):
    y=1/(1+np.exp(-x) )
    return y

def initialize_with_zeros(dim):
    w=np.zeros((dim,1))
    b=0

    assert(w.shape==(dim,1))
    return w,b

print(initialize_with_zeros(4))
