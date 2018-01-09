import numpy as np
import pandas as pd

# da1=np.arange(12).reshape(2,3,2)
# print(da1)
# da2=np.arange(18).reshape(6,3)
# print(da2)
#
# b=np.arange(6).reshape(2,3)
# print(b)
#
# print(np.dot(da1,da2)-b)  #3+6+9+12+15



da1=np.arange(24).reshape(4,2,3)

# da2=np.array([[[1,2,3],[4,5,6]],
#               [[7,8,9],[10,11,12]],
#               [[11,21,31],[41,51,61]],
#               [[71,81,91],[101,111,121]]])
# print(da2.shape)
# print(da2.shape[0])

# da2=np.arange(6).reshape(2,3)
# print(da2)
# print(da2.shape)
# print(da2[0:2,0])
# da3=da1.reshape(da1.shape[0],-1)
# print(da3)

b=0
assert(isinstance(b,float))