import numpy as np
import pandas as pd

da1=np.arange(12).reshape(2,6)
print(da1)
da2=np.arange(18).reshape(6,3)
print(da2)

b=np.arange(6).reshape(2,3)
print(b)

print(np.dot(da1,da2)-b)  #3+6+9+12+15