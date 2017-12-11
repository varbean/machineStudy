import sklearn.datasets as ds

data,y = ds.make_blobs(1500, n_features=3, centers=4, random_state=28)
print(data)
print(y)