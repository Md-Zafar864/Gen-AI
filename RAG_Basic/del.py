import numpy as np

arr=[3,6,7,1,4]

idx=np.argsort(arr)
idx2=idx[::-1][:3]
print(idx2)