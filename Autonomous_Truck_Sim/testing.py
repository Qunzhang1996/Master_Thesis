import numpy as np

XY = np.array([1.0,1.5,4.0,5.0])
X0  = 4.5

idx = np.where(XY-X0 >= 0)[0]
idx_upper = np.argmin(XY[idx]-X0)

print(XY[idx[idx_upper]])