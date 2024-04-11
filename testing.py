import numpy as np

XY = np.array([1.0,1.5,4.0,5.0])
X0  = 4.5

idx = np.where(XY-X0 >= 0)[0]
idx_upper = np.argmin(XY[idx]-X0)

import matplotlib.pyplot as plt
plt.plot(X0)
#! save the plot
plt.savefig('plot.png')
plt.show()