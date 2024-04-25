import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Provided sequences
temptX = np.array([0, 0.904314, 1.32338, 1.64943, 1.92228, 2.16112, 2.3761, 2.57318, 2.7562, 2.9278, 3.08989, 3.24389, 3.3909])
temptY = np.array([0, 0.382754, 0.52193, 0.53374, 0.533851, 0.533894, 0.533909, 0.53391, 0.53391, 0.53391, 0.53391, 0.53391, 0.53391])

# Adjust the data for modeling x_t+1 as a function of x_t
X_temptX_adj = temptX[:-1].reshape(-1, 1)  # Input: all but the last value
y_temptX_adj = temptX[1:]  # Output: all but the first value

X_temptY_adj = temptY[:-1].reshape(-1, 1)  # Same for temptY
y_temptY_adj = temptY[1:]

# Perform linear regression to find coefficients a and b for both sequences
lr_temptX_adj = LinearRegression()
lr_temptX_adj.fit(X_temptX_adj, y_temptX_adj)
a_temptX_adj, b_temptX_adj = lr_temptX_adj.coef_[0], lr_temptX_adj.intercept_


print("####################################################")
print(f"temptX: y = {a_temptX_adj:.4f}x + {b_temptX_adj:.4f}")

lr_temptY_adj = LinearRegression()
lr_temptY_adj.fit(X_temptY_adj, y_temptY_adj)
a_temptY_adj, b_temptY_adj = lr_temptY_adj.coef_[0], lr_temptY_adj.intercept_
print("####################################################")
print(f"temptY: y = {a_temptY_adj:.4f}x + {b_temptY_adj:.4f}")

# Predict the next values using the linear models
predicted_temptX_adj = lr_temptX_adj.predict(X_temptX_adj)
predicted_temptY_adj = lr_temptY_adj.predict(X_temptY_adj)

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# temptX and its regression-predicted values
ax[0].scatter(np.arange(len(temptX)-1), temptX[1:], color='blue', label='Actual temptX')
ax[0].plot(np.arange(len(temptX)-1), predicted_temptX_adj, color='red', label='Predicted temptX')
ax[0].set_title('temptX and Predicted Values')
ax[0].set_xlabel('Index')
ax[0].set_ylabel('Value')
ax[0].legend(loc='lower right')

# temptY and its regression-predicted values
ax[1].scatter(np.arange(len(temptY)-1), temptY[1:], color='green', label='Actual temptY')
ax[1].plot(np.arange(len(temptY)-1), predicted_temptY_adj, color='orange', label='Predicted temptY')
ax[1].set_title('temptY and Predicted Values')
ax[1].set_xlabel('Index')
ax[1].set_ylabel('Value')
ax[1].legend(loc='lower right')
plt.tight_layout()
plt.show()
