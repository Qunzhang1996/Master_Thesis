from matplotlib import pyplot as plt
import numpy as np
from casadi import *
# Define the smooth maximum function
def smooth_max(x, beta=0.00001):
    # exp_linear = np.exp(-beta * x)
    # exp_quadratic = np.exp(beta * x**2)
    # return (exp_linear * -x + exp_quadratic * x**2) / (exp_linear + exp_quadratic)
    sigmoid = 1/(1+exp(-10*(x+2)))
    cost = sigmoid*(-x)+(1-sigmoid)*x**2
    return cost

# Generate x values
x_values = np.linspace(-8, 2, 400)

# Calculate y values using the smooth max function with a chosen beta value
y_values_smooth_max = smooth_max(x_values, beta=10)

# Plot the smooth approximation alongside the original max function
plt.figure(figsize=(10, 8))
# plt.plot(x_values, max(x_values), label='f(x) = max(x, x^2)', linestyle='--')
plt.plot(x_values, y_values_smooth_max, label='Smooth Approximation of max(x, x^2)', color='red')
plt.title('Smooth Approximation vs. Original Max Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
