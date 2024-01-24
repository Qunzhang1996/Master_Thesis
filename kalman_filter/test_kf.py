import numpy as np
import matplotlib.pyplot as plt
from kalman_filter import kalman_filter


# Define the system matrices for a simple 1D case
A = np.array([[1]])  # State transition matrix
B = np.array([[1]])  # Input transition matrix
C = np.array([[1]])  # Observation matrix
x0 = np.array([[0]])   # Initial state
P0 = np.array([[1]]) # Initial state covariance
Q0 = np.array([[0.01]]) # Process noise covariance
R0 = np.array([[0.1]])   # Measurement noise covariance
u = np.array([[2]])
# Create the Kalman Filter
kf = kalman_filter(A, B, C, x0, P0, Q0, R0)

# Generate synthetic data
num_steps = 50
true_state = np.linspace(0, 20, num_steps)
measurements = true_state + np.random.normal(0, np.sqrt(R0[0,0]), num_steps)

# Apply Kalman Filter
estimates = np.zeros(num_steps)
for i in range(num_steps):
    kf.predict(u)
    kf.update(np.array([measurements[i]]))
    estimates[i] = kf.get_estimate

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(true_state, label='True State')
plt.plot(measurements, label='Measurements', linestyle='None', marker='o', color='r')
plt.plot(estimates, label='Kalman Filter Estimate', color='g')
plt.xlabel('Time Step')
plt.ylabel('State')
plt.title('Kalman Filter in Action')
plt.legend()
plt.grid(True)
plt.show()
