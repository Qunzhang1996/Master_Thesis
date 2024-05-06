# Re-import necessary libraries and re-define variables after reset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from MPC_tighten_bound import MPC_tighten_bound
import sys
# path_to_add='C:\\Users\\A490243\\Desktop\\Master_Thesis'
# C:\Users\86232\Desktop\masterthesis\Master_Thesis\
path_to_add='C:\\Users\\86232\\Desktop\\masterthesis\\Master_Thesis'
sys.path.append(path_to_add)
from util.utils import *
import numpy as np
from scipy.linalg import solve_discrete_are


ellipse_dimensions = []
plt.rc('font', family='Times New Roman')

class vehicle:
    def __init__(self, WB=6, dt=0.3):
        self.WB = WB
        self.dt = dt
        self.Q = np.diag([0, 40, 3e2, 5])
        self.R = np.diag([5, 5])

    def vehicle_linear_discrete_model(self, v=15, phi=0, delta=0):
            """
            Calculate linear and discrete time dynamic model.
            """
            A_d = np.array([[1.0, 0.0, self.dt * np.cos(phi), -self.dt * v * np.sin(phi)],
                            [0.0, 1.0, self.dt * np.sin(phi), self.dt * v * np.cos(phi)],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, self.dt * np.tan(delta) / self.WB, 1.0]])
            
            B_d = np.array([[0.0, 0.0],
                            [0.0, 0.0],
                            [0.0, self.dt],
                            [self.dt * v / (self.WB * np.cos(delta) ** 2), 0.0]])
            
            
            g_d = np.array([self.dt * v * np.sin(phi) * phi,
                            -self.dt * v * np.cos(phi) * phi,
                            0.0,
                            -self.dt * v * delta / (self.WB * np.cos(delta) ** 2)])
            self.A = A_d
            self.B = B_d
            self.g = g_d
            return A_d, B_d, g_d
             
    def cal_LQR(self, A, B, Q, R):
        """
        Calculate the LQR controller gain.
        """
        P = solve_discrete_are(A, B, Q, R)
        K = -np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
        return K


N = 12
WB=6 
dt=0.3
vehicle = vehicle(WB, dt)
A, B, g = vehicle.vehicle_linear_discrete_model()
K = vehicle.cal_LQR(A, B, vehicle.Q, vehicle.R)
x0 = np.array([[5], [143.318146], [15], [0]])
xd = np.array([[5], [143.318146], [15], [0]])
states = np.zeros((4, N + 1))
states[:, 0] = x0.flatten()
for k in range(N):
    u = (K @ (states[:, k:k+1] - xd)).flatten()
    states[:, k+1] = A @ states[:, k] + B @ u  + g.flatten()

# Extract x and y for plotting
x_positions = states[0, :]
y_positions = states[1, :]
sigma_x_squared = 0.3  # Process noise variance for x
sigma_y_squared = 0.05  # Process noise variance for y

# Time steps to plot
time_steps = np.arange(0, 13, 1) 
# Maximum extents for plot adjustment
max_width = 0
max_height = 143.318146
P0,process_noise,possibility = set_stochastic_mpc_params()
process_noise[1,1] = sigma_y_squared
MPC_tighten_bound = MPC_tighten_bound(A, B, g, vehicle.Q, vehicle.R, P0, process_noise, possibility)
uncertainty_ellipse_x = []
uncertainty_ellipse_y = []
# Plot each ellipse with adjusted settings for clear visualization
for idx, N in enumerate(time_steps):
    # Time at current N
    current_time = N * dt

    # Mean position
    mean_N = [x_positions[N], y_positions[N]]
    # print(mean_N)   


    _, P_next_N_list = MPC_tighten_bound.calculate_P_next_N(K, N)
    
    if N == 0:
        scaled_sigma_x_squared_N = sigma_x_squared
        scaled_sigma_y_squared_N = sigma_y_squared
        uncertainty_ellipse_x.append(scaled_sigma_x_squared_N)
        uncertainty_ellipse_y.append(scaled_sigma_y_squared_N)
        

    else:
        scaled_sigma_x_squared_N = P_next_N_list[-1][0, 0]
        scaled_sigma_y_squared_N = P_next_N_list[-1][1, 1]
        uncertainty_ellipse_x.append(scaled_sigma_x_squared_N)
        uncertainty_ellipse_y.append(scaled_sigma_y_squared_N)
        
plt.rcParams['xtick.labelsize'] = 18  
plt.rcParams['ytick.labelsize'] = 18  
# plot the x and y separately into two subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


axs[0].plot(time_steps, uncertainty_ellipse_x, label='$\Sigma_x$', color='red', linewidth=2)
axs[0].scatter(time_steps, uncertainty_ellipse_x, color='red')
axs[0].set_xlabel('Time Steps', fontsize=18)
axs[0].set_ylabel('Uncertainty in x [m]', fontsize=18)
axs[0].set_title('Uncertainty in x over Time', fontsize=18)
axs[0].legend(fontsize=18, loc='upper right')


axs[1].plot(time_steps, uncertainty_ellipse_y, label='$\Sigma_y$', color='blue', linewidth=2)
axs[1].scatter(time_steps, uncertainty_ellipse_y, color='blue')
axs[1].set_xlabel('Time Steps', fontsize=18)
axs[1].set_ylabel('Uncertainty in y [m]', fontsize=18)
axs[1].set_title('Uncertainty in y over Time', fontsize=18)
axs[1].legend(fontsize=18, loc='upper right')

plt.rc('font', family='Times New Roman')
plt.tight_layout()
plt.savefig('C:/Users/86232/Desktop/masterthesis/Master_Thesis/Figure/Figure_for_Thesis/TightenedConstraints/uncertainty_total.png', dpi=300)
plt.show()




    



