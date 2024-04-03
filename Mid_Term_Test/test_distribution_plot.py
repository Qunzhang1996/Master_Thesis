# Re-import necessary libraries and re-define variables after reset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from MPC_tighten_bound import MPC_tighten_bound
import sys
path_to_add='C:\\Users\\A490243\\Desktop\\Master_Thesis'
sys.path.append(path_to_add)
from util.utils import *
import numpy as np
from scipy.linalg import solve_discrete_are


ellipse_dimensions = []


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
time_steps = np.arange(0, 13, 3)  # N=1, 4, 8, 12

# Plotting setup
# Plotting setup for combined visualization
fig, ax = plt.subplots(figsize=(12, 8))

# Maximum extents for plot adjustment
max_width = 0
max_height = 143.318146
# Colormap and normalizer setup
colormap = plt.cm.viridis  # Choose a colormap
normalize = Normalize(vmin=min(time_steps), vmax=max(time_steps))
scalar_map = ScalarMappable(norm=normalize, cmap=colormap)
P0,process_noise,possibility = set_stochastic_mpc_params()
process_noise[1,1] = sigma_y_squared
MPC_tighten_bound = MPC_tighten_bound(A, B, g, vehicle.Q, vehicle.R, P0, process_noise, possibility)



# Plot each ellipse with adjusted settings for clear visualization
for N in time_steps:
    # Time at current N
    current_time = N * dt

    # Mean position
    mean_N = [x_positions[N], y_positions[N]]
    print(mean_N)   

    # Get color from colormap
    color = scalar_map.to_rgba(N-5)

    ax.plot(mean_N[0], mean_N[1], 'o', markersize=5, label=f'N={N}', color=color)
    _, P_next_N_list = MPC_tighten_bound.calculate_P_next_N(K, N)
    
    if N == 0:
        scaled_sigma_x_squared_N = sigma_x_squared
        scaled_sigma_y_squared_N = sigma_y_squared
        #! add the vehicle as rect at the center of the ellipse, width=2.89m, length=8.46m
        ax.add_patch(plt.Rectangle((mean_N[0]-8.46/2, mean_N[1]-2.59/2), 8.46, 2.59, color='r', alpha=0.3))
        
    else:
        scaled_sigma_x_squared_N = P_next_N_list[-1][0, 0]
        scaled_sigma_y_squared_N = P_next_N_list[-1][1, 1]

    print(scaled_sigma_x_squared_N, scaled_sigma_y_squared_N)
    # Ellipse parameters for 95% confidence
    ellipse_width_N = 2 * np.sqrt(3 * scaled_sigma_x_squared_N)
    ellipse_height_N = 2 * np.sqrt(3 * scaled_sigma_y_squared_N)
    ellipse_dimensions.append((ellipse_width_N, ellipse_height_N))
    # Add ellipse
    confidence_ellipse_N = Ellipse(xy=mean_N, width=ellipse_width_N, height=ellipse_height_N, 
                                   angle=0, edgecolor='r', facecolor='yellow', alpha=0.3)
    ax.add_patch(confidence_ellipse_N)
    
    
    
    # Update max extents
    max_width = max(max_width, x_positions[N] + ellipse_width_N)
    max_height = max(max_height, ellipse_height_N)
    # Label each ellipse with its N value

np.save(r'C:\Users\A490243\Desktop\Master_Thesis\Parameters\ellipse_dimensions.npy', ellipse_dimensions)

# Decorations
plt.rcParams.update({'font.size': 12, 'font.family': 'Times New Roman'})
plt.title('Propagation of trajectory for different time steps')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend(loc='upper right')

# Adjusting plot limits based on the maximum extents of the ellipses
plt.xlim(0,75)
plt.ylim(130, 160)

# plot the center line 
plt.plot([0,300], [143.318146, 143.318146], color='gray', linestyle='--', lw=1)
plt.plot([0,300], [143.318146-3.5, 143.318146-3.5], color='gray', linestyle='--', lw=1)
plt.plot([0,300], [143.318146+3.5, 143.318146+3.5], color='gray', linestyle='--', lw=1)
# plot the la75 line
plt.plot([0, 75], [143.318146-1.75, 143.318146-1.75], 'k', lw=1)
plt.plot([0, 75], [143.318146+1.75, 143.318146+1.75], 'k', lw=1)
plt.plot([0, 75], [143.318146-1.75-3.5, 143.318146-1.75-3.5], 'k', lw=1)
plt.plot([0, 75], [143.318146+1.75+3.5, 143.318146+1.75+3.5], 'k', lw=1)
plt.tight_layout()
plt.savefig('C:\\Users\\A490243\\Desktop\\Master_Thesis\\Figure\\propagation_of_trajectory.png')
plt.show()
