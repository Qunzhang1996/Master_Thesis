import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import os

# Set default font to Times New Roman for IEEE standards
plt.rc('font', family='Times New Roman')

def plot_trajectory(truck_positions, truck_velocity, figure_dir, figure_name):
    # Adjust figure size for IEEE single-column width
    fig, ax = plt.subplots(figsize=(12, 4))  # 7 inches wide, 2.5 inches tall

    # Prepare segments for the LineCollection
    segments = []
    for i in range(1, len(truck_positions)):
        x0, y0 = truck_positions[i-1][:2]
        x1, y1 = truck_positions[i][:2]
        segment = [(x0, y0), (x1, y1)]
        segments.append(segment)

    # Create a LineCollection from the segments
    lc = LineCollection(segments, array=truck_velocity[:-1], cmap='viridis', norm=Normalize(vmin=min(truck_velocity), vmax=max(truck_velocity)), linewidth=5)  # Adjust linewidth for visibility
    ax.add_collection(lc)

    # Add color bar with adjusted label size
    cbar = plt.colorbar(lc, ax=ax, label='Velocity (m/s)' )
    # label size = 18
    cbar.ax.set_ylabel('Velocity (m/s)', fontsize=12)
    cbar.ax.tick_params(labelsize=12)  # Smaller tick labels

    # Set title and labels with appropriate font size
    plt.title('Truck Trajectory with Velocity Coloring', fontsize=12)
    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Y Position (m)', fontsize=12)

    # Adjusting tick label size
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Draw black and dashed lines
    ax.plot([0, 700], [143.318146 - 1.75, 143.318146 - 1.75], 'k')
    ax.plot([0, 700], [143.318146 + 1.75, 143.318146 + 1.75], 'k')
    ax.plot([0, 700], [143.318146 - 1.75 - 3.5, 143.318146 - 1.75 - 3.5], 'k')
    ax.plot([0, 700], [143.318146 + 1.75 + 3.5, 143.318146 + 1.75 + 3.5], 'k')
    ax.plot([0, 700], [143.318146, 143.318146], 'k--')
    ax.plot([0, 700], [143.318146 - 3.5, 143.318146 - 3.5], 'k--')
    ax.plot([0, 700], [143.318146 + 3.5, 143.318146 + 3.5], 'k--')

    # Set y limit for better focus on the trajectory
    ax.set_ylim(130, 160)

    # Save the figure with higher resolution suitable for publication
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, figure_name), format='png', dpi=300)  # High resolution
    plt.show()
    
def compute_acceleration(positions, sampling_time):
    # Compute velocities
    velocities = np.diff(positions, axis=0) / sampling_time
    # Compute acceleration by diffing the velocities and dividing by sampling time
    accelerations = np.diff(velocities, axis=0) / sampling_time
    return velocities[:-1], accelerations
    
    
def plot_acceleration(ax_data, ay_data, velocities, figure_dir, figure_name, sampling_time=0.3):
    g = 9.81  # Gravitational acceleration in m/s^2
    ax_data /= g  # Convert to g
    ay_data /= g  # Convert to g
    # Set default font to Times New Roman for IEEE standards
    plt.rc('font', family='Times New Roman')
    fig, ax = plt.subplots(figsize=(9, 7))  # Adjusted for a typical IEEE single-column width

    # Scatter plot ax vs ay colored by velocity
    sc = ax.scatter(ax_data, ay_data, array=truck_velocity[:-1], cmap='viridis', edgecolor='none', s=100)
    cbar = plt.colorbar(sc, ax=ax, label='Velocity (m/s)')
    cbar.ax.tick_params(labelsize=12)  # Adjust color bar label size

    # Set axis labels and title
    plt.title('Acceleration Profile', fontsize=12)
    plt.xlabel('ax (g)', fontsize=12)
    plt.ylabel('ay (g)', fontsize=12)
    plt.ylim(-1, 1)  # Limit the y-axis to -1 to 1 g
    plt.xlim(-1, 1)  # Limit the x-axis to -1 to 1 g
    plt.tight_layout()
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Save the figure
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    plt.savefig(os.path.join(figure_dir, figure_name), format='png', dpi=300)  # High resolution
    plt.show()

# Load data
X_traffic = np.load(r'C:\Users\86232\Desktop\masterthesis\Master_Thesis\Parameters\X_traffic.npy')
px_ego = X_traffic[0,:,1]
py_ego = X_traffic[1,:,1]
v_ego = X_traffic[2,:,1]
psi_ego = X_traffic[3,:,1]

truck_positions = [(px_ego[i], py_ego[i], psi_ego[i]) for i in range(len(px_ego))]
truck_velocity = v_ego

# Directory and file name for saving the figure
figure_dir = "C:/Users/86232/Desktop/masterthesis/Master_Thesis/Figure"
figure_name = "velocity_tracking_layout.png"

plot_trajectory(truck_positions, truck_velocity, figure_dir, figure_name)
# using sampling time =0.3s to calculate the acceleration ax and ay
sampling_time = 0.3
velocities, accelerations = compute_acceleration(np.array(truck_positions), sampling_time)
ax_data = accelerations[:, 0]
ay_data = accelerations[:, 1]
plot_acceleration(ax_data, ay_data, velocities, figure_dir, "acceleration_profile.png", sampling_time)

X_traffic_nosmpc = np.load(r'C:\Users\86232\Desktop\masterthesis\Master_Thesis\Parameters\X_traffic_no_stochastic.npy')
px_ego_nosmpc = X_traffic_nosmpc[0,:,1]
py_ego_nosmpc = X_traffic_nosmpc[1,:,1]
v_ego_nosmpc = X_traffic_nosmpc[2,:,1]  
psi_ego_nosmpc = X_traffic_nosmpc[3,:,1]

truck_positions_nosmpc = [(px_ego_nosmpc[i], py_ego_nosmpc[i], psi_ego_nosmpc[i]) for i in range(len(px_ego_nosmpc))]
truck_velocity_nosmpc = v_ego_nosmpc

figure_dir = "C:/Users/86232/Desktop/masterthesis/Master_Thesis/Figure"
figure_name = "velocity_tracking_layout_nosmpc.png"
plot_trajectory(truck_positions_nosmpc, truck_velocity_nosmpc, figure_dir, figure_name)

sampling_time = 0.3
velocities, accelerations = compute_acceleration(np.array(truck_positions_nosmpc), sampling_time)
ax_data = accelerations[:, 0]
ay_data = accelerations[:, 1]
plot_acceleration(ax_data, ay_data, velocities, figure_dir, "acceleration_profile_nosmpc.png", sampling_time)


X_traffic_EKF = np.load(r'C:\Users\86232\Desktop\masterthesis\Master_Thesis\Parameters\X_traffic_EKF.npy')
py_ego_EKF = X_traffic_EKF[0,:,1]
px_ego_EKF = X_traffic_EKF[1,:,1]
v_ego_EKF = X_traffic_EKF[2,:,1]
psi_ego_EKF = X_traffic_EKF[3,:,1]

truck_positions_EKF = [(px_ego_EKF[i], py_ego_EKF[i], psi_ego_EKF[i]) for i in range(len(px_ego_EKF))]
truck_velocity_EKF = v_ego_EKF

figure_dir = "C:/Users/86232/Desktop/masterthesis/Master_Thesis/Figure"
figure_name = "velocity_tracking_layout_EKF.png"
plot_trajectory(truck_positions_EKF, truck_velocity_EKF, figure_dir, figure_name)

sampling_time = 0.3
velocities, accelerations = compute_acceleration(np.array(truck_positions_EKF), sampling_time)
ax_data = accelerations[:, 0]
ay_data = accelerations[:, 1]
plot_acceleration(ax_data, ay_data, velocities, figure_dir, "acceleration_profile_EKF.png", sampling_time)