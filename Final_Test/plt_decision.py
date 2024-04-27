import os
import numpy as np
import matplotlib.pyplot as plt

def plot_combined_figure(trajectory_save_Trailing, trajectory_save_doRight, trajectory_save_doLeft, vehicle_1_trajectory, vehicle_2_trajectory, figure_dir, figure_name, iteration):
    plt.rc('font', family='Times New Roman')
    fig, ax = plt.subplots(figsize=(12, 6))  # Single plot

    # Plot decision trajectories with increased marker size and edge
    ax.plot(trajectory_save_Trailing[0, :], trajectory_save_Trailing[1, :], label='Trailing', linestyle='-', marker='o', markersize=8, markeredgecolor='black', markeredgewidth=1)
    ax.plot(trajectory_save_doRight[0, :], trajectory_save_doRight[1, :], label='doRight', linestyle='--', marker='^', markersize=8, markeredgecolor='black', markeredgewidth=1)
    ax.plot(trajectory_save_doLeft[0, :], trajectory_save_doLeft[1, :], label='doLeft', linestyle='-.', marker='s', markersize=8, markeredgecolor='black', markeredgewidth=1)

    # Plot vehicle trajectories with distinct colors and transparency
    ax.plot(vehicle_1_trajectory[0, :], vehicle_1_trajectory[1, :], label='Vehicle 1', linestyle=':', marker='x', markersize=8, alpha=1, color='blue')
    ax.plot(vehicle_2_trajectory[0, :], vehicle_2_trajectory[1, :], label='Vehicle 2', linestyle=':', marker='d', markersize=8, alpha=0.5, color='red')

    # Setup plot details
    ax.set_title('Iteration {}'.format(iteration))
    ax.set_xlabel('X Position [m]')
    ax.set_ylabel('Y Position [m]')
    
    
    
    # Draw black and dashed lines
    ax.plot([40+(iteration/10-1)*40, 160+(iteration/10-1)*40], [143.318146 - 1.75, 143.318146 - 1.75], 'k')
    ax.plot([40+(iteration/10-1)*40, 160+(iteration/10-1)*40], [143.318146 + 1.75, 143.318146 + 1.75], 'k')
    ax.plot([40+(iteration/10-1)*40, 160+(iteration/10-1)*40], [143.318146 - 1.75 - 3.5, 143.318146 - 1.75 - 3.5], 'k')
    ax.plot([40+(iteration/10-1)*40, 160+(iteration/10-1)*40], [143.318146 + 1.75 + 3.5, 143.318146 + 1.75 + 3.5], 'k')
    ax.plot([40+(iteration/10-1)*40, 160+(iteration/10-1)*40], [143.318146, 143.318146], 'k--')
    ax.plot([40+(iteration/10-1)*40, 160+(iteration/10-1)*40], [143.318146 - 3.5, 143.318146 - 3.5], 'k--')
    ax.plot([40+(iteration/10-1)*40, 160+(iteration/10-1)*40], [143.318146 + 3.5, 143.318146 + 3.5], 'k--')
    ax.legend()

    plt.tight_layout()
    plt.legend(loc='upper right')
    
    # Check if the directory exists and create if not
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    fig.savefig(os.path.join(figure_dir, figure_name), format='png', dpi=300)
    plt.show()

# Example usage
# Load your trajectory data
iteration = 30
trajectory_save_ = np.load(r'C:\Users\A490243\Desktop\Master_Thesis\Parameters\trajectory_save_{}.npy'.format(iteration))
trajectory_save_Trailing = trajectory_save_[2, :, :]
trajectory_save_doRight = trajectory_save_[1, :, :]
trajectory_save_doLeft = trajectory_save_[0, :, :]

# Load vehicle trajectory data
i =int(iteration/10)

X_traffic = np.load(r'C:\Users\A490243\Desktop\Master_Thesis\Parameters\X_traffic_1.npy')
vehicle_1_trajectory = X_traffic[:, 13*(i-1):13*(i), 0]
vehicle_2_trajectory = X_traffic[:, 13*(i-1):13*(i), 5]

figure_dir = "C:/Users/A490243/Desktop/Master_Thesis/Figure"
figure_name = "Combined_Trajectories_{}.png".format(iteration)

# Call the function with the loaded data
plot_combined_figure(trajectory_save_Trailing, trajectory_save_doRight, trajectory_save_doLeft, vehicle_1_trajectory, vehicle_2_trajectory, figure_dir, figure_name,iteration)



#!                               LEFT,           RIGHT,              TRAILING
# INFO:  Controller cost 931.9183409431374 3585.4411878904157 4.244763215895342 Slack: 3.7460214906452856e-17 1.3085044344063195 1.4330095358257274e-18 )
# INFO:  Decision:  trailing

# INFO:  Controller cost 929.1501934944088 6136.019528372632 33261.86923296473 Slack: 3.7510198435162395e-17 12.062767986390273 59.889177844233494 )
# INFO:  Decision:  leftChange

# INFO:  Controller cost 926.3820460456802 8686.597868854849 66419.49375000001 Slack: 3.756018196387193e-17 22.817031538374227 118.34934615384616 )
# INFO:  Decision:  leftChange
