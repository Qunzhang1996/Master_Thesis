import math
import carla
import os
import json
import imageio
import numpy as np
from enum import IntEnum
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Param:
    """
    here is the parameters for the vehicle
    """
    
    dt = 0.2  # [s] time step
     # vehicle config
    RF = 16.71554/3 # [m] distance from rear to vehicle front end of vehicle
    RB = 0.8  # [m] distance from rear to vehicle back end of vehicle
    W = 2.4  # [m] width of vehicle
    WD = 0.7 * W  # [m] distance between left-right wheels
    WB =  6  # [m] Wheel base
    TR = 0.44  # [m] Tyre radius
    TW = 0.7  # [m] Tyre width

class C_k(IntEnum):
    X_km, Y_km, V_km, Psi =range(4)
    
    
    


def setup_carla_environment(Sameline_ACC=True):
    """
    Sets up the CARLA environment by connecting to the server, 
    spawning vehicles, and setting their initial states.
    Returns the car and truck actors.
    """
    '''
    Sameline_ACC=TRUE: the car and truck are in the same lane, and the truck trakcing the car
    Sameline_ACC=FALSE: the car and truck are in the different lane
    '''
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    # Destroy existing vehicles
    for actor in world.get_actors().filter('vehicle.*'):
        actor.destroy()
    center_line = 143.318146
    if Sameline_ACC :
        # ! this scenario is for the ACC
        # Spawn Tesla Model 3
        car_bp = bp_lib.find('vehicle.tesla.model3')
        # car_bp = bp_lib.find('vehicle.ford.ambulance')
        car_spawn_point = carla.Transform(carla.Location(x=120, y=center_line, z=0.3))
        car = spawn_vehicle(world, car_bp, car_spawn_point)

        # Spawn Firetruck
        truck_bp = bp_lib.find('vehicle.carlamotors.firetruck')
        # truck_bp = bp_lib.find('vehicle.carlamotors.european_hgv')
        truck_spawn_point = carla.Transform(carla.Location(x=20, y=center_line, z=0.3))
        truck = spawn_vehicle(world, truck_bp, truck_spawn_point)
        return car, truck
    else:
        # ! this scenario is for the lane changing
        # Spawn Tesla Model 3
        car_bp = bp_lib.find('vehicle.tesla.model3')
        car_spawn_point = carla.Transform(carla.Location(x=30, y=center_line+3.5, z=0.3))
        car = spawn_vehicle(world, car_bp, car_spawn_point)

        # Spawn Firetruck
        truck_bp = bp_lib.find('vehicle.carlamotors.firetruck')
        truck_spawn_point = carla.Transform(car_spawn_point.location + carla.Location(y=-3.5, x=0))
        truck = spawn_vehicle(world, truck_bp, truck_spawn_point)

        return car, truck, center_line
    
    
def setup_complex_carla_environment():
    """
    Sets up a CARLA environment by connecting to the server, destroying existing vehicles,
    and spawning a selection of vehicles with initial states.
    """
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    # Destroy existing vehicles
    for actor in world.get_actors().filter('vehicle.*'):
        actor.destroy()

    vehicles = [
        ('vehicle.tesla.model3', 80),
        ('vehicle.carlamotors.firetruck', 20), # ! this is leading vehicle
        ('vehicle.ford.mustang', 20, -7),
        ('vehicle.carlamotors.carlacola', 20, -3.5),
        ('vehicle.lincoln.mkz_2017', 80, -3.5),
        ('vehicle.ford.ambulance', 200),
        ('vehicle.nissan.patrol_2021', 20, 3.5),
        ('vehicle.mercedes.sprinter', 60, 3.5)
    ]
    center_line = 143.318146
    spawned_vehicles = []

    for vehicle_info in vehicles:
        bp = bp_lib.find(vehicle_info[0])
        location = carla.Location(x=vehicle_info[1], y=center_line + vehicle_info[2] if len(vehicle_info) > 2 else center_line, z=0.3)
        spawn_point = carla.Transform(location)
        vehicle = spawn_vehicle(world, bp, spawn_point)
        spawned_vehicles.append(vehicle)

    return spawned_vehicles, center_line


def spawn_vehicle(world, blueprint, spawn_point):
    """
    Attempts to spawn a vehicle at a given spawn point.
    Returns the vehicle actor if successful, None otherwise.
    """
    try:
        return world.spawn_actor(blueprint, spawn_point)
    except Exception as e:
        print(f"Error spawning vehicle: {e}")
        return None
  
def set_stochastic_mpc_params():
    
    # process_noise=DM(process_noise)
    P0 = np.array([[0.002071893043635329, -9.081755508401041e-07, 6.625835814018275e-06, 3.5644803460420476e-06],
                [-9.081755508401084e-07, 0.0020727698104820867, -1.059020050149629e-05, 2.218506297137566e-06], 
                [6.625835814018292e-06, -1.0590200501496282e-05, 0.0020706909538251504, 5.4487618678242517e-08], 
                [3.5644803460420577e-06, 2.2185062971375356e-06, 5.448761867824289e-08, 0.002071025561957967]])
    process_noise=np.eye(4)  # process noise
    process_noise[0,0]=2  # x bound is [0, 3]
    process_noise[1,1]=0.01/6  # y bound is [0, 0.1]
    process_noise[2,2]=1.8/6*2  # v bound is [0, 1.8]
    process_noise[3,3]=0.05/6  # psi bound is [0, 0.05]

    possibility=0.95  # possibility 
    return P0,process_noise,possibility

def get_state(vehicle):
    """Here is the func that help to get the state of the vehicle

    Args:
        vehicle (_type_): _description_

    Returns:
        _type_: _4X1_ vector that contains the state of the vehicle
        x, y, psi, v
    """
    vehicle_pos = vehicle.get_transform()
    vehicle_loc = vehicle_pos.location
    vehicle_rot = vehicle_pos.rotation
    vehicle_vel = vehicle.get_velocity()

    # Extract relevant states
    x = vehicle_loc.x 
    y = vehicle_loc.y 
    v = math.sqrt(vehicle_vel.x**2 + vehicle_vel.y**2)
    # v = vehicle_vel.length()  #converting it to km/hr
    psi = math.radians(vehicle_rot.yaw)  # Convert yaw to radians
    

    return np.array([[x, y, v, psi]]).T





    
#make sure angle in -pi pi
def angle_trans(angle):
    if angle > np.pi:
        angle = angle - 2*np.pi
    elif angle < -np.pi:
        angle = angle + 2*np.pi
    return angle

def smooth_max(x, beta=10):
    """choose the max(x,x**2)
    """
    exp_linear = np.exp(-beta * x)
    exp_quadratic = np.exp(beta * x**2)
    return (exp_linear * -x + exp_quadratic * x**2) / (exp_linear + exp_quadratic)

def check_dimensions_car(nx, nu):
    """
    Check if the dimensions nx and nu are as expected.

    Parameters:
    - nx (int): The number of state variables.
    - nu (int): The number of control inputs.

    Raises:
    - ValueError: If nx is not equal to 4 or nu is not equal to 2.
    """
    if nx != 4 or nu != 2:
        raise ValueError("Error: nx must be equal to 4 and nu must be equal to 2.")
    
def getTotalCost(L,Lf,x,u,refx,refu,N):
            cost = 0
            for i in range(0,N):
                cost += L(x[:,i],u[:,i],refx[:,i],refu[:,i])
            cost += Lf(x[:,N],refx[:,N])
            return cost
        
        
def smooth_velocity_diff(p_leading, truck_x):
    # Desired maximum value
    max_val = 5
    
    # Sigmoid parameters
    k = 1  # Steepness of the sigmoid curve; adjust as needed
    x0 = 0   # This can be adjusted to control when the function starts to approach 5
    
    # Calculate the difference
    difference = p_leading - truck_x
    
    # Normalizing the difference to fit into the sigmoid function effectively
    # This normalization factor adjusts the difference to a scale where the sigmoid's output is most sensitive
    normalization_factor = 80 / 5  # Adjust based on the desired sensitivity
    
    # Sigmoid function to ensure smooth transition
    sigmoid = 1 / (1 + np.exp(-k * (difference - x0) * normalization_factor))
    
    # Scale the sigmoid output to have a maximum of 5
    vel_diff = sigmoid * max_val
    
    return vel_diff


  
    
def plot_paths(true_x, true_y,estimated_x, estimated_y,t):
    plt.figure(1)
    plt.cla()
    plt.plot(true_x, true_y, label='True Path', color='blue')
    plt.plot(estimated_x, estimated_y, label='Estimated Path', color='red')
    plt.scatter(true_x[-1], true_y[-1], color='blue', s=50)
    plt.scatter(estimated_x[-1], estimated_y[-1], color='red', s=50)
    plt.title(f"Time: {t:.2f}s - Paths")
    plt.legend()
    # plt.gca().invert_xaxis()


def plot_diff(t_axis,x_difference,y_difference):
    # For x difference
    plt.figure(2, figsize=(5, 5))
    plt.plot(t_axis.flatten(), x_difference[:-2], label='x Difference', color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('x Difference')
    plt.title('Difference in x over Time')
    plt.legend()
    plt.savefig('C:\\Users\\A490243\\Desktop\\Master_Thesis\\Figure\\x_difference.jpg')

    # For y difference
    plt.figure(3, figsize=(5, 5))
    plt.plot(t_axis.flatten(), y_difference[:-2], label='y Difference', color='orange')
    plt.xlabel('Time (s)')
    plt.ylabel('y Difference')
    plt.title('Difference in y over Time')
    plt.legend()
    plt.savefig('C:\\Users\\A490243\\Desktop\\Master_Thesis\\Figure\\y_difference.jpg')
    plt.show()
    
    



def animate_constraints(all_tightened_bounds, truck_positions, car_position, Trajectory_pred, figure_dir, gif_name,
                        y_center=143.318146, width=3.5, vehicle_width=2, vehicle_length=6):
    """
    Animates the changes in constraint boxes, vehicle position, and predicted trajectories over iterations.

    :param all_tightened_bounds: A list of lists, each containing tightened_bound_N_IDM_list values for each iteration.
    :param truck_positions: A list of tuples, each containing the (x, y) position of the truck for each iteration.
    :param car_position: A list of tuples, each containing the (x, y) position of the car for each iteration.
    :param Trajectory_pred: A list of lists, each containing the predicted (x, y) positions of the truck for each iteration.
    :param y_center: Y coordinate for the center of the constraint boxes.
    :param width: The width of the constraint boxes.
    :param vehicle_width: The width of the vehicle representation.
    :param vehicle_length: The length of the vehicle representation.
    """
    # Set figure size to 24x6 inches
    fig, ax = plt.subplots(figsize=(24, 2))  # Adjusted to more standard aspect ratio
    fig.tight_layout()
    ax.set_xlim(min(min(bounds) for bounds in all_tightened_bounds), max(max(bounds) for bounds in all_tightened_bounds))
    ax.set_ylim(y_center - width - 15, y_center + width + 15)

    def init():
        return []

    def update(frame):
        ax.clear()
        # Central line
        ax.plot([0, max(max(bounds) for bounds in all_tightened_bounds)], [y_center, y_center], 'k--')
        
        # Reset the y-axis limits after clearing
        ax.set_ylim(y_center - width - 5, y_center + width + 5)
        
        # Constraint box
        x_right_end = min(all_tightened_bounds[frame])
        x_right_end2 = max(all_tightened_bounds[frame])
        rect = plt.Rectangle((0, y_center - width / 2), x_right_end, width, fill=None, edgecolor='red', linewidth=2)
        rect2 = plt.Rectangle((0, y_center - width / 2), x_right_end2, width, fill=None, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        ax.add_patch(rect2)
        
        # Vehicle representation
        truck_x, truck_y = truck_positions[frame]
        car_x, car_y = car_position[frame]
        vehicle_rect = plt.Rectangle((truck_x - vehicle_length / 2, truck_y - vehicle_width / 2), vehicle_length, vehicle_width, color='blue')
        vehicle_rect_car = plt.Rectangle((car_x - vehicle_length / 2, car_y - vehicle_width / 2), vehicle_length, vehicle_width, color='red')
        ax.add_patch(vehicle_rect)
        ax.add_patch(vehicle_rect_car)

        # Predicted trajectory
        if Trajectory_pred and frame < len(Trajectory_pred):
            pred_x, pred_y = Trajectory_pred[frame][0], Trajectory_pred[frame][1]  # Assuming each element is a tuple (x, y)
            ax.plot(pred_x, pred_y, 'g--', label='Predicted Trajectory')

        # Update legend to include predicted trajectory
        # ax.legend(['', 'Constraint Box', 'Truck', 'Car', 'Predicted Trajectory'], loc='upper right')
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Constraint Box and Vehicle Position with Predicted Trajectory')
        
        return [rect, vehicle_rect, vehicle_rect_car]

    ani = FuncAnimation(fig, update, frames=range(len(all_tightened_bounds)), init_func=init, blit=False, repeat=False)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    figure_path = os.path.join(figure_dir, gif_name)
    ani.save(figure_path, writer='imagemagick', fps=10)  # Adjust fps as needed

    plt.close(fig)  # Prevent the figure from displaying inline or in a window
    
    
# ! here is the function to plot the trajectory of the truck (kalman true and estimated)
def plot_kf_trajectory(truck_positions, estimated_position, figure_dir, figure_name):
    plt.figure(figsize=(12, 4))
    plt.plot([pos[0] for pos in truck_positions], [pos[1] for pos in truck_positions], label='True Trajectory', color='blue')
    plt.plot([pos[0] for pos in estimated_position], [pos[1] for pos in estimated_position], label='Estimated Trajectory', color='red')
    plt.scatter(truck_positions[-1][0], truck_positions[-1][1], color='blue', s=50)
    plt.scatter(estimated_position[-1][0], estimated_position[-1][1], color='red', s=50)
    #! plt truck boundary 
    plt.plot([pos[0] for pos in truck_positions], [pos[1]+2.54/2 for pos in truck_positions], label='truck upper boundary', color='green')
    plt.plot([pos[0] for pos in truck_positions], [pos[1]-2.54/2 for pos in truck_positions], label='truck lower boundary', color='green')
    #! plt road boundary
    plt.plot([0, 700], [143.318146 - 1.75, 143.318146 - 1.75], 'k--')
    plt.plot([0, 700], [143.318146 + 1.75, 143.318146 + 1.75], 'k--')
    # ! plot center line
    plt.plot([0, 700], [143.318146, 143.318146], 'k--')
    plt.ylim(143.318146 - 7, 143.318146 + 7)
    plt.title('True and Estimated Trajectories')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    
    # Save the plot
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    figure_path = os.path.join(figure_dir, figure_name)
    plt.savefig(figure_path)
    plt.show()
    
# ! create kf and true trajectory plot
def plot_and_create_gifs(truck_positions, estimated_position, figure_dir, base_figure_name):
    # Calculate differences
    x_difference = [(est[0] - true[0]) for est, true in zip(estimated_position, truck_positions)]
    y_difference = [(est[1] - true[1]) for est, true in zip(estimated_position, truck_positions)]
    
    def plot_frame(idx, save_path):
        fig, axs = plt.subplots(3, 1, figsize=(12, 9))  # Create a 12x12 figure, then split it into 3 subplots
        time_array = np.arange(0, idx * 0.2, 0.2)[:idx]
        # Trajectory plot
        axs[0].plot([pos[0] for pos in truck_positions][:idx], [pos[1] for pos in truck_positions][:idx], label='True Trajectory', color='blue')
        axs[0].plot([pos[0] for pos in estimated_position][:idx], [pos[1] for pos in estimated_position][:idx], label='Estimated Trajectory', color='red')
        # plot center line
        axs[0].plot([0, 700], [143.318146, 143.318146], 'k--')
        #! plt road boundary
        axs[0].plot([0, 700], [143.318146 - 1.75, 143.318146 - 1.75], 'k--')
        axs[0].plot([0, 700], [143.318146 + 1.75, 143.318146 + 1.75], 'k--')
        #! plt truck boundary
        axs[0].plot([pos[0] for pos in truck_positions][:idx],  [pos[1]+2.54/2 for pos in truck_positions][:idx], label='truck upper boundary', color='green')
        axs[0].plot([pos[0] for pos in truck_positions][:idx], [pos[1]-2.54/2 for pos in truck_positions][:idx], label='truck lower boundary', color='green')
        
        axs[0].legend()
        axs[0].set_xlabel('X Position')
        axs[0].set_ylabel('Y Position')
        axs[0].set_title('Trajectories')

        # X difference plot
        axs[1].plot(time_array, x_difference[:idx], label='X Difference', color='orange')
        axs[1].legend()
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Difference')
        axs[1].set_title('X Difference')

        # Y difference plot
        axs[2].plot(time_array, y_difference[:idx], label='Y Difference', color='purple')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Difference')
        axs[2].legend()
        axs[2].set_title('Y Difference')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    frame_paths = []
    for idx in range(1, len(truck_positions) + 1):
        frame_path = os.path.join(figure_dir, f'frame_{idx}.png')
        plot_frame(idx, frame_path)
        frame_paths.append(frame_path)

    # Create GIFs
    gif_paths = [os.path.join(figure_dir, f"{base_figure_name}_{name}.gif") for name in ['trajectory', 'x_difference', 'y_difference']]
    # Since the plots are on separate subplots, we only need to generate one set of frames
    with imageio.get_writer(gif_paths[0], mode='I') as writer:  # Trajectory GIF
        for frame_path in frame_paths:
            image = imageio.imread(frame_path)
            writer.append_data(image)
    # The x_difference and y_difference plots are part of the same frames, so no separate GIF creation is needed

    # Optionally remove the frames after creating the GIF
    for frame_path in frame_paths:
        os.remove(frame_path)


def plot_mpc_y_vel(y_mpc, vel_mpc, y_true, vel_true, figure_dir, figure_name):
    # Create a 2x1 plot layout
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    
    # Calculate time array based on 0.2s per point assumption
    time_array = np.arange(len(y_mpc)) * 0.2
    
    # Y position plot
    axs[0].plot(time_array, y_mpc, label='MPC Y Target Position', color='r')
    axs[0].plot(time_array, y_true, label='True Y Position', color='b')
    # center line
    axs[0].plot(time_array, [143.318146]*len(time_array), 'k--')
    # road boundary
    axs[0].plot(time_array, [143.318146 - 1.75]*len(time_array), 'k--')
    axs[0].plot(time_array, [143.318146 + 1.75]*len(time_array), 'k--')
    axs[0].set_title('Y Position')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Y Position')
    axs[0].legend()
    
    # Velocity plot
    axs[1].plot(time_array, vel_mpc, label='MPC Target Velocity', color='r')
    axs[1].plot(time_array, vel_true, label='True Velocity', color='b')
    # plot the reference velocity
    axs[1].plot(time_array, [15]*len(time_array), 'k--')
    axs[1].set_title('Velocity')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Velocity')
    axs[1].set_ylim([5, 20])
    axs[1].legend()
    
    # Adjust layout for a neat presentation
    plt.tight_layout()
    
    # Save the plot
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    figure_path = os.path.join(figure_dir, figure_name)
    plt.savefig(figure_path)
    plt.show()
    
    
def create_gif_with_plot(y_mpc, vel_mpc, y_true, vel_true, figure_dir, figure_name):
    # Ensure figure directory exists
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    figure_path = os.path.join(figure_dir, figure_name + '.gif')

    # Create a 2x1 plot layout for the animation outside of the update function
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))

    def update(frame):
        # Clear the axes to redraw the updated plots
        for ax in axs:
            ax.clear()

        time_array = np.arange(frame) * 0.2  # Calculate time array up to the current frame
        
        # Redrawing the plots with updated data
        axs[0].plot(time_array, y_mpc[:frame], label='MPC Y Target Position', color='r')
        axs[0].plot(time_array, y_true[:frame], label='True Y Position', color='b')
        axs[0].plot(time_array, [143.318146]*len(time_array), 'k--')
        axs[0].plot(time_array, [143.318146 - 1.75]*len(time_array), 'k--')
        axs[0].plot(time_array, [143.318146 + 1.75]*len(time_array), 'k--')
        axs[0].set_title('Y Position')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Y Position')
        axs[0].legend()
        
        axs[1].plot(time_array, vel_mpc[:frame], label='MPC Target Velocity', color='r')
        axs[1].plot(time_array, vel_true[:frame], label='True Velocity', color='b')
        axs[1].plot(time_array, [15]*len(time_array), 'r--',label='Ref Velocity')
        axs[1].plot(time_array, [11]*len(time_array),'k--', label='Leading Velocity')
        axs[1].set_title('Velocity')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Velocity')
        axs[1].set_ylim([5, 20])
        axs[1].legend()
        
        plt.tight_layout()

    # Creating the animation using the same figure and axes
    anim = FuncAnimation(fig, update, frames=len(y_mpc), blit=False)
    
    # Save the animation
    anim.save(figure_path, writer='pillow', fps=20)
    plt.close(fig)  # Ensure the figure is closed to free up memory
    
    
    
    
    
    
    
############################################################################################################
    
def plot_and_save_simulation_data(truck_positions, timestamps, truck_velocities, truck_accelerations,
                                  truck_jerks, car_positions, leading_velocities, ref_velocity, truck_vel_mpc, truck_vel_control, 
                                  figure_dir,figure_name):
    # Prepare data for plotting
    x_positions, y_positions = zip(*truck_positions) if truck_positions else ([], [])
    x_positions_leading, y_positions_leading = zip(*car_positions) if car_positions else ([], [])
    velocity_times = timestamps[1:] if len(timestamps) > 1 else []
    acceleration_times = timestamps[2:] if len(timestamps) > 2 else []
    jerk_times = timestamps[3:] if len(timestamps) > 3 else []
    
    fig, axs = plt.subplots(2, 3, figsize=(12, 10)) 
    # Trajectory plot
    if x_positions and y_positions:
        axs[0, 0].plot(x_positions, y_positions, '-', color='r', label='Trajectory')
        # Correct way to plot car contour with a blue dash line
        left_boundary_x = [x - 3 for x in x_positions]
        right_boundary_x = [x + 3 for x in x_positions]
        left_boundary_y = [y - 1.2 for y in y_positions]
        right_boundary_y = [y + 1.2 for y in y_positions]

        axs[0, 0].plot(left_boundary_x, left_boundary_y, 'b--', label='Left car Boundary')
        axs[0, 0].plot(right_boundary_x, right_boundary_y, 'b--', label='Right car Boundary')

        # plot road boundary with black dash line
        axs[0, 0].plot([0, 700], [143.318146 - 1.75, 143.318146 - 1.75], 'k--')
        axs[0, 0].plot([0, 700], [143.318146 + 1.75, 143.318146 + 1.75], 'k--')
        axs[0, 0].set_title('Truck Trajectory')
        axs[0, 0].set_xlabel('X Position')
        axs[0, 0].set_ylim([143.318146 - 7, 143.318146 + 7])
        axs[0, 0].set_ylabel('Y Position')
        axs[0, 0].grid(True)
        axs[0, 0].legend()

    # Velocity plot
    if velocity_times and truck_velocities:
        axs[0, 1].plot(velocity_times, truck_velocities[:-1], '-', color='r',label='Truck Velocity')
        axs[0, 1].plot(velocity_times, leading_velocities[:-1], '-', color='g' ,label='Leading Velocity')
        # plot the reference velocity
        axs[0, 1].plot(velocity_times, [ref_velocity]*len(velocity_times), '--', color='b',label='Reference Velocity')
        axs[0, 1].set_ylim([0, 20])
        axs[0, 1].set_title('Velocity')
        axs[0, 1].set_xlabel('Time')
        axs[0, 1].set_ylabel('Velocity (m/s)')
        axs[0, 1].grid(True)
        axs[0, 1].legend()

    # Acceleration plot
    if acceleration_times and truck_accelerations:
        axs[1, 0].plot(acceleration_times, truck_accelerations[:-2], '-', color='r',label='Acceleration')
        axs[1, 0].set_title('Truck Acceleration')
        axs[1, 0].set_xlabel('Time')
        axs[1, 0].set_ylim([-4, 4])
        axs[1, 0].set_ylabel('Acceleration (m/s²)')
        axs[1, 0].grid(True)
        axs[1, 0].legend()

    # Jerk plot
    if jerk_times and truck_jerks:
        axs[1, 1].plot(jerk_times, truck_jerks[:-3], '-', color='r',label='Jerk')
        axs[1, 1].set_ylim([-15, 15])
        axs[1, 1].set_title('Truck Jerk')
        axs[1, 1].set_xlabel('Time')
        axs[1, 1].set_ylabel('Jerk (m/s³)')
        axs[1, 1].grid(True)
        axs[1, 1].legend()
        
    # difference between leading car and truck plot
    if x_positions_leading and y_positions_leading:
        axs[0, 2].plot(timestamps,np.sqrt((np.array(x_positions_leading)-np.array(x_positions))**2+(np.array(y_positions_leading)-np.array(y_positions))**2), 
                       '-',color='r',label='Difference')
        axs[0, 2].set_ylim([0, 120])
        axs[0, 2].set_title('distance between Leading Car and Truck')
        axs[0, 2].set_xlabel('Time')
        axs[0, 2].set_ylabel('distance (m)')
        axs[0, 2].grid(True)
        axs[0, 2].legend()
        
    # difference between mpc target and truck velocity plot
    if velocity_times and truck_velocities:
        # axs[1, 2].plot(velocity_times, np.array(truck_velocities[:-1])-np.array(leading_velocities[:-1]), '-', color='r',label='Difference')
        axs[1, 2].plot(velocity_times, np.array(truck_vel_mpc[1:]), '-', color='r', label='mpc reference velocity')
        axs[1, 2].plot(velocity_times, np.array(truck_vel_control[:-1]), '-', color='b', label='truck velocity')
        # axs[1, 2].set_ylim([5, 20])
        axs[1, 2].set_title('MPC reference and velocity after pid control')
        axs[1, 2].set_xlabel('Time')
        axs[1, 2].set_ylabel('Velocity (m/s)')
        axs[1, 2].grid(True)
        axs[1, 2].legend() 
        # legend right top
        axs[1, 2].legend(loc='upper right')

        
    
    # Adjust layout for a neat presentation
    plt.tight_layout()



    # Save the plot
    # figure_dir = r'C:\Users\A490243\Desktop\Master_Thesis\Figure'
    figure_path = os.path.join(figure_dir, figure_name)
    plt.savefig(figure_path)
    # Show the plot
    plt.show()



