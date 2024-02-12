from casadi import *
import carla
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import os
import json
sys.path.append(r'C:\Users\A490243\CARLA\CARLA_Latest\WindowsNoEditor\PythonAPI\carla')
sys.path.append(r'C:\Users\A490243\Desktop\Master_Thesis')
from Controller.LTI_MPC import MPC
from vehicleModel.vehicle_model import car_VehicleModel
from agents.navigation.controller import VehiclePIDController
from util.utils import *

# # ------------------------change map to Town06------------------------
# import subprocess
# # Command to run your script
# command = (
#     r'cd C:\Users\A490243\CARLA\CARLA_Latest\WindowsNoEditor\PythonAPI\util && '
#     r'python config.py --map Town06')
# subprocess.run(command, shell=True)
# # Run the command

car,truck = setup_carla_environment(Sameline_ACC=True)
time.sleep(1)
velocity1 = carla.Vector3D(8, 0, 0)
velocity2 = carla.Vector3D(10, 0, 0)
car.set_target_velocity(velocity1)
truck.set_target_velocity(velocity2)

desired_interval = 0.2  # Desired time interval in seconds
# To start a basic agent
client = carla.Client('localhost', 2000)
world = client.get_world()
carla_map = world.get_map()
# initial the carla built in pid controller
car_contoller = VehiclePIDController(car, args_lateral = {'K_P': 2, 'K_I': 0.2, 'K_D': 0.02, 'dt': desired_interval}, 
                                     args_longitudinal = {'K_P': 1.95, 'K_I': 0.5, 'K_D': 0, 'dt': desired_interval})
local_controller = VehiclePIDController(truck, args_lateral = {'K_P': 2, 'K_I': 0.2, 'K_D': 0.01, 'dt': desired_interval}, 
                                        args_longitudinal = {'K_P': 1.95, 'K_I': 0.5, 'K_D': 0, 'dt': desired_interval})
# To start a behavior agent with an aggressive car for truck to track
spawn_points = carla_map.get_spawn_points()
destination = carla.Location(x=1000, y=143.318146, z=0.3)
print(f"destination: {destination}")

#------------------initialize the robust MPC---------------------

dt = 0.2
N=12
vehicleADV = car_VehicleModel(dt,N, width = 2, length = 4)
nx,nu,nrefx,nrefu = vehicleADV.getSystemDim()
int_opt = 'rk'
vehicleADV.integrator(int_opt,dt)
F_x_ADV  = vehicleADV.getIntegrator()
vx_init_ego = 10
vehicleADV.setInit([30,143.318146],vx_init_ego)
Q_ADV = [0,40,3e2,5]                            # State cost, Entries in diagonal matrix
R_ADV = [5,5]                                    # Input cost, Entries in diagonal matrix
vehicleADV.cost(Q_ADV,R_ADV)
vehicleADV.costf(Q_ADV)
L_ADV,Lf_ADV = vehicleADV.getCost()
# process_noise=DM(process_noise)
P0 = np.array([[0.002071893043635329, -9.081755508401041e-07, 6.625835814018275e-06, 3.5644803460420476e-06],
            [-9.081755508401084e-07, 0.0020727698104820867, -1.059020050149629e-05, 2.218506297137566e-06], 
            [6.625835814018292e-06, -1.0590200501496282e-05, 0.0020706909538251504, 5.4487618678242517e-08], 
            [3.5644803460420577e-06, 2.2185062971375356e-06, 5.448761867824289e-08, 0.002071025561957967]])
process_noise=np.eye(4)  # process noise
process_noise[0,0]=0.5  # x bound is [0, 3]
process_noise[1,1]=0.01/6  # y bound is [0, 0.1]
process_noise[2,2]=1.8/6  # v bound is [0, 1.8]
process_noise[3,3]=0.05/6  # psi bound is [0, 0.05]
mpc_controller = MPC(vehicleADV, np.diag([0,40,3e2,5]), np.diag([5,5]), P0, process_noise, 0.95, N)

# get initial state of truck 
x_iter = DM(int(nx),1)
x_iter[:],u_iter = vehicleADV.getInit()
print(f"initial state of the truck is: {x_iter}")
print(f"initial input of the truck is: {u_iter}")

ref_trajectory = np.zeros((nx, N + 1)) # Reference trajectory (states)
ref_trajectory[0,:] = 0
ref_trajectory[1,:] = 0
ref_trajectory[2,:] = 15
ref_control = np.zeros((nu, N))  # Reference control inputs

# Set the controller (this step initializes the optimization problem with cost and constraints)
mpc_controller.setController()


# Data storage initialization
car_positions = []  # To store (x, y) positions
truck_positions = []  # To store (x, y) positions
truck_velocities = []  # To store velocity
leading_velocities = []  # To store leading vehicle velocity
truck_accelerations = []  # To store acceleration
truck_jerks = []  # To store jerk
truck_vel_mpc = []
lambda_s_list = []
truck_vel_control = []
timestamps = []  # To store timestamps for calculating acceleration and jerk
previous_acceleration = 0  # To help in jerk calculation
car_state = get_state(car)
car_x, car_y, car_v = car_state[C_k.X_km].item(), car_state[C_k.Y_km].item(), car_state[C_k.V_km].item()
p_leading = car_x


# Simulation parameters
total_iterations = 100  # Total number of iterations
start_time = time.time()  # Record the start time of the simulation
noise = np.random.normal(0, 0.5)
velocity_leading = 10


for i in range(1000):
    
    iteration_start = time.time()
    # keep leading vehicle velocity keeps change
    # import  random  noise
    
    control_car = car_contoller.run_step(velocity_leading*3.6, 143.318146, False)
    car.apply_control(control_car)
    
    car_state = get_state(car)
    truck_state = get_state(truck)
    car_x, car_y, car_v = car_state[C_k.X_km].item(), car_state[C_k.Y_km].item(), car_state[C_k.V_km].item()
    truck_x, truck_y, truck_v, truck_psi = truck_state[C_k.X_km].item(), truck_state[C_k.Y_km].item(), truck_state[C_k.V_km].item(), truck_state[C_k.Psi].item()
    p_leading=p_leading + (velocity_leading)*desired_interval
    # p_leading=car_x
    if i%5==0:
        # get the CARLA state
        x_iter = [truck_x, truck_y, truck_v, truck_psi]
        u_opt, x_opt, lambda_s = mpc_controller.solve(x_iter, ref_trajectory, ref_control, p_leading)
        x_iter=x_opt[:,1]
    # print("leading velocity",car_v, "velocity of the truck: ", x_iter[2])
    truck_vel_mpc.append(x_iter[2])
    lambda_s_list.append(lambda_s)
    
    #PID controller according to the x_iter of the MPC
    control_truck = local_controller.run_step(x_iter[2]*3.6, x_iter[1], False)
    truck.apply_control(control_truck)
    
    truck_state_ctr = get_state(truck)
    truck_vel_ctr=truck_state_ctr[C_k.V_km].item()
    truck_vel_control.append(truck_vel_ctr)
    
    
    # Data collection inside the loop
    current_time = time.time()
    timestamps.append(current_time)
    car_positions.append((car_x, car_y))
    truck_positions.append((truck_x, truck_y))
    truck_velocities.append(truck_v)
    leading_velocities.append(car_v)

    # Calculate acceleration if possible
    if len(timestamps) > 1:
        delta_v = truck_velocities[-1] - truck_velocities[-2]
        delta_t = timestamps[-1] - timestamps[-2]
        acceleration = delta_v / delta_t
        truck_accelerations.append(acceleration)

        # Calculate jerk if possible
        if len(truck_accelerations) > 1:
            delta_a = truck_accelerations[-1] - previous_acceleration
            jerk = delta_a / delta_t
            truck_jerks.append(jerk)
            previous_acceleration = truck_accelerations[-1]
    else:
        truck_accelerations.append(0)  # Initial acceleration is 0
        truck_jerks.append(0)  # Initial jerk is 0
    
    
        
    
    # Dynamically adjust sleep time to maintain desired interval
    iteration_duration = time.time() - iteration_start
    sleep_duration = max(0.001, desired_interval - iteration_duration)
    time.sleep(sleep_duration)
    if i == 250: break






# Assuming the lists truck_positions, timestamps, truck_velocities, truck_accelerations, and truck_jerks are filled with your data
# Prepare data for plotting
x_positions, y_positions = zip(*truck_positions) if truck_positions else ([], [])
x_positions_leading, y_positions_leading = zip(*car_positions) if car_positions else ([], [])
velocity_times = timestamps[1:] if len(timestamps) > 1 else []
acceleration_times = timestamps[2:] if len(timestamps) > 2 else []
jerk_times = timestamps[3:] if len(timestamps) > 3 else []
# save data to json file
parameters_dir = r'C:\Users\A490243\Desktop\Master_Thesis\Parameters'
os.makedirs(parameters_dir, exist_ok=True)

# Prepare the data for saving
data_to_save = {
    "truck_positions": truck_positions,
    "truck_velocities": truck_velocities,
    "truck_accelerations": truck_accelerations,
    "truck_jerks": truck_jerks
}

# Save data as JSON
json_file_path = os.path.join(parameters_dir, 'simulation_data.json')
with open(json_file_path, 'w') as json_file:
    json.dump(data_to_save, json_file, indent=4)



# Here is for the plot
import matplotlib.pyplot as plt
# Create a 2x3 plot layout
fig, axs = plt.subplots(2, 3, figsize=(12, 10)) 
# Trajectory plot
if x_positions and y_positions:
    axs[0, 0].plot(x_positions, y_positions, '-',color='r',label='Trajectory')
    axs[0, 0].set_title('Truck Trajectory')
    axs[0, 0].set_xlabel('X Position')
    axs[0, 0].set_ylim([143.318146-1.75, 143.318146+1.75])
    axs[0, 0].set_ylabel('Y Position')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

# Velocity plot
if velocity_times and truck_velocities:
    axs[0, 1].plot(velocity_times, truck_velocities[:-1], '-', color='r',label='Truck Velocity')
    axs[0, 1].plot(velocity_times, leading_velocities[:-1], '-', color='g' ,label='Leading Velocity')
    # plot the reference velocity
    axs[0, 1].plot(velocity_times, [15]*len(velocity_times), '--', color='b',label='Reference Velocity')
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
    axs[1, 1].set_title('Truck Jerk')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Jerk (m/s³)')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    
# difference between leading car and truck plot
if x_positions_leading and y_positions_leading:
    axs[0, 2].plot(timestamps,np.sqrt((np.array(x_positions_leading)-np.array(x_positions))**2+(np.array(y_positions_leading)-np.array(y_positions))**2), 
                   '-',color='r',label='Difference')
    # axs[0, 2].set_ylim([10, 30])
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
    axs[1, 2].set_title('MPC reference and velocity after pid control')
    axs[1, 2].set_xlabel('Time')
    axs[1, 2].set_ylabel('Velocity (m/s)')
    axs[1, 2].grid(True)
    axs[1, 2].legend() 

#print lambda_s_list with time
# if lambda_s_list:
#     axs[0, 3].plot(velocity_times, np.array(lambda_s_list[:-1]), '-', color='r', label='lambda var')
#     axs[0, 3].set_title('slack var with time')
#     axs[0, 3].set_xlabel('Time')
#     axs[0, 3].grid(True)
#     axs[0, 3].legend()
    
   
# Adjust layout for a neat presentation
plt.tight_layout()
#delete useless pic
# plt.delaxes(axs[1, 3])


# Save the plot
figure_dir = r'C:\Users\A490243\Desktop\Master_Thesis\Figure'
figure_path = os.path.join(figure_dir, 'simulation_plots.png')
plt.savefig(figure_path)



# Show the plot
plt.show()

