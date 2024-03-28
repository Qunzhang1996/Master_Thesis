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
# ██████╗  █████╗ ███████╗███████╗
# ██╔══██╗██╔══██╗██╔════╝██╔════╝
# ██████╔╝███████║███████╗███████╗
# ██╔═══╝ ██╔══██║╚════██║╚════██║
# ██║     ██║  ██║███████║███████║
# ╚═╝     ╚═╝  ╚═╝╚══════╝╚══════╝
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
velocity1 = carla.Vector3D(10, 0, 0)
velocity2 = carla.Vector3D(15, 0, 0)
car.set_target_velocity(velocity1)
truck.set_target_velocity(velocity2)
time.sleep(1)

desired_interval = 0.2  # Desired time interval in seconds
# initial the carla built in pid controller
car_contoller = VehiclePIDController(car, 
                                     args_lateral = {'K_P': 1.1, 'K_I': 0.2, 'K_D': 0.02, 'dt': desired_interval}, 
                                     args_longitudinal = {'K_P': 0.950, 'K_I': 0.1, 'K_D': 0.05, 'dt': desired_interval})
local_controller = VehiclePIDController(truck, 
                                        args_lateral = {'K_P': 0.95, 'K_I': 0.2, 'K_D': 0.03, 'dt': desired_interval}, 
                                        args_longitudinal = {'K_P': 1.7, 'K_I': 0.5, 'K_D': 0.1, 'dt': desired_interval})

#------------------initialize the robust MPC---------------------
ref_velocity = 15
dt = 0.2
N=12
vehicleADV = car_VehicleModel(dt,N, width = 2, length = 4)
nx,nu,nrefx,nrefu = vehicleADV.getSystemDim()
int_opt = 'rk'
vehicleADV.integrator(int_opt,dt)
F_x_ADV  = vehicleADV.getIntegrator()
vx_init_ego = 15
vehicleADV.setInit([20,143.318146],vx_init_ego)
Q_ADV = [0,1e3,3e2,5]                            # State cost, Entries in diagonal matrix
R_ADV = [5,5]                                   # Input cost, Entries in diagonal matrix
vehicleADV.cost(Q_ADV,R_ADV)
vehicleADV.costf(Q_ADV)
L_ADV,Lf_ADV = vehicleADV.getCost()
P0, process_noise, possibility = set_stochastic_mpc_params()
mpc_controller = MPC(vehicleADV, np.diag(Q_ADV), np.diag(R_ADV), P0, process_noise, possibility, N)

# get initial state of truck 
x_iter = DM(int(nx),1)
x_iter[:],u_iter = vehicleADV.getInit()
print(f"initial state of the truck is: {x_iter}")
print(f"initial input of the truck is: {u_iter}")

ref_trajectory = np.zeros((nx, N + 1)) # Reference trajectory (states)
ref_trajectory[0,:] = 0
ref_trajectory[1,:] = 143.318146
ref_trajectory[2,:] = ref_velocity
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
Trajectory_pred = []  # To store the predicted trajectory
timestamps = []  # To store timestamps for calculating acceleration and jerk
all_tightened_bounds = []  # To store all tightened bounds for visualization
previous_acceleration = 0  # To help in jerk calculation
car_state = get_state(car)
car_x, car_y, car_v = car_state[C_k.X_km].item(), car_state[C_k.Y_km].item(), car_state[C_k.V_km].item()
p_leading = car_x


# Simulation parameters
start_time = time.time()  # Record the start time of the simulation
noise = np.random.normal(0, 0.5)
velocity_leading = 11


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
    p_leading=car_x
    if i%1==0:
        # get the CARLA state
        x_iter = [truck_x, truck_y, truck_v, truck_psi]
        # print(f"current state of the truck is: {x_iter}")
        # exit()
        vel_diff=smooth_velocity_diff(p_leading, truck_x) # prevent the vel_diff is too small
        u_opt, x_opt, lambda_s, tightened_bound_N_IDM_list = mpc_controller.solve(x_iter, ref_trajectory, ref_control, 
                                                      p_leading, velocity_leading, vel_diff)
        # Example place inside your loop
        all_tightened_bounds.append(tightened_bound_N_IDM_list)  # Ensure you're copying the list if it's mutable
        Trajectory_pred.append(x_opt[:2,:]) # store the predicted trajectory
        # print("this the constrained tightened_bound_N_IDM_list: ",tightened_bound_N_IDM_list)
        x_iter=x_opt[:,1]
    #PID controller according to the x_iter of the MPC
    control_truck = local_controller.run_step(x_iter[2]*3.6, x_iter[1], False)
    truck.apply_control(control_truck)
    
    truck_vel_mpc.append(x_iter[2])
    lambda_s_list.append(lambda_s)
    
    car_x, car_y, car_v = car_state[C_k.X_km].item(), car_state[C_k.Y_km].item(), car_state[C_k.V_km].item()
    truck_x, truck_y, truck_v, truck_psi = truck_state[C_k.X_km].item(), truck_state[C_k.Y_km].item(), truck_state[C_k.V_km].item(), truck_state[C_k.Psi].item()
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
    if i == 220: break
    
    
    
# gif_dir = r'C:\Users\A490243\Desktop\Master_Thesis\Figure'
# gif_name = 'IDM_constraint_simulation_plots.gif'
# animate_constraints(all_tightened_bounds, truck_positions, car_positions, Trajectory_pred, gif_dir,gif_name)
# figure_dir = r'C:\Users\A490243\Desktop\Master_Thesis\Figure'
# figure_name = 'simulation_plots.png'
# plot_and_save_simulation_data(truck_positions, timestamps, truck_velocities, truck_accelerations, truck_jerks, 
#                               car_positions, leading_velocities, ref_velocity, truck_vel_mpc, truck_vel_control, 
#                               figure_dir,figure_name)


