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
# import observor
from kalman_filter.kalman_filter import kalman_filter
# import controller
from Controller.LTI_MPC import MPC
from vehicleModel.vehicle_model import car_VehicleModel
from agents.navigation.controller import VehiclePIDController
# import helpers
from util.utils import *

# # ------------------------change map to Town06------------------------
# import subprocess
# # Command to run your script
# command = (
#     r'cd C:\Users\A490243\CARLA\CARLA_Latest\WindowsNoEditor\PythonAPI\util && '
#     r'python config.py --map Town06')
# subprocess.run(command, shell=True)
# # --------------------------Run the command--------------------------



# ███████╗██╗   ██╗███╗   ██╗ ██████╗     ██████╗████████╗██████╗ ██╗         
# ██╔════╝╚██╗ ██╔╝████╗  ██║██╔════╝    ██╔════╝╚══██╔══╝██╔══██╗██║         
# ███████╗ ╚████╔╝ ██╔██╗ ██║██║         ██║        ██║   ██████╔╝██║         
# ╚════██║  ╚██╔╝  ██║╚██╗██║██║         ██║        ██║   ██╔══██╗██║         
# ███████║   ██║   ██║ ╚████║╚██████╗    ╚██████╗   ██║   ██║  ██║███████╗    
# ╚══════╝   ╚═╝   ╚═╝  ╚═══╝ ╚═════╝     ╚═════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝    
                                                                              
# !----------------- PID MPC Settings ------------------------                                                                                                                                     
SYNC_CTRL = False  # True for syncronized control, False for different control
frequence = 1 if SYNC_CTRL else 10  # frequence of the MPC controller
## !----------------- Carla Settings ------------------------
car,truck = setup_carla_environment(Sameline_ACC=True)
time.sleep(1)
velocity1 = carla.Vector3D(10, 0, 0)
velocity2 = carla.Vector3D(15, 0, 0)
car.set_target_velocity(velocity1)
truck.set_target_velocity(velocity2)
time.sleep(1)
client = carla.Client('localhost', 2000)
world = client.get_world()
carla_map = world.get_map()
# ██████╗ ██╗██████╗ 
# ██╔══██╗██║██╔══██╗
# ██████╔╝██║██║  ██║
# ██╔═══╝ ██║██║  ██║
# ██║     ██║██████╔╝
# ╚═╝     ╚═╝╚═════╝ 
## !----------------- PID Controller Settings ------------------------                   
desired_interval = 0.2  # Desired time interval in seconds
car_contoller = VehiclePIDController(car, 
                                     args_lateral = {'K_P': 1.1, 'K_I': 0.2, 'K_D': 0.02, 'dt': desired_interval}, 
                                     args_longitudinal = {'K_P': 0.950, 'K_I': 0.1, 'K_D': 0.05, 'dt': desired_interval})
local_controller = VehiclePIDController(truck, 
                                        args_lateral = {'K_P': 1.1, 'K_I': 0.2, 'K_D': 0.03, 'dt': desired_interval}, 
                                        args_longitudinal = {'K_P': 1.7, 'K_I': 0.5, 'K_D': 0.1, 'dt': desired_interval})
# ███╗   ███╗██████╗  ██████╗
# ████╗ ████║██╔══██╗██╔════╝
# ██╔████╔██║██████╔╝██║     
# ██║╚██╔╝██║██╔═══╝ ██║     
# ██║ ╚═╝ ██║██║     ╚██████╗
# ╚═╝     ╚═╝╚═╝      ╚═════╝
## !----------------- Robust MPC Controller Settings ------------------------                           
ref_velocity = 15  # TODO: here is the reference velocity for the truck
dt = desired_interval
N=12
vehicleADV = car_VehicleModel(dt,N, width = 2, length = 6)
nx,nu,nrefx,nrefu = vehicleADV.getSystemDim()
int_opt = 'rk'
vehicleADV.integrator(int_opt,dt)
F_x_ADV  = vehicleADV.getIntegrator()
vx_init_ego = 15
vehicleADV.setInit([20,143.318146],vx_init_ego)
Q_ADV = [0,40,5e2,5]                            # State cost, Entries in diagonal matrix
R_ADV = [5,5]                                    # Input cost, Entries in diagonal matrix
vehicleADV.cost(Q_ADV,R_ADV)
vehicleADV.costf(Q_ADV)
L_ADV,Lf_ADV = vehicleADV.getCost()
P0, process_noise, possibility = set_stochastic_mpc_params()
mpc_controller = MPC(vehicleADV, np.diag(Q_ADV), np.diag(R_ADV), P0, process_noise, possibility, N)

## !----------------- get initial state and set ref for the ccontroller ------------------------
                                                                                  
x_iter = DM(int(nx),1)
x_iter[:],u_iter = vehicleADV.getInit()
print(f"initial state of the truck is: {x_iter}")
print(f"initial input of the truck is: {u_iter}")

ref_trajectory = np.zeros((nx, N + 1)) # Reference trajectory (states)
ref_trajectory[0,:] = 0
ref_trajectory[1,:] = 143.318146
ref_trajectory[2,:] = ref_velocity
ref_trajectory[3,:] = 0
ref_control = np.zeros((nu, N))  # Reference control inputs

# Set the controller (this step initializes the optimization problem with cost and constraints)
mpc_controller.setController()


#  ██████╗ ██████╗ ███████╗███████╗██████╗ ██╗   ██╗ ██████╗ ██████╗ 
# ██╔═══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗██║   ██║██╔═══██╗██╔══██╗
# ██║   ██║██████╔╝███████╗█████╗  ██████╔╝██║   ██║██║   ██║██████╔╝
# ██║   ██║██╔══██╗╚════██║██╔══╝  ██╔══██╗╚██╗ ██╔╝██║   ██║██╔══██╗
# ╚██████╔╝██████╔╝███████║███████╗██║  ██║ ╚████╔╝ ╚██████╔╝██║  ██║
#  ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═╝  ╚═══╝   ╚═════╝ ╚═╝  ╚═╝
# !----------------- Kalman Filter Settings ------------------------                                                                                                                                                                                                                         
# set the process and measurement noise
sigma_process=0.1
sigma_measurement=0.01
Q_0=np.eye(nx)*sigma_process**2
Q_0[0,0]=2  # x bound is [0, 3]
Q_0[1,1]=0.01  # y bound is [0, 0.1]
Q_0[2,2]=1.8/6*2  # v bound is [0, 1.8]
Q_0[3,3]=0.001  # psi bound is [0, 0.05]
R_0=np.eye(nx)*sigma_measurement

# set the initial state and control input
x_0 = x_iter
P_kf=np.eye(nx)  # initial state covariance
u_iter = np.array([0,0])
# get system dynamic matrices
A,B,_=mpc_controller.get_dynammic_model()
H=np.eye(nx)   #measurement matrix, y=H@x_iter
# initial the Kalman Filter
ekf=kalman_filter(A,B,H,x_0,P_kf,Q_0,R_0)

# ██████╗  █████╗ ████████╗ █████╗     ███████╗████████╗ ██████╗ ██████╗  █████╗  ██████╗ ███████╗
# ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗    ██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗██╔══██╗██╔════╝ ██╔════╝
# ██║  ██║███████║   ██║   ███████║    ███████╗   ██║   ██║   ██║██████╔╝███████║██║  ███╗█████╗  
# ██║  ██║██╔══██║   ██║   ██╔══██║    ╚════██║   ██║   ██║   ██║██╔══██╗██╔══██║██║   ██║██╔══╝  
# ██████╔╝██║  ██║   ██║   ██║  ██║    ███████║   ██║   ╚██████╔╝██║  ██║██║  ██║╚██████╔╝███████╗
# ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝    ╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝
# !----------------- Data Collection ------------------------                                                                                               
car_positions = []  # To store (x, y) positions
truck_positions = []  # To store (x, y) positions
truck_estimate_positions = []  # To store estimate (x, y) positions
truck_velocities = []  # To store velocity
leading_velocities = []  # To store leading vehicle velocity
truck_accelerations = []  # To store acceleration
truck_jerks = []  # To store jerk


truck_vel_mpc = []
truck_y_mpc = []  # To store ref y position
lambda_s_list = []
truck_vel_control = []
truck_y_control = []


timestamps = []  # To store timestamps for calculating acceleration and jerk
all_tightened_bounds = []  # To store all tightened bounds for visualization
Trajectory_pred = []  # To store the predicted trajectory
previous_acceleration = 0  # To help in jerk calculation

## !----------------- get initial state of leading vehicle ------------------------
car_state = get_state(car)
car_x, car_y, car_v = car_state[C_k.X_km].item(), car_state[C_k.Y_km].item(), car_state[C_k.V_km].item()
p_leading = car_x

# ! Simulation parameters
start_time = time.time()  # Record the start time of the simulation
velocity_leading = 12  # m/s

# ███████╗██╗███╗   ███╗██╗   ██╗██╗      █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
# ██╔════╝██║████╗ ████║██║   ██║██║     ██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
# ███████╗██║██╔████╔██║██║   ██║██║     ███████║   ██║   ██║██║   ██║██╔██╗ ██║
# ╚════██║██║██║╚██╔╝██║██║   ██║██║     ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
# ███████║██║██║ ╚═╝ ██║╚██████╔╝███████╗██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
# ╚══════╝╚═╝╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝                                                                                                                                         
## !----------------- start simulation!!!!!!!!!!!!!!!!!!! ------------------------
for i in range(1000):
    iteration_start = time.time()
    control_car = car_contoller.run_step(velocity_leading*3.6, 143.318146, False)
    car.apply_control(control_car)
    # get the state of the leading vehicle and truck
    car_state = get_state(car)
    # !----------------- get the state of the truck ------------------------
    truck_state = get_state(truck)
    r = np.random.normal(0.0, sigma_measurement, size=(nx, 1))
    measurement_truck = truck_state + r # add noise to the truck state
    
    # !-----------------  do extended kalman filter ------------------------
    ekf.predict(u_iter, A, B)  # predict the state
    ekf.update(measurement_truck)  # update the state
    truck_estimate = ekf.get_estimate  # get the estimated state
    P_estimate = ekf.get_covariance  # get the estimated covariance
    
    # !-----------------  store the car and truck state ------------------------
    car_x, car_y, car_v = car_state[C_k.X_km].item(), car_state[C_k.Y_km].item(), car_state[C_k.V_km].item()
    truck_x, truck_y, truck_v, truck_psi = truck_estimate[C_k.X_km], truck_estimate[C_k.Y_km],truck_estimate[C_k.V_km], truck_estimate[C_k.Psi]
    
    car_positions.append((car_x, car_y))
    truck_estimate_positions.append((float(truck_x[0]), float(truck_y[0])))
    truck_positions.append((truck_state[C_k.X_km].item(), truck_state[C_k.Y_km].item()))
    # TODO: predict the state of car, assuming the car is moving at a constant velocity
    p_leading=p_leading + (velocity_leading)*desired_interval
    #! uncomment  here  to ignore kalman filter
    # truck_x, truck_y, truck_v, truck_psi = truck_state[C_k.X_km].item(), truck_state[C_k.Y_km].item(), truck_state[C_k.V_km].item(), truck_state[C_k.Psi].item()
    # print("this is heading angle of the truck: ", truck_psi)
    # p_leading=car_x  # we can also use the car state as the leading vehicle state, more realistic
    if i % frequence == 0:
        # get the CARLA state
        x_iter = vertcat(truck_x, truck_y, truck_v, truck_psi)
        vel_diff=smooth_velocity_diff(p_leading, truck_x) # prevent the vel_diff is too small
        u_opt, x_opt, lambda_s, tightened_bound_N_IDM_list = mpc_controller.solve(x_iter, ref_trajectory, ref_control, 
                                                      p_leading, velocity_leading, vel_diff)
        x_iter=x_opt[:,1]
        print(f"the optimal state of the truck is: {x_iter}")
        # ! get the first input of the optimal input for the kalman filter
    
        
    all_tightened_bounds.append(tightened_bound_N_IDM_list)  
    Trajectory_pred.append(x_opt[:2,:]) # store the predicted trajectory
    
    #PID controller according to the x_iter of the MPC
    control_truck = local_controller.run_step(x_iter[2]*3.6, x_iter[1], False)
    truck.apply_control(control_truck)
    #-!----------------- store the state of the truck ------------------------
    truck_y_mpc.append(x_iter[1])
    truck_vel_mpc.append(x_iter[2])
    lambda_s_list.append(lambda_s)
    #-------------------------------------------------------------------------
    
    
    truck_state_ctr = get_state(truck)
    #truck state after the PID controller
    truck_x_ctr, truck_y_ctr, truck_vel_ctr, truck_psi_ctr = truck_state_ctr[C_k.X_km].item(), truck_state_ctr[C_k.Y_km].item(), truck_state_ctr[C_k.V_km].item(), truck_state_ctr[C_k.Psi].item()
    truck_vel_control.append(truck_vel_ctr)
    # Data collection inside the loop
    current_time = time.time()
    timestamps.append(current_time)
    truck_y_control.append(truck_y_ctr)
    truck_velocities.append(truck_vel_ctr)
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
    
    
# ██████╗ ██╗      ██████╗ ████████╗
# ██╔══██╗██║     ██╔═══██╗╚══██╔══╝
# ██████╔╝██║     ██║   ██║   ██║   
# ██╔═══╝ ██║     ██║   ██║   ██║   
# ██║     ███████╗╚██████╔╝   ██║   
# ╚═╝     ╚══════╝ ╚═════╝    ╚═╝   
                            
if SYNC_CTRL==True:
    gif_dir = r'C:\Users\A490243\Desktop\Master_Thesis\Figure'
    gif_name = 'CARLA_IDM_constraint_simulation_plots_with_filter.gif'
    animate_constraints(all_tightened_bounds, truck_positions, car_positions, Trajectory_pred, gif_dir,gif_name)
    figure_dir = r'C:\Users\A490243\Desktop\Master_Thesis\Figure'
    figure_name = 'CARLA_simulation_plots_with_filter.png'
    plot_and_save_simulation_data(truck_positions, timestamps, truck_velocities, truck_accelerations, truck_jerks, 
                                car_positions, leading_velocities, ref_velocity, truck_vel_mpc, truck_vel_control, 
                                figure_dir,figure_name)

    figure_dir = r'C:\Users\A490243\Desktop\Master_Thesis\Figure'
    figure_name = 'CARLA_simulation_plots_kf_state_compare.png'
    plot_kf_trajectory(truck_positions, truck_estimate_positions, figure_dir, figure_name)

    figure_dir = r'C:\Users\A490243\Desktop\Master_Thesis\Figure'
    figure_name = 'CARLA_simulation_compare_ref.png'
    plot_mpc_y_vel(truck_y_mpc, truck_vel_mpc, truck_y_control, truck_vel_control, figure_dir, figure_name)
else:
    gif_dir = r'C:\Users\A490243\Desktop\Master_Thesis\Figure'
    gif_name = 'CARLA_IDM_constraint_simulation_plots_with_filter_diffF.gif'
    animate_constraints(all_tightened_bounds, truck_positions, car_positions, Trajectory_pred, gif_dir,gif_name)
    figure_dir = r'C:\Users\A490243\Desktop\Master_Thesis\Figure'
    figure_name = 'CARLA_simulation_plots_with_filter_diffF.png'
    plot_and_save_simulation_data(truck_positions, timestamps, truck_velocities, truck_accelerations, truck_jerks, 
                                car_positions, leading_velocities, ref_velocity, truck_vel_mpc, truck_vel_control, 
                                figure_dir,figure_name)

    figure_dir = r'C:\Users\A490243\Desktop\Master_Thesis\Figure'
    figure_name = 'CARLA_simulation_plots_kf_state_compare_diffF.png'
    plot_kf_trajectory(truck_positions, truck_estimate_positions, figure_dir, figure_name)

    figure_dir = r'C:\Users\A490243\Desktop\Master_Thesis\Figure'
    figure_name = 'CARLA_simulation_compare_ref_diffF.png'
    plot_mpc_y_vel(truck_y_mpc, truck_vel_mpc, truck_y_control, truck_vel_control, figure_dir, figure_name)