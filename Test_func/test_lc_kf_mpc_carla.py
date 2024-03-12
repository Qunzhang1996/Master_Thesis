from casadi import *
import carla
import numpy as np
import matplotlib.pyplot as pltw
import sys
import time
import os
import json
sys.path.append(r'C:\Users\A490242\Desktop\Documents\WindowsNoEditor\PythonAPI\carla')
sys.path.append(r'C:\Users\A490242\Desktop\Master_Thesis')
# import observor
from kalman_filter.kalman_filter import kalman_filter
# import controller
from Controller.LC_MPC import LC_MPC
from vehicleModel.vehicle_model import car_VehicleModel
from agents.navigation.controller_upd import VehiclePIDController
from Controller.scenarios import trailing, simpleOvertake
from Controller.scenarios_lc import simpleLaneChange
# import helpers
from util.utils import *
# from Traffic.Traffic import Traffic
from Traffic.Traffic_qun import Traffic

# # ------------------------change map to Town06------------------------
# import subprocess
# # Command to run your script
# command = (
#     r'cd C:\Users\A490242\Desktop\Documents\WindowsNoEditor\PythonAPI\util && '
#     r'python config.py --map Town06')
# subprocess.run(command, shell=True)
# # --------------------------Run the command--------------------------

## !----------------- Carla Settings --------------------------------
N = 12
Traffic = Traffic(N)
# spawn the vehicle
spawned_vehicles, center_line = Traffic.setup_complex_carla_environment()
vehicle_list = spawned_vehicles
trafficList = [vehicle_list[0]] + vehicle_list[2:]
Ntraffic = len(trafficList)
truck = vehicle_list[1]
## !----------------- Set the velocity of the vehicles ------------------------
ref_velocity = 54/3.6  # TODO: here is the reference velocity for the truck
velocities = {
        'normal': carla.Vector3D(0.9 * ref_velocity, 0, 0),
        'passive': carla.Vector3D(0.7 * ref_velocity, 0, 0),
        'aggressive': carla.Vector3D(ref_velocity, 0, 0),  # Equal to 1.0 * ref_velocity for clarity
        'reference': carla.Vector3D(ref_velocity, 0, 0)  # Specifically for the truck
    }
Traffic.set_velocity(velocities)
time.sleep(1)
pred_traj = Traffic.predict_trajectory()
pred_traj_ego = pred_traj[:,:,1]
# pred_traj_traffic = [pred_traj[:,:,0]] + pred_traj[:,:,2:]
pred_traj_traffic = pred_traj[:, :, [0] + list(range(2, len(trafficList)+1))] # predicted trajectory of the traffic vehicles
traffic_state = pred_traj_traffic

# ██████╗ ██╗██████╗ 
# ██╔══██╗██║██╔══██╗
# ██████╔╝██║██║  ██║
# ██╔═══╝ ██║██║  ██║
# ██║     ██║██████╔╝
# ╚═╝     ╚═╝╚═════╝ 
## !----------------- PID Controller Settings ------------------------        
car = vehicle_list[0]
car2 = vehicle_list[2]
truck = vehicle_list[1]           
desired_interval = 0.2  # Desired time interval in seconds
car_contoller = VehiclePIDController(car, 
                                     args_lateral = {'K_P': 1.1, 'K_I': 0.2, 'K_D': 0.02, 'dt': desired_interval}, 
                                     args_longitudinal = {'K_P': 0.950, 'K_I': 0.1, 'K_D': 0.02, 'dt': desired_interval})
car_contoller2 = VehiclePIDController(car2, 
                                     args_lateral = {'K_P': 1.1, 'K_I': 0.2, 'K_D': 0.02, 'dt': desired_interval}, 
                                     args_longitudinal = {'K_P': 0.950, 'K_I': 0.1, 'K_D': 0.02, 'dt': desired_interval})
local_controller = VehiclePIDController(truck, 
                                        args_lateral = {'K_P': 1.5, 'K_I': 0.05, 'K_D': 0.2, 'dt': desired_interval/10}, 
                                        args_longitudinal = {'K_P': 1.95, 'K_I': 0.5, 'K_D': 0.2, 'dt': desired_interval/10})
# ███╗   ███╗██████╗  ██████╗
# ████╗ ████║██╔══██╗██╔════╝
# ██╔████╔██║██████╔╝██║     
# ██║╚██╔╝██║██╔═══╝ ██║     
# ██║ ╚═╝ ██║██║     ╚██████╗
# ╚═╝     ╚═╝╚═╝      ╚═════╝
## !----------------- Robust MPC Controller Settings ------------------------                           
ref_velocity = 15  # TODO: here is the reference velocity for the truck
dt = desired_interval
# create a model for the ego vehicle
vehicleADV = car_VehicleModel(dt,N, width = 2, length = 6)
nx,nu,nrefx,nrefu = vehicleADV.getSystemDim()
int_opt = 'rk'
vehicleADV.integrator(int_opt,dt)
F_x_ADV  = vehicleADV.getIntegrator()
vx_init_ego = 10
vehicleADV.setInit([20,143.318146],vx_init_ego)
Q_ADV = [0,5e2,50,10]                            # State cost, Entries in diagonal matrix
R_ADV = [5,5]                                    # Input cost, Entries in diagonal matrix
vehicleADV.cost(Q_ADV,R_ADV)
vehicleADV.costf(Q_ADV)
L_ADV,Lf_ADV = vehicleADV.getCost()
P0, process_noise, possibility = set_stochastic_mpc_params()

# ------------------ Problem definition ---------------------
scenarioTrailADV = trailing(vehicleADV,N,lanes = 3, laneWidth=3.5)
# scenarioOvertakeADV = simpleOvertake(vehicleADV,N,lanes = 3, laneWidth=3.5)
scenarioOvertakeADV = simpleLaneChange(N, Ntraffic, lanes = 3, laneWidth=3.5)
versionL = {"version" : "leftChange"}
versionR = {"version" : "rightChange"}
versionT = {"version" : "trailing"}

# mpc controller for right lane change
mpc_controllerR = LC_MPC(vehicleADV, trafficList, np.diag(Q_ADV), np.diag(R_ADV), P0, process_noise, possibility, N, versionR, scenarioOvertakeADV)
mpc_controllerR.setController()
# changeRight = mpc_controllerR.getFunction() # this function is not working. why?

## !----------------- get initial state and set ref for the ccontroller ------------------------
                                                                                  
x_iterR = DM(int(nx),1)
x_iterR[:],u_iter = vehicleADV.getInit()
print(f"initial state of the truck is: {x_iterR}")
print(f"initial input of the truck is: {u_iter}")

# reference trajectory and control for trailing
# ref_trajectory = np.zeros((nx, N + 1)) # Reference trajectory (states)
# ref_trajectory_R = np.zeros((nx, N + 1)) # Reference trajectory (states)

ref_trajectory, ref_control = create_reference_trajectory(nx, nu, N, ref_velocity, versionT)
ref_trajectory_R, ref_control_R = create_reference_trajectory(nx, nu, N, ref_velocity, versionR)
ref_trajectory_L, ref_control_L = create_reference_trajectory(nx, nu, N, ref_velocity, versionL)




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
Q_0[1,1]=0.001  # y bound is [0, 0.1]
Q_0[2,2]=1.8/6*2  # v bound is [0, 1.8]
Q_0[3,3]=0.001  # psi bound is [0, 0.05]
R_0=np.eye(nx)*sigma_measurement**2
r = np.random.normal(0.0, sigma_measurement, size=(nx, 1))
# set the initial state and control input
x_0 = x_iterR
P_kf=np.eye(nx)*(sigma_process)**4  # initial state covariance
u_iter = np.array([0,0])
# get system dynamic matrices
A,B,_=mpc_controllerR.get_dynammic_model()
H=np.eye(nx)   #measurement matrix, y=H@x_iterR
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
truck_velocities = []  # To store velocity
leading_velocities = []  # To store leading vehicle velocity
truck_accelerations = []  # To store acceleration
truck_jerks = []  # To store jerk
truck_vel_mpc = []
lambda_s_list = []
truck_vel_control = []
timestamps = []  # To store timestamps for calculating acceleration and jerk
# all_tightened_bounds = []  # To store all tightened bounds for visualization
Trajectory_pred = []  # To store the predicted trajectory
previous_acceleration = 0  # To help in jerk calculation

## !----------------- get initial state of leading vehicle ------------------------
car_state = get_state(car)
car_x, car_y, car_v = car_state[C_k.X_km].item(), car_state[C_k.Y_km].item(), car_state[C_k.V_km].item()
p_leading = car_x

# ! Simulation parameters
start_time = time.time()  # Record the start time of the simulation
velocity_leading = 13
Nveh = len(trafficList) # Number of vehicles in the traffic

# ███████╗██╗███╗   ███╗██╗   ██╗██╗      █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
# ██╔════╝██║████╗ ████║██║   ██║██║     ██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
# ███████╗██║██╔████╔██║██║   ██║██║     ███████║   ██║   ██║██║   ██║██╔██╗ ██║
# ╚════██║██║██║╚██╔╝██║██║   ██║██║     ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
# ███████║██║██║ ╚═╝ ██║╚██████╔╝███████╗██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
# ╚══════╝╚═╝╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝                                                                                                                                         
## !----------------- start simulation!!!!!!!!!!!!!!!!!!! ------------------------
for i in range(500):
    iteration_start = time.time()
    control_car = car_contoller.run_step(velocity_leading*3.6, 143.318146, False)
    car.apply_control(control_car)

    # get the state of the leading vehicle and truck
    car_state = get_state(car)
    # !----------------- get the state of the truck ------------------------
    truck_state = get_state(truck)
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
    truck_positions.append((truck_state[C_k.X_km].item(), truck_state[C_k.Y_km].item()))
    p_leading=p_leading + (velocity_leading)*desired_interval


    if i%1==0: # get the CARLA state every 10 steps
        # get the CARLA state
        x_iterR = vertcat(truck_x, truck_y, truck_v, truck_psi)
        vel_diff=smooth_velocity_diff(p_leading, truck_x) # prevent the vel_diff is too small
        # !-----------------  solve the MPC problem ------------------------
        u_optR, x_optR = mpc_controllerR.solve(x_iterR, ref_trajectory_R, ref_control_R, 
                                                      p_leading, traffic_state, velocity_leading, vel_diff)
        x_iterR=x_optR[:,1]
        
        
    # all_tightened_bounds.append(tightened_bound_N_IDM_list)  
    Trajectory_pred.append(x_optR[:2,:]) # store the predicted trajectory
    
    #PID controller according to the x_iterR of the MPC for 
    control_truck = local_controller.run_step(x_iterR[2]*3.6, x_iterR[1], False)
    print(f"the y position of the truck is: {x_iterR[1]}")
    print(f"the velocity of the truck is: {x_iterR[2]}")
    truck.apply_control(control_truck)
    
    truck_vel_mpc.append(x_iterR[2])
    # lambda_s_list.append(lambda_s)
    
    
    truck_state_ctr = get_state(truck)
    # truck_vel_ctr=truck_state_ctr[C_k.V_km].item()
    #truck state after the PID controller
    truck_x_ctr, truck_y_ctr, truck_vel_ctr, truck_psi_ctr = truck_state_ctr[C_k.X_km].item(), truck_state_ctr[C_k.Y_km].item(), truck_state_ctr[C_k.V_km].item(), truck_state_ctr[C_k.Psi].item()
    truck_vel_control.append(truck_vel_ctr)
    # Data collection inside the loop
    current_time = time.time()
    timestamps.append(current_time)
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
    if i == 200: break
    
# print(Trajectory_pred)

gif_dir = r'C:\Users\A490242\Desktop\Master_Thesis\Figure'
gif_name = 'IDM_constraint_simulation_plots_with_filter.gif'
# animate_constraints(all_tightened_bounds, truck_positions, car_positions, Trajectory_pred, gif_dir,gif_name)
figure_dir = r'C:\Users\A490242\Desktop\Master_Thesis\Figure'
figure_name = 'simulation_plots_with_filter.png'
plot_and_save_simulation_data(truck_positions, timestamps, truck_velocities, truck_accelerations, truck_jerks, 
                              car_positions, leading_velocities, ref_velocity, truck_vel_mpc, truck_vel_control, 
                              figure_dir,figure_name)