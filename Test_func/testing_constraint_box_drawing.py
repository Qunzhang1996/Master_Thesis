from casadi import *
import carla
import numpy as np
import matplotlib.pyplot as pltw
from matplotlib.lines import Line2D
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
# import helpers
from util.utils import *

# # ------------------------change map to Town06------------------------
# import subprocess
# # Command to run your script
# command = (
#     r'cd C:\Users\A490242\Desktop\Documents\WindowsNoEditor\PythonAPI\util && '
#     r'python config.py --map Town06')
# subprocess.run(command, shell=True)
# # --------------------------Run the command--------------------------

## !----------------- Carla Settings --------------------------------
car,truck,car2 = setup_carla_environment(Sameline_ACC=False)
time.sleep(1)
velocity1 = carla.Vector3D(10, 0, 0)
velocity2 = carla.Vector3D(8, 0, 0)
# add traffic vehicles
car.set_target_velocity(velocity2) # set the velocity of the car1 slower than the truck
car2.set_target_velocity(velocity2) 
# create a list of traffic vehicles
trafficList = [car,car2]
# add ego vehicle
truck.set_target_velocity(velocity1)
time.sleep(1)
client = carla.Client('localhost', 2000)
world = client.get_world()
carla_map = world.get_map()
          
desired_interval = 0.2  # Desired time interval in seconds                      
ref_velocity = 15  # TODO: here is the reference velocity for the truck
dt = desired_interval
N=12
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

# ------------------ Problem definition ---------------------
scenarioTrailADV = trailing(vehicleADV,N,lanes = 3, laneWidth=3.5)
scenarioOvertakeADV = simpleOvertake(vehicleADV,N,lanes = 3, laneWidth=3.5)
versionL = {"version" : "leftChange"}
versionR = {"version" : "rightChange"}
versionT = {"version" : "trailing"}

## !----------------- get initial state of leading vehicle ------------------------
car_state = get_state(car)
car_x, car_y, car_v = car_state[C_k.X_km].item(), car_state[C_k.Y_km].item(), car_state[C_k.V_km].item()
p_leading = car_x

# ! Simulation parameters
start_time = time.time()  # Record the start time of the simulation
velocity_leading = 13
Nveh = len(trafficList) # Number of vehicles in the traffic
traffic_state = np.zeros((5,N+1,Nveh)) # the size is defined as 5x(N+1)xNveh, 5: {x,y,sign,shift,flip}, N is the prediction horizon, Nveh is the number of vehicles
traffic_state = get_traffic_state(trafficList, Nveh, N, dt)
func1, func2 = draw_constraint_box(truck, trafficList, traffic_state, Nveh, N, versionR)


# Compute the sum of the two density matrices
sum_output = func1 + func2

# Define road parameters
center_lane_x = 124 # center points of the center lane: x
center_lane_y = 143.318146 # center points of the center lane: y
lane_width = 3.5
num_lanes = 3

# Plotting the lanes
fig, ax = plt.subplots()

# Plot left lane
left_lane_y = center_lane_y + lane_width
plt.plot([0, 248], [left_lane_y + lane_width/2, left_lane_y + lane_width/2], color='black', linestyle='-')

# Plot center lane
plt.plot([0, 248], [center_lane_y+ lane_width/2, center_lane_y + lane_width/2], color='black', linestyle='dashed')
plt.plot([0, 248], [center_lane_y - lane_width/2, center_lane_y - lane_width/2], color='black', linestyle='dashed')

# Plot right lane
right_lane_y = center_lane_y - lane_width
plt.plot([0, 248], [right_lane_y - lane_width/2, right_lane_y - lane_width/2], color='black', linestyle='-')

# Set y limits
ax.set_ylim(top=right_lane_y - 2*lane_width, bottom=left_lane_y + 2*lane_width)

# Hide x-axis
ax.xaxis.set_visible(False)
car_x=124
car_y=143.318146
car2_x = 124 - 30
car2_y = 143.318146 + 3.5
truck_x = 124 - 50
truck_y = 143.318146
plt.scatter(car_x, car_y, color='red', marker='o')
plt.scatter(car2_x, car2_y, color='blue', marker='o')
plt.scatter(truck_x, truck_y, color='green', marker='o')

plt.show()


