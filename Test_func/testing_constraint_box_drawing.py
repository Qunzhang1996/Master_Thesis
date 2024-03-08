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
from Traffic.Traffic import Traffic

# # ------------------------change map to Town06------------------------
# import subprocess
# Command to run your script
# command = (
#     r'cd C:\Users\A490242\Desktop\Documents\WindowsNoEditor\PythonAPI\util && '
#     r'python config.py --map Town06')
# subprocess.run(command, shell=True)
# # --------------------------Run the command--------------------------

## !----------------- Carla Settings --------------------------------
N = 200
Traffic = Traffic(N=200)
# spawn the vehicle
spawned_vehicles, center_line = Traffic.setup_complex_carla_environment()
vehicle_list = spawned_vehicles
trafficList = [vehicle_list[0]] + vehicle_list[2:]
truck = vehicle_list[1]
## !----------------- Set the velocity of the vehicles ------------------------
ref_velocity = 54/3.6  # TODO: here is the reference velocity for the truck
velocities = {
        'normal': carla.Vector3D(0.9 * ref_velocity, 0, 0),
        'passive': carla.Vector3D(0.7 * ref_velocity, 0, 0),
        'aggressive': carla.Vector3D(ref_velocity, 0, 0),  # Equal to 1.0 * ref_velocity for clarity
        'reference': carla.Vector3D(ref_velocity, 0, 0)  # Specifically for the truck
    }
Traffic.set_velocity(vehicle_list, velocities)
time.sleep(1)
pred_traj = Traffic.predict_trajectory(vehicle_list)
pred_traj_traffic = Traffic.predict_trajectory(trafficList)
pred_traj_ego = pred_traj[1]
# pred_traj_traffic[0] = pred_traj[0] + pred_traj[2:]
# print(pred_traj)

          
desired_interval = 0.2  # Desired time interval in seconds                      
ref_velocity = 15  # TODO: here is the reference velocity for the truck
dt = desired_interval
versionL = {"version" : "leftChange"}
versionR = {"version" : "rightChange"}
versionT = {"version" : "trailing"}

# ! Simulation parameters
start_time = time.time()  # Record the start time of the simulation
velocity_leading = 13
Nveh = len(trafficList) # Number of vehicles in the traffic
traffic_state = np.zeros((5,N+1,Nveh)) # the size is defined as 5x(N+1)xNveh, 5: {x,y,sign,shift,flip}, N is the prediction horizon, Nveh is the number of vehicles


traffic_state[:2, :, :] = np.transpose(pred_traj_traffic,(1,2,0))
constraint_values_all = draw_constraint_box(truck, trafficList, traffic_state, pred_traj_ego, Nveh, N, versionR)

px_traj_all = pred_traj_ego[0]
laneWidth = 3.5

# Now, plotting the entire trajectory
plt.figure(figsize=(12,4))

# Plotting constraints for all i
plt.scatter(px_traj_all, constraint_values_all[:,0],label='Constraint 1')
plt.scatter(px_traj_all, constraint_values_all[:,1],label='Constraint 2 on car_2')

# Plotting positions of the vehicles
plt.scatter(pred_traj[2][0],pred_traj[2][1],label='car_1')
plt.scatter(pred_traj[0][0],pred_traj[0][1],label='car_leading')
plt.scatter( pred_traj[1][0], pred_traj[1][1],label='ego_vehicle')

# Marking lanes
plt.plot([-30, 700], [center_line, center_line], 'k--')
plt.plot([-30, 700], [center_line+laneWidth, center_line+laneWidth], 'k--')
plt.plot([-30, 700], [center_line-laneWidth, center_line-laneWidth], 'k--')
plt.plot([-30, 700], [center_line-laneWidth*3/2, center_line-laneWidth*3/2], 'b')
plt.plot([-30, 700], [center_line+laneWidth/2, center_line+laneWidth/2], 'b')
plt.plot([-30, 700], [center_line+laneWidth*3/2, center_line+laneWidth*3/2], 'b')
plt.plot([-30, 700], [center_line-laneWidth/2, center_line-laneWidth/2], 'b')

plt.xlabel('Position along X')
plt.ylabel('Constraint Value')
plt.title('Constraint Curve for the Whole Trajectory')
plt.grid(True)
plt.legend()
plt.savefig('C:\\Users\\A490242\\Desktop\\Master_Thesis\\Figure\\right_lane_constraint.png')
plt.show()



# # Define road parameters
# center_lane_x = 124 # center points of the center lane: x
# center_lane_y = 143.318146 # center points of the center lane: y
# lane_width = 3.5
# num_lanes = 3
# car_x=124
# car_y=143.318146
# car2_x = 124 - 30
# car2_y = 143.318146 + 3.5
# truck_x = 124 - 50
# truck_y = 143.318146

# # Plotting the lanes
# fig, ax = plt.subplots()

# # Plot left lane
# left_lane_y = center_lane_y + lane_width
# plt.plot([0, 248], [left_lane_y + lane_width/2, left_lane_y + lane_width/2], color='black', linestyle='-')

# # Plot center lane
# plt.plot([0, 248], [center_lane_y+ lane_width/2, center_lane_y + lane_width/2], color='black', linestyle='dashed')
# plt.plot([0, 248], [center_lane_y - lane_width/2, center_lane_y - lane_width/2], color='black', linestyle='dashed')

# # Plot right lane
# right_lane_y = center_lane_y - lane_width
# plt.plot([0, 248], [right_lane_y - lane_width/2, right_lane_y - lane_width/2], color='black', linestyle='-')

# # Set y limits
# ax.set_ylim(top=right_lane_y - 2*lane_width, bottom=left_lane_y + 2*lane_width)

# # Hide x-axis
# ax.xaxis.set_visible(False)

# plt.scatter(car_x, car_y, color='red', marker='o')
# plt.scatter(car2_x, car2_y, color='blue', marker='o')
# plt.scatter(truck_x, truck_y, color='green', marker='o')


# plt.show()

