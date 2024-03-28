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
from Traffic.Traffic import Traffic
# import helpers
from util.utils import *

# #! ------------------------change map to Town06------------------------
# import subprocess
# # Command to run your script
# command = (
#     r'cd C:\Users\A490243\CARLA\CARLA_Latest\WindowsNoEditor\PythonAPI\util && '
#     r'python config.py --map Town06')
# subprocess.run(command, shell=True)
# exit()
# #! --------------------------Run the command--------------------------
N=400
Traffic = Traffic(N, 0.2)
# spawn the vehicle
spawned_vehicles, center_line = Traffic.setup_complex_carla_environment()
vehicle_list = spawned_vehicles
## !----------------- Set the velocity of the vehicles ------------------------
ref_velocity = 54/3.6  # TODO: here is the reference velocity for the truck
ref_vx = 54/3.6  
velocities = {
        'normal': carla.Vector3D(0.65 * ref_vx, 0, 0),
        'passive': carla.Vector3D(0.65 * ref_vx, 0, 0),
        'aggressive': carla.Vector3D(1.1*ref_vx, 0, 0),  # Equal to 1.0 * ref_velocity for clarity
        'reference': carla.Vector3D(ref_vx, 0, 0)  # Specifically for the truck
    }
Traffic.set_velocity(velocities)
time.sleep(1)
pred_traj = Traffic.prediction()
print(Traffic.get_velocity())


leadLength = 6
leadWidth = 1.9
laneWidth = 3.5


class test:
    def __init__(self, traffic_x, traffic_y, traffic_shift,traffic_sign, min_distx=5) -> None:
        self.traffic_sign = traffic_sign

        
        self.traffic_x = traffic_x
        self.traffic_y = traffic_y
        self.traffic_shift = traffic_shift
        
        
        
        self.Time_headway = 0.5
        self.min_distx = min_distx
        self.L_tract=6
        self.L_trail=0
        self.egoWidth = 2.54
        self.init_bound  = 143.318146-laneWidth/2

    def constraint(self, px):
        
        #! In this situation, we do not have egoTheta_max. no trailor

            
        # #! avoid the ego vehicle itself
        # if i == 1 : continue
        # Get Vehicle Properties
        v0_i =0.75 * ref_vx
        # traffic.getVehicles
        l_front,l_rear = 5.5/2, 5.5/2
        leadWidth= 2.54

        # Define vehicle specific constants
        alpha_0 = self.traffic_sign * (self.traffic_sign*(self.traffic_y-self.traffic_shift)+leadWidth/2)
        alpha_1 = l_rear + self.L_tract + v0_i * self.Time_headway + self.min_distx 
        alpha_2 = l_front + self.L_trail + v0_i * self.Time_headway+ self.min_distx 
        alpha_3 = self.traffic_shift
        d_w_e = (self.egoWidth/2)*self.traffic_sign
        # Construct function
        func1 = alpha_0 / 2 * tanh(px - self.traffic_x + alpha_1)+alpha_3/2
        func2 = alpha_0 / 2 * tanh(self.traffic_x - px + alpha_2)+alpha_3/2
        # !SHIFT ACCORDING TO THE INIT_BOUND
        S = func1 + func2 + self.init_bound + d_w_e
        return S


px_traj_all = []
constraint_values_all = []
constraint_values_2_all = []
traffic_x_all = []
traffic_y_all = []

    
    
# here, test using vector of traffic_x and traffic_y to calculate the constraint
"""
Traffic_y should be scaled to -laneWidth/2, laneWidth/2, laneWidth*3/2,
"""
my_test = test(DM(pred_traj[0,:,2]), DM(pred_traj[1,:,2])-(143.318146-laneWidth/2), laneWidth/2, 1)    #! ('vehicle.carlamotors.carlacola', 20, -3.5),  right lane
my_test_2 = test(DM(pred_traj[0,:,0]), DM(pred_traj[1,:,0])-(143.318146-laneWidth/2), 0.5*laneWidth, -1)   #!('vehicle.tesla.model3', 40, self.laneWidth),   left lane
px_traj = pred_traj[0,:,1]
px_traj_all = px_traj
constraint_values_all = my_test.constraint(px_traj).full().flatten()
constraint_values_2_all = my_test_2.constraint(px_traj).full().flatten()
print(DM(pred_traj[1,:,0])-(143.318146-3.5/2))
# print(constraint_values_all.shape)
# print(px_traj_all.shape)
# exit()
    

# print(len(pred_traj[1][0]))

# Now, plotting the entire trajectory
plt.figure(figsize=(12,4))

# Plotting constraints for all i
plt.scatter(px_traj_all, constraint_values_all,label='Constraint left')
plt.scatter(px_traj_all, constraint_values_2_all, label='Constraint right')
# Plotting positions of the vehicles
plt.scatter(pred_traj[0,:,2],pred_traj[1,:,2],label='car_left')
plt.scatter(pred_traj[0,:,0],pred_traj[1,:,0],label='car_right')
plt.scatter( pred_traj[0,:,1], pred_traj[1,:,1],label='ego_vehicle',alpha=0.5)
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
plt.legend(loc='upper right')
plt.savefig('C:\\Users\\A490243\\Desktop\\Master_Thesis\\Figure\\right_lane_constraint.png')
plt.show()

