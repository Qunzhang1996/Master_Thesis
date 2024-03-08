from casadi import *
import carla
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import os
import json
sys.path.append(r'C:\Users\A490242\Desktop\Documents\WindowsNoEditor\PythonAPI\carla')
sys.path.append(r'C:\Users\A490242\Desktop\Master_Thesis')
# import observor
from kalman_filter.kalman_filter import kalman_filter
# import controller
from Controller.LTI_MPC import MPC
from vehicleModel.vehicle_model import car_VehicleModel
from agents.navigation.controller_upd import VehiclePIDController
from Traffic.Traffic_qun import Traffic
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
N=200
Traffic = Traffic(N)
# spawn the vehicle
spawned_vehicles, center_line = Traffic.setup_complex_carla_environment()
vehicle_list = spawned_vehicles
## !----------------- Set the velocity of the vehicles ------------------------
ref_velocity = 54/3.6  # TODO: here is the reference velocity for the truck
velocities = {
        'normal': carla.Vector3D(0.9 * ref_velocity, 0, 0),
        'passive': carla.Vector3D(0.7 * ref_velocity, 0, 0),
        'aggressive': carla.Vector3D(ref_velocity-5, 0, 0),  # Equal to 1.0 * ref_velocity for clarity
        'reference': carla.Vector3D(ref_velocity, 0, 0)  # Specifically for the truck
    }
Traffic.set_velocity(velocities)
time.sleep(1)
pred_traj = Traffic.predict_trajectory()
print(Traffic.get_velocity())

versionL = {"version" : "leftChange"}
versionR = {"version" : "rightChange"}
versionT = {"version" : "trailing"}


leadLength = 6
v0_i = 15
leadWidth = 1.7
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

    def constraint(self, px):
        func1 = self.traffic_sign * (self.traffic_sign*(self.traffic_y-self.traffic_shift)/2 + self.egoWidth + leadWidth) / 2 * \
                    tanh(px - self.traffic_x + leadLength/2 + self.L_tract + v0_i * self.Time_headway + self.min_distx )  + self.traffic_shift/2 
        func2 = self.traffic_sign * (self.traffic_sign*(self.traffic_y-self.traffic_shift)/2 + self.egoWidth + leadWidth) / 2 * \
                    tanh( - (px - self.traffic_x)  + leadLength/2 + self.L_trail + v0_i * self.Time_headway+ self.min_distx )  + self.traffic_shift/2

        return func1 + func2 + 143.318146 -laneWidth/2


px_traj_all = []
constraint_values_all = []
constraint_values_2_all = []
traffic_x_all = []
traffic_y_all = []

for i in range(N+1):  # Loop through i = 0 to 12
    traffic_x = pred_traj[0,i,3]
    traffic_x_2 = pred_traj[0,i,0]
    px_traj = pred_traj[0,i,1]
    
    
    # Create instances of the test class for each scenario
    my_test = test(traffic_x, traffic_y=-laneWidth/2, traffic_shift=-laneWidth, traffic_sign=1)    #! ('vehicle.carlamotors.carlacola', 20, -3.5),  right lane
    my_test_2 = test(traffic_x_2, traffic_y=laneWidth/2, traffic_shift=laneWidth, traffic_sign=-1)   #!('vehicle.tesla.model3', 80),   center line
    
    
    
    # Collecting the data for plotting
    px_traj_all.append(px_traj)
    constraint_values_all.append(my_test.constraint(px_traj))
    constraint_values_2_all.append(my_test_2.constraint(px_traj))

print(len(pred_traj[1][0]))

# Now, plotting the entire trajectory
plt.figure(figsize=(12,4))

# Plotting constraints for all i
plt.scatter(px_traj_all, constraint_values_all,label='Constraint 1')
plt.scatter(px_traj_all, constraint_values_2_all, label='Constraint 2')
# Plotting positions of the vehicles
plt.scatter(pred_traj[0,:,3],pred_traj[1,:,3],label='car_4')
plt.scatter(pred_traj[0,:,0],pred_traj[1,:,0],label='car_leading')
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
plt.savefig('C:\\Users\\A490242\\Desktop\\Master_Thesis\\Figure\\right_lane_constraint.png')
plt.show()