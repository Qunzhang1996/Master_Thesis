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
Traffic = Traffic(N=30)
# spawn the vehicle
spawned_vehicles, center_line = Traffic.setup_complex_carla_environment()
vehicle_list = spawned_vehicles
## !----------------- Set the velocity of the vehicles ------------------------
ref_velocity = 54/3.6  # TODO: here is the reference velocity for the truck
velocities = {
        'normal': carla.Vector3D(0.9 * ref_velocity, 0, 0),
        'passive': carla.Vector3D(0.7 * ref_velocity, 0, 0),
        'aggressive': carla.Vector3D(ref_velocity, 0, 0),  # Equal to 1.0 * ref_velocity for clarity
        'reference': carla.Vector3D(ref_velocity, 0, 0)  # Specifically for the truck
    }
Traffic.set_velocity(vehicle_list,velocities)
time.sleep(1)
pred_traj = Traffic.predict_trajectory(vehicle_list)
# print(pred_traj)


leadLength = 6
N = 12
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
        self.L_tract=16.1544/3
        self.L_trail=16.1544-self.L_tract
        self.egoWidth = 2.54

    def constraint(self, px):
        func1 = self.traffic_sign * (self.traffic_sign*(self.traffic_y-self.traffic_shift)/2 + self.egoWidth + leadWidth) / 2 * \
                    tanh(px - self.traffic_x + leadLength/2 + self.L_tract + v0_i * self.Time_headway + self.min_distx )  + self.traffic_shift/2 + 143.318146 /2
        func2 = self.traffic_sign * (self.traffic_sign*(self.traffic_y-self.traffic_shift)/2 + self.egoWidth + leadWidth) / 2 * \
                    tanh( - (px - self.traffic_x)  + leadLength/2 + self.L_trail + v0_i * self.Time_headway+ self.min_distx )  + self.traffic_shift/2+ 143.318146 /2
        # print(func1 + func2)
        # exit()
        return func1 + func2

# Create instances for traffic_x, traffic_y, and traffic_shift

traffic_x = np.array([10*i for i in range(30)]).reshape(-1,1)
traffic_x = pred_traj[3][0]

traffic_x_2 = np.array([10*i+60 for i in range(30)]).reshape(-1,1)
traffic_x_2 = pred_traj[0][0]


traffic_shift =  -laneWidth 
# traffic_shift = 0*traffic_shift
# Create an instance of the test class

my_test = test(traffic_x, traffic_y=-laneWidth/2, traffic_shift=-laneWidth, traffic_sign=1)
my_test_2 = test(traffic_x_2, traffic_y=laneWidth/2, traffic_shift=laneWidth, traffic_sign=-1)

# Time points at which to evaluate the constraint
time_points = np.linspace(-30, 100, 100)  # Example time points
# px = pred_traj[1][0
px_traj = pred_traj[1][0]

# Evaluate the constraint function at each time point

constraint_values = [float(my_test.constraint(px)[0]) for px in pred_traj[1][0]]
constraint_values_2 = [float(my_test_2.constraint(px)[0]) for px in pred_traj[1][0]]
# Plot the constraint curve
plt.figure(figsize=(12,4))
plt.plot(constraint_values,'r', linewidth=2)
plt.plot(constraint_values_2,'r', linewidth=2)
plt.plot([-30,100],[center_line,center_line],'k--')
plt.plot([-30,100],[center_line+laneWidth,center_line+laneWidth],'k--')
plt.plot([-30,100],[center_line-laneWidth,center_line-laneWidth],'k--')
plt.plot([-30,100],[center_line+2*laneWidth,center_line+2*laneWidth],'k--')
plt.plot([-30,100],[center_line+laneWidth/2,center_line+laneWidth/2],'b--')
plt.plot([-30,100],[center_line-laneWidth/2,center_line-laneWidth/2],'b--')
plt.xlabel('Time')
plt.ylabel('Constraint Value')
plt.title('Constraint Curve')
plt.grid(True)
plt.show()
