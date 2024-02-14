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

# # ----------------- Carla Settings ------------------------
car,truck = setup_carla_environment(Sameline_ACC=True)
time.sleep(1)
velocity1 = carla.Vector3D(10, 0, 0)
velocity2 = carla.Vector3D(10, 0, 0)
car.set_target_velocity(velocity1)
truck.set_target_velocity(velocity2)
time.sleep(1)
client = carla.Client('localhost', 2000)
world = client.get_world()
carla_map = world.get_map()

## ----------------- PID Controller Settings ------------------------
desired_interval = 0.2  # Desired time interval in seconds
car_contoller = VehiclePIDController(car, 
                                     args_lateral = {'K_P': 2, 'K_I': 0.2, 'K_D': 0.02, 'dt': desired_interval}, 
                                     args_longitudinal = {'K_P': 0.950, 'K_I': 0.1, 'K_D': 0.02, 'dt': desired_interval})
local_controller = VehiclePIDController(truck, 
                                        args_lateral = {'K_P': 2, 'K_I': 0.2, 'K_D': 0.01, 'dt': desired_interval}, 
                                        args_longitudinal = {'K_P': 1.1, 'K_I': 0.2, 'K_D': 0.02, 'dt': desired_interval})

## ----------------- Robust MPC Controller Settings ------------------------
ref_velocity = 15
dt = desired_interval
N=12