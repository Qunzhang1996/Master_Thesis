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
from Controller.LC_MPC import LC_MPC
from vehicleModel.vehicle_model import car_VehicleModel
from agents.navigation.controller import VehiclePIDController
from Traffic.Traffic import Traffic
# import helpers
from util.utils import *
# ██████╗  █████╗ ███████╗███████╗
# ██╔══██╗██╔══██╗██╔════╝██╔════╝
# ██████╔╝███████║███████╗███████╗
# ██╔═══╝ ██╔══██║╚════██║╚════██║
# ██║     ██║  ██║███████║███████║
# ╚═╝     ╚═╝  ╚═╝╚══════╝╚══════╝
# #! ------------------------change map to Town06------------------------
# import subprocess
# # Command to run your script
# command = (
#     r'cd C:\Users\A490243\CARLA\CARLA_Latest\WindowsNoEditor\PythonAPI\util && '
#     r'python config.py --map Town06')
# subprocess.run(command, shell=True)
# exit()
# #! --------------------------Run the command--------------------------
N=12
desired_interval=0.2
Traffic = Traffic(N)
# spawn the vehicle
spawned_vehicles, center_line = Traffic.setup_complex_carla_environment()
vehicle_list = spawned_vehicles   # ! vehicle_list[1] is the ego vehicle
## !----------------- Set the velocity of the vehicles ------------------------
ref_velocity = 54/3.6  # TODO: here is the reference velocity for the truck
velocities = {
        'normal': carla.Vector3D(0.9 * ref_velocity, 0, 0),
        'passive': carla.Vector3D(0.7 * ref_velocity, 0, 0),
        'aggressive': carla.Vector3D(ref_velocity-5, 0, 0),  # Equal to 1.0 * ref_velocity for clarity
        'reference': carla.Vector3D(ref_velocity, 0, 0)  # Specifically for the truck
    }
Traffic.set_velocity(velocities)
## !----------------- Robust MPC Controller Settings ------------------------   
center_line = 143.318146                        
ref_velocity = 15  # TODO: here is the reference velocity for the truck
dt = desired_interval
N=12
vehicleADV = car_VehicleModel(dt,N, width = 2, length = 6)
nx,nu,nrefx,nrefu = vehicleADV.getSystemDim()
int_opt = 'rk'
vehicleADV.integrator(int_opt,dt)
F_x_ADV  = vehicleADV.getIntegrator()
vx_init_ego = 15
vehicleADV.setInit([20,center_line],vx_init_ego)
Q_ADV = [0,40,3e2,5]                            # State cost, Entries in diagonal matrix
R_ADV = [5,5]                                    # Input cost, Entries in diagonal matrix
vehicleADV.cost(Q_ADV,R_ADV)
vehicleADV.costf(Q_ADV)
L_ADV,Lf_ADV = vehicleADV.getCost()
P0, process_noise, possibility = set_stochastic_mpc_params()
mpc_controller = LC_MPC(vehicleADV, Traffic, np.diag(Q_ADV), np.diag(R_ADV), P0, process_noise, possibility, N)
#! get the ref_X
refxT_in, refxL_in, refxR_in = vehicleADV.setReferences()
print(refxT_in, refxL_in, refxR_in)