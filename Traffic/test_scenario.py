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
from Controller.LC_MPC import LC_MPC
from vehicleModel.vehicle_model import car_VehicleModel
from agents.navigation.controller import VehiclePIDController
from Traffic import Traffic
from util.utils import *
from Scenarios import trailing, simpleOvertake
N=12
dt = 0.2
Traffic = Traffic(N)
vehicleADV = car_VehicleModel(dt,N, width = 2.5, length =6 )
Trailing = trailing(vehicleADV,N)