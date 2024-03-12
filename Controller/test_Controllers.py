import sys
path_to_add='C:\\Users\\A490243\\Desktop\\Master_Thesis'
sys.path.append(path_to_add)
from Autonomous_Truck_Sim.helpers import *
from Controller.MPC_tighten_bound import MPC_tighten_bound
from Controller.Controllers import makeController
from vehicleModel.vehicle_model import car_VehicleModel
from Traffic.Traffic import Traffic
from Traffic.Scenarios import trailing, simpleOvertake
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

# System initialization 
dt = 0.2                    # Simulation time step (Impacts traffic model accuracy)
f_controller = 1            # Controller update frequency, i.e updates at each t = dt*f_controller
N =  12                    # MPC Horizon length
laneWidth = 3.5
ref_vx = 54/3.6             # Higway speed limit in (m/s)
q_traffic_slack = 1e4
traffic = Traffic()
velocities = {
        'normal': carla.Vector3D(0.9 * ref_vx, 0, 0),
        'passive': carla.Vector3D(0.7 * ref_vx, 0, 0),
        'aggressive': carla.Vector3D(ref_vx, 0, 0),  # Equal to 1.0 * ref_velocity for clarity
        'reference': carla.Vector3D(ref_vx, 0, 0)  # Specifically for the truck
    }
spawned_vehicles, center_line = traffic.setup_complex_carla_environment()
traffic.set_velocity(velocities)

vehicleADV = car_VehicleModel(dt,N)
vehWidth,vehLength,L_tract,L_trail = vehicleADV.getSize()
nx,nu,nrefx,nrefu = vehicleADV.getSystemDim()
# Set Cost parameters
Q_ADV = [0,40,3e2,5]                            # State cost, Entries in diagonal matrix
R_ADV = [5,5]                                    # Input cost, Entries in diagonal matrix
q_ADV_decision = 50
vehicleADV.cost(Q_ADV,R_ADV)
vehicleADV.costf(Q_ADV)
L_ADV,Lf_ADV = vehicleADV.getCost()
# ------------------ Problem definition ---------------------
testWidth = 3.75
scenarioTrailADV = trailing(vehicleADV,N,lanes = 3,v_legal = ref_vx, laneWidth=laneWidth)
roadMin, roadMax, laneCenters = scenarioTrailADV.getRoad()
# -------------------- Traffic Set up -----------------------
# * Be carful not to initilize an unfeasible scenario where a collsion can not be avoided
# # Initilize ego vehicle
vx_init_ego = 10                                # Initial velocity of the ego vehicle
vehicleADV.setRoad(roadMin,roadMax,laneCenters)
vehicleADV.setInit([0,laneCenters[0]],vx_init_ego)


opts1 = {"version" : "trailing", "solver": "ipopt", "integrator":"LTI"}

scenarioTrailADV.slackCost(q_traffic_slack)
MPC_trailing= makeController(vehicleADV,traffic,scenarioTrailADV,N,opts1,dt)
MPC_trailing.setController()
testPred = traffic.prediction()
print("this is testPred",testPred)
print("this is testPred size",testPred.shape)