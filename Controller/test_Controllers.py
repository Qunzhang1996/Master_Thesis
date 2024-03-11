import sys
path_to_add='C:\\Users\\A490243\\Desktop\\Master_Thesis'
sys.path.append(path_to_add)
from Autonomous_Truck_Sim.helpers import *
from Controller.MPC_tighten_bound import MPC_tighten_bound
from vehicleModel.vehicle_model import car_VehicleModel
from Traffic.Traffic import Traffic
from Traffic.Scenarios import trailing, simpleOvertake
from util.utils import *

# System initialization 
dt = 0.2                    # Simulation time step (Impacts traffic model accuracy)
f_controller = 1            # Controller update frequency, i.e updates at each t = dt*f_controller
N =  30                     # MPC Horizon length

ref_vx = 54/3.6             # Higway speed limit in (m/s)

traffic = Traffic()
vehicleADV = car_VehicleModel(dt,N)
vehWidth,vehLength,L_tract,L_trail = vehicleADV.getSize()
nx,nu,nrefx,nrefu = vehicleADV.getSystemDim()
# Set Cost parameters
Q_ADV = [0,40,3e2,5,5]                            # State cost, Entries in diagonal matrix
R_ADV = [5,5]                                    # Input cost, Entries in diagonal matrix
q_ADV_decision = 50
vehicleADV.cost(Q_ADV,R_ADV)
vehicleADV.costf(Q_ADV)
L_ADV,Lf_ADV = vehicleADV.getCost()
# ------------------ Problem definition ---------------------
testWidth = 3.75
scenarioTrailADV = trailing(vehicleADV,N,lanes = 3,v_legal = ref_vx, laneWidth=testWidth)
scenarioADV = simpleOvertake(vehicleADV,N,lanes = 3,v_legal = ref_vx,laneWidth=testWidth)
roadMin, roadMax, laneCenters = scenarioADV.getRoad()
# -------------------- Traffic Set up -----------------------
# * Be carful not to initilize an unfeasible scenario where a collsion can not be avoided
# # Initilize ego vehicle
vx_init_ego = 10                                # Initial velocity of the ego vehicle
vehicleADV.setRoad(roadMin,roadMax,laneCenters)
vehicleADV.setInit([0,laneCenters[0]],vx_init_ego)


print(vehicleADV.xConstraints())
print(vehicleADV.uConstraints())