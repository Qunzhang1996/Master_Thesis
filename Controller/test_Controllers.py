import sys
path_to_add='C:\\Users\\A490243\\Desktop\\Master_Thesis'
sys.path.append(path_to_add)
from Autonomous_Truck_Sim.helpers import *
from Controller.MPC_tighten_bound import MPC_tighten_bound
from Controller.Controllers import makeController, makeDecisionMaster
from vehicleModel.vehicle_model import car_VehicleModel
from Traffic.Traffic import Traffic
from Traffic.Scenarios import trailing, simpleOvertake
from util.utils import *


# # ------------------------change map to Town06------------------------
# import subprocess
# # Command to run your script
# command = (
#     r'cd C:\Users\A490243\CARLA\CARLA_Latest\WindowsNoEditor\PythonAPI\util && '
#     r'python config.py --map Town06')
# subprocess.run(command, shell=True)
# exit()
# # Run the command

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
Nveh = traffic.getDim()
px_init,py_init=traffic.getStates()[:2,1]
# print("this is px,py",px_init,py_init)

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
scenarioTrailADV.slackCost(q_traffic_slack)
roadMin, roadMax, laneCenters = scenarioTrailADV.getRoad()
# -------------------- Traffic Set up -----------------------
# * Be carful not to initilize an unfeasible scenario where a collsion can not be avoided
# # Initilize ego vehicle
vx_init_ego = 10                                # Initial velocity of the ego vehicle
vehicleADV.setRoad(roadMin,roadMax,laneCenters)
vehicleADV.setInit([px_init,py_init],vx_init_ego)



opts1 = {"version" : "trailing", "solver": "ipopt", "integrator":"LTI"}


#! this is the MPC controller

MPC_trailing= makeController(vehicleADV,traffic,scenarioTrailADV,N,opts1,dt)
MPC_trailing.setController()
#! this is the decision master
# Initilize Decision maker
decisionMaster = makeDecisionMaster(vehicleADV,traffic,MPC_trailing,
                                [scenarioTrailADV])
decisionMaster.setDecisionCost(q_ADV_decision)                  # Sets cost of changing decision

# # -----------------------------------------------------------------
# # -----------------------------------------------------------------
# #                         Simulate System
# # -----------------------------------------------------------------
# # -----------------------------------------------------------------

tsim = 10                         # Total simulation time in seconds
Nsim = int(tsim/dt)
tspan = np.linspace(0,tsim,Nsim)

# # Initialize simulation
x_iter = DM(int(nx),1)
x_iter[:],u_iter = vehicleADV.getInit()
vehicleADV.update(x_iter,u_iter)

# print("this is get leading vehicle",scenarioTrailADV.getLeadVehicle(traffic))



refxADV = [0,laneCenters[1],ref_vx,0,0]
refxT_in, refxL_in, refxR_in = vehicleADV.setReferences(ref_vx)

refu_in = [0,0,0]
refxT_out,refu_out = scenarioTrailADV.getReference(refxT_in,refu_in)
refxL_out,refu_out = scenarioTrailADV.getReference(refxL_in,refu_in)
refxR_out,refu_out = scenarioTrailADV.getReference(refxR_in,refu_in)
refxADV_out,refuADV_out = scenarioTrailADV.getReference(refxADV,refu_in)

# Traffic
nx_traffic = traffic.nx
x_lead = DM(Nveh,N+1)
traffic_state = np.zeros((nx_traffic,N+1,Nveh))

# # Store variables
X = np.zeros((nx,Nsim,1))
U = np.zeros((nu,Nsim,1))
paramLog = np.zeros((5,Nsim,Nveh,3))
decisionLog = np.zeros((Nsim,),dtype = int)

X_pred = np.zeros((nx,N+1,Nsim))

X_traffic = np.zeros((nx_traffic,Nsim,Nveh))
X_traffic_ref = np.zeros((4,Nsim,Nveh))
# print("this is the size of teh traffic",(traffic.getStates()).shape)
X_traffic[:,0,:] = traffic.getStates()
# print(X_traffic)
testPred = traffic.prediction()

feature_map = np.zeros((5,Nsim,Nveh+1))

#! TEST THOSE SHOWN BELOW:!!!!!
x_lead[:,:] = traffic.prediction()[0,:,:].transpose()
traffic_state[:2,:,] = traffic.prediction()[:2,:,:]
# print("this is the traffic state",traffic_state[:2,:,0])
decisionMaster.storeInput([x_iter,refxL_out,refxR_out,refxT_out,refu_out,x_lead,traffic_state])
u_opt, x_opt, cost=decisionMaster.chooseController()
print("this is the cost",cost)