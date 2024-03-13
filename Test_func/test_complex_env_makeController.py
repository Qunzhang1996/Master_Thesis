'''
#!This file is to test the makeController function in the Controllers.py without considering the kalman filter
'''
import time
import sys
sys.path.append(r'C:\Users\A490243\Desktop\Master_Thesis')
from Autonomous_Truck_Sim.helpers import *
from Controller.MPC_tighten_bound import MPC_tighten_bound
from Controller.Controllers import makeController, makeDecisionMaster
from vehicleModel.vehicle_model import car_VehicleModel
from Traffic.Traffic import Traffic
from Traffic.Scenarios import trailing, simpleOvertake
from util.utils import *

sys.path.append(r'C:\Users\A490243\CARLA\CARLA_Latest\WindowsNoEditor\PythonAPI\carla')
from agents.navigation.controller import VehiclePIDController

# # ------------------------change map to Town06------------------------
# import subprocess
# # Command to run your script
# command = (
#     r'cd C:\Users\A490243\CARLA\CARLA_Latest\WindowsNoEditor\PythonAPI\util && '
#     r'python config.py --map Town06')
# subprocess.run(command, shell=True)
# exit()
# # Run the command

## ! --------------------------------------System initialization--------------------------------------------
dt = 0.2                    # Simulation time step (Impacts traffic model accuracy)
desired_interval = dt
dt_PID = 0.2/10                # Time step for the PID controller
f_controller = 1            # Controller update frequency, i.e updates at each t = dt*f_controller
N =  12                    # MPC Horizon length
laneWidth = 3.5
ref_vx = 54/3.6             # Higway speed limit in (m/s)
q_traffic_slack = 1e4
traffic = Traffic()
velocities = {
        'normal': carla.Vector3D(0.8 * ref_vx, 0, 0),
        'passive': carla.Vector3D(0.65 * ref_vx, 0, 0),
        'aggressive': carla.Vector3D(1.1*ref_vx, 0, 0),  # Equal to 1.0 * ref_velocity for clarity
        'reference': carla.Vector3D(ref_vx, 0, 0)  # Specifically for the truck
    }
spawned_vehicles, center_line = traffic.setup_complex_carla_environment()
traffic.set_velocity(velocities)
Nveh = traffic.getDim()
px_init,py_init,vx_init=traffic.getStates()[:3,1] # get the initial position of the truck
print("this is vx_init",vx_init)
truck = traffic.getEgo()  # get the ego vehicle


## ! -----------------------------------initialize the local controller-----------------------------------------
local_controller = VehiclePIDController(truck, 
                                        args_lateral = {'K_P': 1.5, 'K_I': 0.05, 'K_D': 0.2, 'dt': dt_PID}, 
                                        args_longitudinal = {'K_P': 1.95, 'K_I': 0.5, 'K_D': 0.2, 'dt': dt_PID})

'''The following code is to test the controller with the carla environment'''
# while(True):
#     control_Truck = local_controller.run_step(ref_vx*3.6-50, 143.318146, False)
#     truck.apply_control(control_Truck)


## ! -----------------------------------initialize the VehicleModel-----------------------------------------
vehicleADV = car_VehicleModel(dt,N)
vehWidth,vehLength,L_tract,L_trail = vehicleADV.getSize()
nx,nu,nrefx,nrefu = vehicleADV.getSystemDim()
# Set Cost parameters
Q_ADV = [0,40,3e2,5]                            # State cost, Entries in diagonal matrix
R_ADV = [5,5]                                   # Input cost, Entries in diagonal matrix
q_ADV_decision = 50
vehicleADV.cost(Q_ADV,R_ADV)
vehicleADV.costf(Q_ADV)
L_ADV,Lf_ADV = vehicleADV.getCost()


## ! --------------------------------------- Problem definition ---------------------------------------------
scenarioTrailADV = trailing(vehicleADV,N,lanes = 3,v_legal = ref_vx, laneWidth=laneWidth)
scenarioTrailADV.slackCost(q_traffic_slack)
#TODO: ADD scenarioADV LATTER
# scenarioADV = simpleOvertake(vehicleADV,N,lanes = 3,v_legal = ref_vx,laneWidth=laneWidth)
# scenarioTrailADV.slackCost(q_traffic_slack)
roadMin, roadMax, laneCenters = scenarioTrailADV.getRoad()
#! initilize the ego vehicle
vehicleADV.setRoad(roadMin,roadMax,laneCenters)
vehicleADV.setInit([px_init,py_init],10 )


#! -----------------------------------------------------------------
#! -----------------------------------------------------------------
#!      Formulate optimal control problem using opti framework
#! -----------------------------------------------------------------
#! -----------------------------------------------------------------


opts1 = {"version" : "trailing", "solver": "ipopt", "integrator":"LTI"}
MPC_trailing= makeController(vehicleADV,traffic,scenarioTrailADV,N,opts1,dt)
MPC_trailing.setController()
#TODO: ADD LEFT AND RIGHT LANE CHANGE LATTER
print("INFO:  Initilization succesful.")               

                                                                                          
#! -----------------------------------------Initilize Decision Master-----------------------------------------
decisionMaster = makeDecisionMaster(vehicleADV,traffic,MPC_trailing,
                                [scenarioTrailADV])
decisionMaster.setDecisionCost(q_ADV_decision)                  # Sets cost of changing decision

# ███████╗██╗███╗   ███╗██╗   ██╗██╗      █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
# ██╔════╝██║████╗ ████║██║   ██║██║     ██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
# ███████╗██║██╔████╔██║██║   ██║██║     ███████║   ██║   ██║██║   ██║██╔██╗ ██║
# ╚════██║██║██║╚██╔╝██║██║   ██║██║     ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
# ███████║██║██║ ╚═╝ ██║╚██████╔╝███████╗██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
tsim = 100                         # Total simulation time in seconds
Nsim = int(tsim/dt)
# # Initialize simulation
x_iter = DM(int(nx),1)
x_iter[:],u_iter = vehicleADV.getInit()
vehicleADV.update(x_iter,u_iter)

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

#TODO: Store variables
X = np.zeros((nx,Nsim,1))
U = np.zeros((nu,Nsim,1))    


X_pred = np.zeros((nx,N+1,Nsim))
X_traffic = np.zeros((nx_traffic,Nsim,Nveh))
X_traffic_ref = np.zeros((4,Nsim,Nveh))
# print("this is the size of teh traffic",(traffic.getStates()).shape)
X_traffic[:,0,:] = traffic.getStates()   

for i in range(Nsim):
    iteration_start = time.time()
    x_lead[:,:] = traffic.prediction()[0,:,:].transpose()
    traffic_state[:2,:,] = traffic.prediction()[:2,:,:]
    if i%5==0:
        count = 1
        print("----------")
        print('Step: ', i)
        decisionMaster.storeInput([x_iter,refxL_out,refxR_out,refxT_out,refu_out,x_lead,traffic_state])
        #TODO: Update reference based on current lane
        # refxL_out,refxR_out,refxT_out = decisionMaster.updateReference()
        u_opt, x_opt, cost = decisionMaster.chooseController()
        Traj_ref = x_opt # Reference trajectory (states)
        # print("this is the reference trajectory",Traj_ref)
        X_ref=Traj_ref[:,count] #last element
        print("INFO:  The Cost is: ", cost)
    else: #! when mpc is asleep, the PID will track the Traj_ref step by step
        count = count + 1
        X_ref=Traj_ref[:,count]
        
    for j in range(10):
        control_truck = local_controller.run_step(X_ref[2]*3.6, X_ref[1], False)
        truck.apply_control(control_truck)
        
    #TODO: Update traffic and store data
    # X[:,i] = x_iter
    # U[:,i] = u_iter
    x_iter = get_state(truck)
    print("this is the state of the truck",x_iter.T)

    iteration_duration = time.time() - iteration_start
    sleep_duration = max(0.001, desired_interval - iteration_duration)
    time.sleep(sleep_duration)
    
    if i == 220: break
        
                                                       