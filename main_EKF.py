'''
#!This file is to test the makeController function in the Controllers.py without considering the kalman filter
'''
import time
import sys
from casadi import *
from Controllers import makeController, makeDecisionMaster
from vehicle_model import car_VehicleModel
from Traffic import Traffic
from Scenarios import trailing, simpleOvertake
from kalman_filter import kalman_filter
from utils import *

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
# #  Run the command  


stochasticMPC = False
makeMovie = True
directory = r"C:\Users\A490243\Desktop\Master_Thesis\Figure\crazy_traffic_mix3_EKF_MPC.gif"

## ! --------------------------------------System initialization--------------------------------------------
dt = 0.3                  # Simulation time step (Impacts traffic model accuracy)
desired_interval = dt
dt_PID = dt/5              # Time step for the PID controller
f_controller = 10          # Controller update frequency, i.e updates at each t = dt*f_controller
N =  12        # MPC Horizon length
laneWidth = 3.5

ref_vx = 54/3.6             # Higway speed limit in (m/s)
ref_velocity=ref_vx
q_traffic_slack = 1e4
traffic = Traffic(N,dt)
velocities = {
        'normal': carla.Vector3D(0.75 * ref_vx, 0, 0),
        'passive': carla.Vector3D(0.65 * ref_vx, 0, 0),
        'aggressive': carla.Vector3D(0.9*ref_vx, 0, 0),  # Equal to 1.0 * ref_velocity for clarity
        'reference': carla.Vector3D(ref_vx, 0, 0)  # Specifically for the truck
    }
spawned_vehicles, center_line = traffic.setup_complex_carla_environment()
traffic.set_velocity(velocities)
Nveh = traffic.getDim()
vehList = traffic.getVehicles()
time.sleep(1)
px_init,py_init,vx_init=traffic.getStates()[:3,1] # get the initial position of the truck
truck = traffic.getEgo()  # get the ego vehicle

## ! -----------------------------------initialize the local controller-----------------------------------------
local_controller = VehiclePIDController(truck, 
                                        args_lateral = {'K_P': 1.5, 'K_I': 0.05, 'K_D': 0.2, 'dt': dt_PID}, 
                                        args_longitudinal = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.5, 'dt': dt_PID})

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
scenarioADV = simpleOvertake(vehicleADV,N,lanes = 3,v_legal = ref_vx,laneWidth=laneWidth)
scenarioADV.slackCost(q_traffic_slack)
#! get road INFOS
roadMin, roadMax, laneCenters, _ = scenarioTrailADV.getRoad()
#! initilize the ego vehicle
vehicleADV.setRoad(roadMin,roadMax,laneCenters)
vehicleADV.setInit([px_init,py_init],ref_vx )
# !----------------- Kalman Filter Settings ------------------------   
sigma_process=0.01
sigma_measurement=0.01
Q_0=np.eye(nx)*sigma_process**2
Q_0[0,0]=0.3  # x bound is [0, 3]
Q_0[1,1]=0.05  # y bound is [0, 0.1]
Q_0[2,2]=0.5  # v bound is [0, 1.8]
Q_0[3,3]=0.01**2  # psi bound is [0, 0.05]

R_0=np.eye(nx)*sigma_measurement
R_0[0,0]=0.1**2
R_0[1,1]=0.1**2 
R_0[2,2]=0.1**2
R_0[3,3]=(1/180*np.pi)**2

# ! get the param for the stochastic mpc
P0, _, possibility = set_stochastic_mpc_params()
vehicleADV.setStochasticMPCParams(P0, Q_0, possibility)
#! -----------------------------------------------------------------
#! -----------------------------------------------------------------
#!      Formulate optimal control problem using opti framework
#! -----------------------------------------------------------------
#! -----------------------------------------------------------------


opts1 = {"version" : "leftChange", "solver": "ipopt", "integrator":"LTI"}
MPC_LC= makeController(vehicleADV,traffic,scenarioADV,N,opts1,dt,stochasticMPC)
MPC_LC.setController()
#TODO: MPC_LC.setController()

opts2 = {"version" : "rightChange", "solver": "ipopt", "integrator":"LTI"}
MPC_RC= makeController(vehicleADV,traffic,scenarioADV,N,opts2,dt,stochasticMPC)
MPC_RC.setController()

opts3 = {"version" : "trailing", "solver": "ipopt", "integrator":"LTI"}
MPC_trailing= makeController(vehicleADV,traffic,scenarioTrailADV,N,opts3,dt,stochasticMPC)
MPC_trailing.setController()




print("INFO:  Initilization succesful.")               
                                                                        
#! -----------------------------------------Initilize Decision Master-----------------------------------------
decisionMaster = makeDecisionMaster(vehicleADV,traffic,[MPC_LC, MPC_RC, MPC_trailing],
                                [scenarioTrailADV,scenarioADV])
decisionMaster.setDecisionCost(q_ADV_decision)                  # Sets cost of changing decision

# ███████╗██╗███╗   ███╗██╗   ██╗██╗      █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
# ██╔════╝██║████╗ ████║██║   ██║██║     ██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
# ███████╗██║██╔████╔██║██║   ██║██║     ███████║   ██║   ██║██║   ██║██╔██╗ ██║
# ╚════██║██║██║╚██╔╝██║██║   ██║██║     ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
# ███████║██║██║ ╚═╝ ██║╚██████╔╝███████╗██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
tsim = 40                     # Total simulation time in seconds
Nsim = int(tsim/dt)
# # Initialize simulation
x_iter = DM(int(nx),1)
x_iter[:],u_iter = vehicleADV.getInit()
# ! get param for the kalman filter
P_kf=np.eye(nx)*1  # initial state covariance
# get system dynamic matrices
A,B=MPC_trailing.A, MPC_trailing.B
H=np.eye(nx)   #measurement matrix, y=H@x_iter         
#! set the initial state and control input
x_0 = x_iter
u_iter = np.array([0,0])
#! initial the Kalman Filter
ekf=kalman_filter(A,B,H,x_0,P_kf,Q_0,R_0)       
#! initial the Vehicle Model
vehicleADV.update(x_0,u_iter)

refxADV = [0,laneCenters[1],ref_vx,0,0]
refxT_in, refxL_in, refxR_in = vehicleADV.setReferences(ref_vx)
refu_in = [0,0,0]

refxT_out,refu_out = scenarioTrailADV.getReference(refxT_in,refu_in)
refxL_out,refu_out = scenarioADV.getReference(refxL_in,refu_in)
refxR_out,refu_out = scenarioADV.getReference(refxR_in,refu_in)
refxADV_out,refuADV_out = scenarioTrailADV.getReference(refxADV,refu_in)

# Traffic
nx_traffic = traffic.nx 
x_lead = DM(Nveh,N+1)
traffic_state = np.zeros((nx_traffic+1,N+1,Nveh))#! x, y, sign, shift, flip

#TODO: Store variables
X = np.zeros((nx,Nsim,1))
U = np.zeros((nu,Nsim,1))  
paramLog = np.zeros((5,Nsim,Nveh,3))
decisionLog = np.zeros((Nsim,),dtype = int) 

#! Data storage initialization
car_positions = []  # To store (x, y) positions
truck_positions = []  # To store (x, y) positions
truck_velocities = []  # To store velocity
leading_velocities = []  # To store leading vehicle velocity
truck_accelerations = []  # To store acceleration
truck_jerks = []  # To store jerk
truck_vel_mpc = []
lambda_s_list = []
truck_vel_control = []
Trajectory_pred = []  # To store the predicted trajectory
timestamps = []  # To store timestamps for calculating acceleration and jerk
all_tightened_bounds = []  # To store all tightened bounds for visualization
previous_acceleration = 0  # To help in jerk calculation
truck_y_mpc = []  
truck_y_control = []

#! TEST


X_pred = np.zeros((nx,N+1,Nsim))
X_traffic = np.zeros((nx_traffic,Nsim,Nveh))
X_traffic_ref = np.zeros((4,Nsim,Nveh))
# print("this is the size of teh traffic",(traffic.getStates()).shape)
X_traffic[:,0,:] = traffic.getStates()   
testPred = traffic.prediction()
feature_map = np.zeros((8,Nsim,Nveh+1))
 



# for i in range(0,Nsim):
# use tqdm for progress bar
import tqdm
for i in tqdm.tqdm(range(0,Nsim)):
    iteration_start = time.time()
    x_lead[:,:] = traffic.prediction()[0,:,:].transpose()
    traffic_state[:2,:,] = traffic.prediction()[:2,:,:]
    #! according to R_0 matrix, we add the noise to the measurement
    r = np.random.normal(0.0, 0.1, size=(nx, 1))
    r[3]=np.random.normal(0.0, (1/180*np.pi))

    if i % f_controller == 0:
        count = 0
        print("----------")
        print('Step: ', i)
        
        decisionMaster.storeInput([x_iter,refxL_out,refxR_out,refxT_out,refu_out,x_lead,traffic_state])
        #TODO: Update reference based on current lane
        
        u_opt, x_opt, X_out, decision_i  = decisionMaster.chooseController()
        Traj_ref = x_opt # Reference trajectory (states)
        # print("INFO: The referenc e of the truck is: ", Traj_ref[1,:])
        X_ref=Traj_ref[:,count] 
        u_iter = u_opt[:,count].reshape(-1,1)
        #! get the computed time of the MPC of real time
        print("INFO:  The computation time of the MPC is: ", [time.time()-iteration_start])
        # print("INFO: The reference of the truck is: ", Traj_ref[1,:])
    else: #! when mpc is asleep, the PID will track the Traj_ref step by step
        count = count + 1
        X_ref=Traj_ref[:,count]
        u_iter = u_opt[:,count].reshape(-1,1)
    for j in range(5):
        control_truck = local_controller.run_step(X_ref[2]*3.6, [X_ref[1],X_ref[0]] , False)
        truck.apply_control(control_truck)
    
    refxL_out,refxR_out,refxT_out = decisionMaster.updateReference(r)        
    #TODO: Update traffic and store data
    measurement_truck = get_state(truck) + r
    
    # !-----------------  do extended kalman filter ------------------------
    ekf.predict(u_iter, A, B)  # predict the state
    ekf.update(measurement_truck)  # update the state
    truck_estimate = ekf.get_estimate  # get the estimated state
    P_estimate = ekf.get_covariance  # get the estimated covariance
    x_iter = truck_estimate
    # !---------------------------------------------------------------------
    
    vehicleADV.update(x_iter,u_iter)
    truck_state = x_iter
    truck_x, truck_y, truck_v, truck_psi = float(truck_state[C_k.X_km]), float(truck_state[C_k.Y_km]), \
                                            float(truck_state[C_k.V_km]), float(truck_state[C_k.Psi])
    X[:,i] = np.array([truck_x, truck_y, truck_v, truck_psi]).reshape(-1,1)
    U[:,i] = u_iter
    X_pred[:,:,i] = X_out
    X_traffic[:,i,:] = traffic.getStates()
    #TODO: 
    # X_traffic_ref[:,i,:] = traffic.getReference()
    paramLog[:,i,:] = decisionMaster.getTrafficState()
    decisionLog[i] = decision_i.item()
    
    
    #! ------------------------------------------------------------------------------------------------
    car_state = traffic.getStates()[:,2]
    car_x, car_y, car_v = car_state[C_k.X_km].item(), car_state[C_k.Y_km].item(), car_state[C_k.V_km].item()
    
    truck_state_ctr = get_state(truck)
    # print(f"current state of the truck is: {truck_state_ctr}")
    truck_vel_mpc.append(X_ref[2])
    truck_y_mpc.append(X_ref[1])
    truck_vel_ctr=truck_state_ctr[C_k.V_km].item()
    truck_y_control.append(truck_state_ctr[C_k.Y_km].item())
    truck_vel_control.append(truck_vel_ctr)
    # Data collection inside the loop
    Trajectory_pred.append(x_opt[:2,:]) # store the predicted trajectory
    current_time = time.time()
    timestamps.append(current_time)
    car_positions.append((car_x, car_y))
    truck_positions.append((truck_x, truck_y, truck_psi))
    truck_velocities.append(truck_v)
    leading_velocities.append(car_v)

    # Calculate acceleration if possible
    if len(timestamps) > 1:
        delta_v = truck_velocities[-1] - truck_velocities[-2]
        delta_t = timestamps[-1] - timestamps[-2]
        acceleration = delta_v / delta_t
        truck_accelerations.append(acceleration)

        # Calculate jerk if possible
        if len(truck_accelerations) > 1:
            delta_a = truck_accelerations[-1] - previous_acceleration
            jerk = delta_a / delta_t
            truck_jerks.append(jerk)
            previous_acceleration = truck_accelerations[-1]
    else:
        truck_accelerations.append(0)  # Initial acceleration is 0
        truck_jerks.append(0)  # Initial jerk is 0
        
        
    
    #! ------------------------------------------------------------------------------------------------

    iteration_duration = time.time() - iteration_start
    sleep_duration = max(0.001, desired_interval - iteration_duration)
    time.sleep(sleep_duration)

print("Simulation finished")
i_crit = i     
                                                
if makeMovie:
    borvePictures(X,X_traffic,X_traffic_ref,paramLog,decisionLog,vehList,X_pred,vehicleADV,scenarioTrailADV,scenarioADV,traffic,i_crit,f_controller,directory)
