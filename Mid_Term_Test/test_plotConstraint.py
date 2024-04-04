import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are
from Controllers import makeController, makeDecisionMaster
from vehicle_model import car_VehicleModel
from Traffic import Traffic
import carla
from Scenarios import trailing, simpleOvertake
from util.utils import *
frameSize = 100
Nveh = 8
#! import data from the parameters



dt = 0.3
laneWidth=3.5
N = 12
ref_vx = 54/3.6             # Higway speed limit in (m/s)
ref_velocity=ref_vx
q_traffic_slack = 1e4
leadLength = 4.78536
d_lat_spread = 0

traffic = Traffic(N,dt)
velocities = {
        'normal': carla.Vector3D(0.75 * ref_vx, 0, 0),
        'passive': carla.Vector3D(0.65 * ref_vx, 0, 0),
        'aggressive': carla.Vector3D(0.9*ref_vx, 0, 0),  # Equal to 1.0 * ref_velocity for clarity
        'reference': carla.Vector3D(ref_vx, 0, 0)  # Specifically for the truck
    }
spawned_vehicles, center_line = traffic.setup_complex_carla_environment()
traffic.set_velocity(velocities)
Nveh = 8

vehicleADV = car_VehicleModel(dt,N)
vehWidth,vehLength,L_tract,L_trail = vehicleADV.getSize()
nx,nu,nrefx,nrefu = vehicleADV.getSystemDim()
# Set Cost parameters
Q_ADV = [0,80,3e2,5]                            # State cost, Entries in diagonal matrix
R_ADV = [5,5]                                   # Input cost, Entries in diagonal matrix
q_ADV_decision = 50
vehicleADV.cost(Q_ADV,R_ADV)
vehicleADV.costf(Q_ADV)
L_ADV,Lf_ADV = vehicleADV.getCost()
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

trailing_plot = False


if trailing_plot:
    i = 30
else:
    i = 20
scenarioTrailADV = trailing(vehicleADV,N,lanes = 3,v_legal = ref_vx, laneWidth=laneWidth)
scenarioTrailADV.slackCost(q_traffic_slack)
scenarioADV = simpleOvertake(vehicleADV,N,lanes = 3,v_legal = ref_vx,laneWidth=laneWidth)
scenarioADV.slackCost(q_traffic_slack)
vehicle = vehicleADV
scenario = scenarioADV
vehWidth, vehLength,_,_ = vehicle.getSize()
temp_x, tempt_y = vehicle.getTemptXY()
roadMin, roadMax, laneCenters,laneWidth = scenario.getRoad()
decision_string = ["Change Left","Change Right","Keep Lane"]



print("this is the decisionLog: ", temp_x, tempt_y )

constraint_laneChange = scenario.constraint(traffic,[])
paramLog = np.load(r'C:\Users\A490243\Desktop\Master_Thesis\Parameters\paramLog_no_stochastic.npy', allow_pickle=True)
decisionLog = np.load(r'C:\Users\A490243\Desktop\Master_Thesis\Parameters\decisionLog_no_stochastic.npy', allow_pickle=True)
X_traffic = np.load(r'C:\Users\A490243\Desktop\Master_Thesis\Parameters\X_traffic_no_stochastic.npy', allow_pickle=True)
X = np.load(r'C:\Users\A490243\Desktop\Master_Thesis\Parameters\X_no_stochastic.npy', allow_pickle=True)

plt.rcParams.update({'font.size': 12, 'font.family': 'Times New Roman'})
plt.figure(figsize=(12, 6))

plot_tanhConstraint(i, X_traffic, traffic,constraint_laneChange,paramLog,decisionLog,X,vehWidth,d_lat_spread,scenarioTrailADV,frameSize,Nveh,laneWidth,leadLength,color_plt='g--')



    
temp_x =5
tempt_y =0.3
constraint_laneChange = scenario.constrain_tightened(traffic,[],temp_x, tempt_y)
#! plot the tightened lane change constraint
paramLog = np.load(r'C:\Users\A490243\Desktop\Master_Thesis\Parameters\paramLog.npy', allow_pickle=True)
decisionLog = np.load(r'C:\Users\A490243\Desktop\Master_Thesis\Parameters\decisionLog.npy', allow_pickle=True)
X = np.load(r'C:\Users\A490243\Desktop\Master_Thesis\Parameters\X.npy', allow_pickle=True)
X_traffic = np.load(r'C:\Users\A490243\Desktop\Master_Thesis\Parameters\X_traffic.npy', allow_pickle=True)
plot_tanhConstraint(i,X_traffic, traffic, constraint_laneChange,paramLog,decisionLog,X,vehWidth,d_lat_spread,scenarioTrailADV,frameSize,Nveh,laneWidth,leadLength,color_plt='r')
    
plt.plot([0,300], [143.318146, 143.318146], color='gray', linestyle='--', lw=1)
plt.plot([0,300], [143.318146-3.5, 143.318146-3.5], color='gray', linestyle='--', lw=1)
plt.plot([0,300], [143.318146+3.5, 143.318146+3.5], color='gray', linestyle='--', lw=1)
# plot th[0,300]line
plt.plot([0,300], [143.318146-1.75, 143.318146-1.75], 'k', lw=1)
plt.plot([0,300], [143.318146+1.75, 143.318146+1.75], 'k', lw=1)
plt.plot([0,300], [143.318146-1.75-3.5, 143.318146-1.75-3.5], 'k', lw=1)
plt.plot([0,300], [143.318146+1.75+3.5, 143.318146+1.75+3.5], 'k', lw=1)
if trailing_plot:
    plt.xlim(150,250)
    plt.ylim(130, 160)
else:
    plt.xlim(120,210)
    plt.ylim(130, 160)

plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.tight_layout()
plt.legend(loc='upper right')
#! save the figure
if trailing_plot:
    plt.title('Tightened Lane Change Constraint when doing Trailing')
    plt.savefig(r'C:\Users\A490243\Desktop\Master_Thesis\Figure\Tightened_trailing.png')
else:
    plt.title('Tightened Lane Change Constraint when doing Overtake')
    plt.savefig(r'C:\Users\A490243\Desktop\Master_Thesis\Figure\Tightened_overtake.png')

plt.show()