from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import sys
path_to_add='C:\\Users\\A490242\\Desktop\\Master_Thesis'
sys.path.append(path_to_add)
# from Controller.LTI_MPC import MPC
from Controller.LC_MPC import LC_MPC
from vehicleModel.vehicle_model import car_VehicleModel
from util.utils import *
from Controller.scenarios import trailing, simpleOvertake


dt = 0.1
N=12
vehicleADV = car_VehicleModel(dt,N, width = 2, length = 4)
trafficADV1 = car_VehicleModel(dt,N, width = 2, length = 4) # create a traffic vehicle
trafficADV2 = car_VehicleModel(dt,N, width = 2, length = 4) # create a traffic vehicle
nx,nu,nrefx,nrefu = vehicleADV.getSystemDim()
int_opt = 'rk'
vehicleADV.integrator(int_opt,dt)
F_x_ADV  = vehicleADV.getIntegrator()
vx_init_ego = 10
vx_init_traffic = 9
laneWidth = 3.5  
vehicleADV.setInit([0,0],vx_init_ego) # the ego vehicle is at the origin and center lane
trafficADV1.setInit([10,0],vx_init_traffic) # the traffic vehicle is 10m ahead of the ego vehicle and center lane
# set the initial states of the second traffic vehicle to be 10m ahead of the first traffic vehicle and right lane
trafficADV2.setInit([10,3.5],vx_init_traffic)

# create a list of traffic vehicles
traffic = [trafficADV1,trafficADV2]


Q_ADV = [0,40,3e2,5]                            # State cost, Entries in diagonal matrix
R_ADV = [5,5]                                    # Input cost, Entries in diagonal matrix
vehicleADV.cost(Q_ADV,R_ADV)
vehicleADV.costf(Q_ADV)
L_ADV,Lf_ADV = vehicleADV.getCost()
# process_noise=DM(process_noise)
P0 = np.array([[0.002071893043635329, -9.081755508401041e-07, 6.625835814018275e-06, 3.5644803460420476e-06],
            [-9.081755508401084e-07, 0.0020727698104820867, -1.059020050149629e-05, 2.218506297137566e-06], 
            [6.625835814018292e-06, -1.0590200501496282e-05, 0.0020706909538251504, 5.4487618678242517e-08], 
            [3.5644803460420577e-06, 2.2185062971375356e-06, 5.448761867824289e-08, 0.002071025561957967]])
process_noise=np.eye(4)  # process noise
process_noise[0,0]=0.5  # x bound is [0, 3]
process_noise[1,1]=0.01/6  # y bound is [0, 0.1]
process_noise[2,2]=1.8/6  # v bound is [0, 1.8]
process_noise[3,3]=0.05/6  # psi bound is [0, 0.05]

scenarioTrailADV = trailing(vehicleADV,N,lanes = 3, laneWidth=3.5)
scenarioOvertakeADV = simpleOvertake(vehicleADV,N,lanes = 3, laneWidth=3.5)
print(scenarioOvertakeADV.constraint(traffic=trafficADV1,opts=None))
version1 = {"version" : "leftChange"}
version2 = {"version" : "rightChange"}
version3 = {"version" : "trailing"}

# controller to calculate MPC for the ego vehicle in trailing scenario 
mpc_controller = LC_MPC(vehicleADV, traffic, np.diag([0,40,3e2,5]), np.diag([5,5]), P0, process_noise, 0.95, N, version3, scenarioTrailADV)
# controller to calculate MPC for the ego vehicle in simple overtake scenario
mpc_controller1 = LC_MPC(vehicleADV, traffic, np.diag([0,40,3e2,5]), np.diag([5,5]), P0, process_noise, 0.95, N, version1, scenarioOvertakeADV)
# Set initial conditions for the ego vehicle
x0 = np.array([[0], [0], [10], [0]])  # Initial state: [x, y, psi, v]. Example values provided

# Define reference trajectory and control for N steps
# For simplicity, setting reference states and inputs to zero or desired states.
ref_trajectory = np.zeros((nx, N + 1)) # Reference trajectory (states)
ref_trajectory[0,:] = 0
ref_trajectory[1,:] = 0
ref_trajectory[2,:] = 10

ref_control = np.zeros((nu, N))  # Reference control inputs

# Position of the leading vehicle (for IDM constraint)
# Assuming the leading vehicle is at 20 meters ahead initially
p_leading = 30

# Set the controller (this step initializes the optimization problem with cost and constraints)
mpc_controller.setController()
x_plot = [] 
x_plot.append(x0[0].item())
y_plot = []
y_plot.append(x0[1].item())
v_plot = [10]
tightened_IDM_constraints_plot = []

# Solve the MPC problem
for i in range(100):
    u_opt, x_opt, lambda_s, tightened_IDM_constraints = mpc_controller.solve(x0, ref_trajectory, ref_control, p_leading)
    x0=x_opt[:,1]
    p_leading = p_leading + x0[2]*dt
    x_plot.append(x0[0])
    y_plot.append(x0[1])
    v_plot.append(x0[2])
    tightened_IDM_constraints_plot.append(tightened_IDM_constraints)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x_plot, y_plot, '-o')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Optimal trajectory')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(v_plot, '-o')
plt.xlabel('Time step')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity')
plt.grid(True)

plt.show()
    
# plt.figure(figsize=(15, 5))

# # Plotting optimal trajectory
# plt.subplot(1, 3, 1)
# plt.plot(x_plot, y_plot, '-o')
# plt.xlabel('x (m)')
# plt.ylabel('y (m)')
# plt.title('Optimal trajectory')
# plt.grid(True)

# # Plotting velocity
# plt.subplot(1, 3, 2)
# plt.plot(v_plot, '-o', label='Velocity')
# plt.xlabel('Time step')
# plt.ylabel('Velocity (m/s)')
# plt.title('Velocity')
# plt.grid(True)

# # Plotting tightened IDM constraints
# plt.subplot(1, 3, 3)
# plt.plot(tightened_IDM_constraints_plot, '-o', label='Tightened IDM Constraints')
# plt.xlabel('Time step')
# plt.ylabel('Constraint')
# plt.title('Tightened IDM Constraints')
# plt.grid(True)

# plt.tight_layout()  # Adjust subplot parameters to give specified padding.
# plt.show()
