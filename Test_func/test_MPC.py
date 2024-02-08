from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import sys
path_to_add='C:\\Users\\A490243\\Desktop\\Master_Thesis'
sys.path.append(path_to_add)
from Controller.MPC_tighten_bound import MPC_tighten_bound
from Controller.LTI_MPC import MPC
from vehicleModel.vehicle_model import car_VehicleModel
from util.utils import *


dt = 0.2
N=12
vehicleADV = car_VehicleModel(dt,N, width = 2, length = 4)
nx,nu,nrefx,nrefu = vehicleADV.getSystemDim()
int_opt = 'rk'
vehicleADV.integrator(int_opt,dt)
F_x_ADV  = vehicleADV.getIntegrator()
vx_init_ego = 10   
vehicleADV.setInit([0,0],vx_init_ego)

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
mpc_controller = MPC(vehicleADV, np.eye(nx), np.eye(nu), P0, process_noise, 0.95, N)
# Set initial conditions for the ego vehicle
x0 = np.array([[1], [0], [10], [0]])  # Initial state: [x, y, psi, v]. Example values provided

# Define reference trajectory and control for N steps
# For simplicity, setting reference states and inputs to zero or desired states.
ref_trajectory = np.zeros((nx, N + 1)) # Reference trajectory (states)
ref_trajectory[0,:] = 100
ref_trajectory[1,:] = 1


ref_control = np.zeros((nu, N))  # Reference control inputs

# Position of the leading vehicle (for IDM constraint)
# Assuming the leading vehicle is at 20 meters ahead initially
p_leading = 20

# Set the controller (this step initializes the optimization problem with cost and constraints)
mpc_controller.setController()

# Solve the MPC problem
u_opt, x_opt = mpc_controller.solve(x0, ref_trajectory, ref_control, p_leading)


plt.plot(x_opt[0,:],x_opt[1,:])
plt.show()
# Print the optimized control input for the first step
print("Optimized control input (steer_angle,acc) for the first step:", u_opt[:, 0])
