import numpy as np
from casadi import *
import sys
# sys.path.append(r'C:\Users\A490243\Desktop\Master_Thesis\Controller')
# C:\Users\86232\Desktop\masterthesis\Master_Thesis\Controller
sys.path.append(r'C:\Users\86232\Desktop\masterthesis\Master_Thesis\Controller')
from MPC_tighten_bound import MPC_tighten_bound

#test the function
# ██████╗  █████╗ ███████╗███████╗
# ██╔══██╗██╔══██╗██╔════╝██╔════╝
# ██████╔╝███████║███████╗███████╗
# ██╔═══╝ ██╔══██║╚════██║╚════██║
# ██║     ██║  ██║███████║███████║
# ╚═╝     ╚═╝  ╚═╝╚══════╝╚══════╝

#! A,B here is the linearized system, and the system is constant velocity model at x=0,y=0, v=10m/s, theta=0
A=np.array([[1, 0, 0.3, -0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
# A = DM(A)
B=np.array([[0.0, 0.0], 
            [0.0, 0.0], 
            [0.0, 0.3], 
            [0.75, 0.0]])
# B=DM(B)
#! P0 here is converged  kf covariance
x0=np.array([[0],[0],[15],[0]])
# x0=DM(x0)
P0 = np.array([[0.002071893043635329, -9.081755508401041e-07, 6.625835814018275e-06, 3.5644803460420476e-06],
            [-9.081755508401084e-07, 0.0020727698104820867, -1.059020050149629e-05, 2.218506297137566e-06], 
            [6.625835814018292e-06, -1.0590200501496282e-05, 0.0020706909538251504, 5.4487618678242517e-08], 
            [3.5644803460420577e-06, 2.2185062971375356e-06, 5.448761867824289e-08, 0.002071025561957967]])
# P0=DM(P0)
nx=A.shape[0]
nu=B.shape[1]
D=np.eye(nx)  # noise matrix
# D=DM(D)
R_ADV = [5,5]  
Q=np.diag([0,40,3e2,5])  # cost matrix
# Q=DM(Q)
R=np.diag([5,5])  # cost matrix  for input
# R=DM(R)
process_noise=np.eye(4)  # process noise
# process_noise=DM(process_noise)
process_noise[0,0]=0.3  # x bound is [0, 3]
process_noise[1,1]=0.05  # y bound is [0, 0.1]
process_noise[2,2]=0.5  # v bound is [0, 1.8]
process_noise[3,3]=0.01**2  # psi bound is [0, 0.05]
# stochastic mpc parameters
Possibilty=0.95  # the possibility of the tightened bound
# initial upper bound of the constraints
# Here is the upper bound of the constraints
H_up=[np.array([[1],[0],[0],[0]]),np.array([[0],[1],[0],[0]]), np.array([[0],[0],[1],[0]]), np.array([[0],[0],[0],[1]])]
upb=np.array([[5000],[143.318146+3.5+1.75],[30],[3.14/8]])  #no limit to the x, y should be in [-1.75, 1.75], v should be in [0,30], psi should be in [-3.14/8,3.14/8]
# Here is the lower bound of the constraints
H_low=[np.array([[-1],[0],[0],[0]]),np.array([[0],[-1],[0],[0]]), np.array([[0],[0],[-1],[0]]), np.array([[0],[0],[0],[-1]])]
lwb=np.array([[0],[-143.318146+3.5+1.75],[0],[3.14/8]])  #no limit to the x, y should be in [-1.75, 1.75], v should be in [0,30], psi should be in [-3.14/8,3.14/8]
# Predict Horizon 
N=12
#calculate the Nth tightened bound
MPC_tighten_bound_=MPC_tighten_bound(A,B,D,Q,R,P0,process_noise,Possibilty)
tightened_bound_N_list_up=MPC_tighten_bound_.tighten_bound_N(P0,H_up,upb,N,1)
tightened_bound_N_list_lw=MPC_tighten_bound_.tighten_bound_N(P0,H_low,lwb,N,0)
print(tightened_bound_N_list_up)
print(tightened_bound_N_list_lw)
print("###############################################################################################" )
print("This is the temptX", MPC_tighten_bound_.getXtemp(N))
print("###############################################################################################" )
print("This is the temptY", MPC_tighten_bound_.getYtemp(N))

IDM_constraint_list = []
for i in range(N+1):
    IDM_constraint_list.append(100+10*i*0.2)
IDM_constraint_list = np.array(IDM_constraint_list).reshape(-1, 1)
tightened_bound_N_IDM_list,_=MPC_tighten_bound_.tighten_bound_N_IDM(IDM_constraint_list,N)
# print(tightened_bound_N_IDM_list)


# #visualize the tightened bound in 4 subplots
import matplotlib.pyplot as plt
import numpy as np

# Assuming 'tightened_bound_N_list_up', 'tightened_bound_N_list_lw', 'upb', 'lwb', 'N', 
# 'IDM_constraint_list', 'tightened_bound_N_IDM_list' are defined elsewhere in your code

# Font settings for readability
plt.rcParams.update({'font.size': 12, 'font.family': 'Times New Roman'})
# use latex for text rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# fig, axs = plt.subplots(3, 2, figsize=(12, 8))


# Titles for the first 4 subplots


titles = ['Constraint for x', 'Constraint for y', 'Constraint for v', 'Constraint for $\psi$']
constraints_units = ['[m]', '[m]', '[m/s]', '[rad]']
file_names = ['Tightened_x.png', 'Tightened_y.png', 'Tightened_v.png', 'Tightened_psi.png']

for i in range(4):
    plt.figure(figsize=(6, 4))
    plt.plot(tightened_bound_N_list_up[:, i], '-', label='Tightened Upper Constraint', linewidth=2)  # Enhanced visibility
    plt.scatter(range(N+1), tightened_bound_N_list_up[:, i], color='blue', s=10)
    plt.plot(tightened_bound_N_list_lw[:, i], '-', label='Tightened Lower Constraint', linewidth=2)  
    plt.scatter(range(N+1), tightened_bound_N_list_lw[:, i], color='red', s=10)
    plt.plot(np.ones((N+1, 1))*upb[i], '--', label='Original Upper Constraint', linewidth=1.5)
    plt.plot(-np.ones((N+1, 1))*lwb[i], '--', label='Original Lower Constraint', linewidth=1.5)
    plt.legend()
    # x axis label time steps
    plt.title(titles[i])
    plt.xlabel('Time Steps')
    plt.ylabel('Constraint ' + constraints_units[i])
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'C:/Users/86232/Desktop/masterthesis/Master_Thesis/Figure/Figure_for_Thesis/TightenedConstraints/Tightened_States_{file_names[i]}', dpi=300)
    plt.close()


# for i in range(4):
#     plt.figure(figsize=(6, 4))
#     plt.plot(range(N+1), tightened_bound_N_list_up[:, i], 'b-', label='Tightened Upper Bound')
#     plt.plot(range(N+1), tightened_bound_N_list_lw[:, i], 'r-', label='Tightened Lower Bound')
#     plt.axhline(y=upb[i, 0], color='b', linestyle='--', label='Original Upper Bound')
#     plt.axhline(y=-lwb[i, 0], color='r', linestyle='--', label='Original Lower Bound')
#     plt.title(titles[i])
#     plt.xlabel('Time Steps')
#     plt.ylabel('Constraint ' + constraints_units[i])
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'C:/Users/86232/Desktop/masterthesis/Master_Thesis/Figure/Figure_for_Thesis/TightenedConstraints/Tightened_States_{file_names[i]}', dpi=300)
#     plt.close()




