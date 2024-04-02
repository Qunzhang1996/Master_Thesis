import numpy as np
from casadi import *
import sys
sys.path.append(r'C:\Users\A490243\Desktop\Master_Thesis\Controller')
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
# print(tightened_bound_N_list_up)

IDM_constraint_list = []
for i in range(N+1):
    IDM_constraint_list.append(100+10*i*0.2)
IDM_constraint_list = np.array(IDM_constraint_list).reshape(-1, 1)
tightened_bound_N_IDM_list,_=MPC_tighten_bound_.tighten_bound_N_IDM(IDM_constraint_list,N)
print(tightened_bound_N_IDM_list)


# #visualize the tightened bound in 4 subplots
import matplotlib.pyplot as plt
import numpy as np

# Assuming 'tightened_bound_N_list_up', 'tightened_bound_N_list_lw', 'upb', 'lwb', 'N', 
# 'IDM_constraint_list', 'tightened_bound_N_IDM_list' are defined elsewhere in your code

# Font settings for readability
plt.rcParams.update({'font.size': 12, 'font.family': 'Times New Roman'})

fig, axs = plt.subplots(3, 2, figsize=(12, 8))
fig.suptitle('Tightened Bound and IDM Constraint Comparison', fontsize=14, fontweight='bold')

# Titles for the first 4 subplots
titles = ['Constraint for x', 'Constraint for y', 'Constraint for v', 'Constraint for phi']
for i in range(4):
    row, col = divmod(i, 2)
    axs[row, col].plot(tightened_bound_N_list_up[:, i], '-', label='Upper Bound', linewidth=2)  # Enhanced visibility
    axs[row, col].plot(tightened_bound_N_list_lw[:, i], '-', label='Lower Bound', linewidth=2)  
    axs[row, col].plot(np.ones((N+1, 1))*upb[i], '--', label='Original Upper Bound', linewidth=1.5)
    axs[row, col].plot(-np.ones((N+1, 1))*lwb[i], '--', label='Original Lower Bound', linewidth=1.5)
    axs[row, col].legend(loc='upper right', fontsize='small')
    axs[row, col].set_title(titles[i], fontsize=12)
    axs[row, col].set_xticks(range(N))

plt.delaxes(axs[2, 0]) 
plt.delaxes(axs[2, 1]) 

ax_big = fig.add_subplot(3, 1, 3)
ax_big.set_ylim(90, 130)  # Set y-axis limits to frame the data

# Use the 'step' function for a ladder-like appearance
# 'where' parameter can be 'pre', 'post', or 'mid'. It determines where the step jumps.
ax_big.step(range(len(IDM_constraint_list)), IDM_constraint_list, where='mid', label='IDM Constraint', linewidth=2, linestyle='--',color='r')

ax_big.step(range(len(tightened_bound_N_IDM_list)), tightened_bound_N_IDM_list, where='mid', label='Tightened Constraint', linewidth=2, linestyle='-')

ax_big.legend(loc='upper right', fontsize='small')
ax_big.set_title('IDM Constraint vs. Tightened Bound', fontsize=12)
ax_big.set_xticks(range(N))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout

plt.savefig('C:\\Users\\A490243\\Desktop\\Master_Thesis\\Figure\\MPC_tighten_bound.jpg', dpi=300)  # High resolution
plt.show()





