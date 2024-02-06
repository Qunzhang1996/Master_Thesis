import numpy as np
import sys
sys.path.append(r'C:\Users\A490243\Desktop\Master_Thesis\Controller')
from LTI_MPC import MPC




#! A,B here is the linearized system, and the system is constant velocity model at x=0,y=0, v=10m/s, theta=0
A=np.array([[1.0, 0.0, -0.15072664142860284, -0.0038424017710879006], 
            [0.0, 1.0, 0.13145904139180914, -0.004405572320044926], 
            [0.0, 0.0, 1.0, 0.0], 
            [0.0, 0.0, 0.0008723107079130039, 1.0]])
B=np.array([[0.0, 0.0], 
            [0.0, 0.0], 
            [0.0, 0.2], 
            [0.017606115703087965, 0.0]])
#! P0 here is converged  kf covariance
x0=np.array([[0],[0],[10],[0]])
P0 = np.array([[0.002071893043635329, -9.081755508401041e-07, 6.625835814018275e-06, 3.5644803460420476e-06],
            [-9.081755508401084e-07, 0.0020727698104820867, -1.059020050149629e-05, 2.218506297137566e-06], 
            [6.625835814018292e-06, -1.0590200501496282e-05, 0.0020706909538251504, 5.4487618678242517e-08], 
            [3.5644803460420577e-06, 2.2185062971375356e-06, 5.448761867824289e-08, 0.002071025561957967]])
nx=A.shape[0]
nu=B.shape[1]
D=np.eye(nx)  # noise matrix
Q=np.eye(nx)  # cost matrix
R=np.eye(nu)  # cost matrix  for input
process_noise=np.eye(4)  # process noise
process_noise[0,0]=0.5  # x bound is [0, 3]
process_noise[1,1]=0.01/6  # y bound is [0, 0.1]
process_noise[2,2]=1.8/6  # v bound is [0, 1.8]
process_noise[3,3]=0.05/6  # psi bound is [0, 0.05]
# stochastic mpc parameters
Possibilty=0.95  # the possibility of the tightened bound
# initial upper bound of the constraints
# Here is the upper bound of the constraints
H_up=[np.array([[1],[0],[0],[0]]),np.array([[0],[1],[0],[0]]), np.array([[0],[0],[1],[0]]), np.array([[0],[0],[0],[1]])]
upb=np.array([[5000],[1.75],[30],[3.14/6]])  #no limit to the x, y should be in [-1.75, 1.75], v should be in [0,30], psi should be in [-3.14/8,3.14/8]
# Here is the lower bound of the constraints
H_low=[np.array([[-1],[0],[0],[0]]),np.array([[0],[-1],[0],[0]]), np.array([[0],[0],[-1],[0]]), np.array([[0],[0],[0],[-1]])]
lwb=np.array([[0],[1.75],[0],[3.14/6]])  #no limit to the x, y should be in [-1.75, 1.75], v should be in [0,30], psi should be in [-3.14/8,3.14/8]
# Predict Horizon 
N=12
mpc=MPC(A,B,D,Q,R,P0,process_noise,Possibilty,N)
print(mpc.calc_linear_discrete_model())