"""Here, tighten the constraints for the system
"""
from casadi import *
import numpy as np
from scipy.linalg import solve_discrete_are,sqrtm
from scipy.stats import norm
# using np to calculate the LQR gain
def calculate_Dlqr(A,B,Q,R):
    """return LQR gain matrix

    Args:
        A B (np.array): state dynamics
        Q R (np.array): cost matrix

    Returns:
        _type_: _description_
    """
    P=solve_discrete_are(A,B,Q,R)
    temp=np.linalg.inv(B.T@P@B+R)
    k=-temp@(B.T@P@A)
    return P,k

#calculate the system dynamics: X_next=A*X+B*U
def calculate_system_dynamics(A,B,X,U):
    """calculate the system dynamics: X_next=A*X+B*U

    Args:
        A B (np.array): state dynamics
        X U (np.array): state and control input

    Returns:
        _type_: _description_
    """
    #check the dimension of x is 4X1 and u is 2X1
    assert A.shape==(4,4)
    assert B.shape==(4,2)
    assert X.shape==(4,1)
    assert U.shape==(2,1)
    X_next=A@X+B@U
    return X_next

# calculate the system dynamics for N steps
def calculate_system_dynamics_N(A,B,X,U,N):
    """calculate the system dynamics for N steps

    Args:
        A B (np.array): state dynamics
        X U (np.array): state and control input
        N (int): predict horizon

    Returns:
        _type_: _description_
    """
    X_next=calculate_system_dynamics(A,B,X,U)
    for i in range(N-1):
        X_next=calculate_system_dynamics(A,B,X_next,U)
    return X_next


def calculate_P_next(A,B,K,P0,process_noise):
    """calculate the P_next

    Args:
        A B (np.array): state dynamics
        K (np.array): dlqr gain
        P0 (np.array): covariance
        process_noise (np.array): process noise

    Returns:
        _type_: _description_
    """
    assert A.shape==(4,4)
    assert B.shape==(4,2)
    assert P0.shape==(4,4)
    assert process_noise.shape==(4,4)
    P_next=(A+B@K)@P0@(A+B@K).T+process_noise
    return P_next

#calculate the P_next for N steps
def calculate_P_next_N(A,B,K,P0,process_noise,N):
    """calculate the P_next for N steps

    Args:
        A B (np.array): state dynamics
        K (np.array): dlqr gain
        P0 (np.array): covariance
        process_noise (np.array): process noise
        N (int): predict horizon

    Returns:
        _type_: _description_
    """
    assert A.shape==(4,4)
    assert B.shape==(4,2)
    assert P0.shape==(4,4)
    assert process_noise.shape==(4,4)
    P_next=calculate_P_next(A,B,K,P0,process_noise)
    for i in range(N-1):
        P_next=calculate_P_next(A,B,K,P_next,process_noise)
    return P_next

# calculate the new upper bound for the constraints
def tightened_bound(sigma,h,b):
    """with φ^-1 the inverse cumulative distribution function of the standard normal distribution

    Args:
        sigma (np.array): corvariance
        h,b (np.array): hx<=b
    """
    assert sigma.shape==(4,4)
    # calculate the tightening term
    temp=sqrtm(h.T@sigma@h)@norm.ppf(Possibilty)
    tighten_bound=b-temp
    return tighten_bound

#calculate the Nth tighten bound
def tighten_bound_N(sigma,h,b,N):
    """calculate the Nth tighten bound

    Args:
        sigma (np.array): corvariance
        h,b (np.array): hx<=b
        N (int): predict horizon
    """
    # calculate the tightening term
    for i in range(N):
        sigma=calculate_P_next(A,B,K,sigma,process_noise)
        # print("eig value of sigma is:",np.linalg.eigvals(sigma))
    tighten_bound=tightened_bound(sigma,h,b)
    return tighten_bound




nx=4
nu=2
#! P0 here is converged  kf covariance
#! A,B here is the linearized system, and the system is constant velocity model at x=0,y=0, v=10m/s, theta=0
x0=np.array([[0],[0],[10],[0]])
P0 = np.array([[0.002071893043635329, -9.081755508401041e-07, 6.625835814018275e-06, 3.5644803460420476e-06],
            [-9.081755508401084e-07, 0.0020727698104820867, -1.059020050149629e-05, 2.218506297137566e-06], 
            [6.625835814018292e-06, -1.0590200501496282e-05, 0.0020706909538251504, 5.4487618678242517e-08], 
            [3.5644803460420577e-06, 2.2185062971375356e-06, 5.448761867824289e-08, 0.002071025561957967]])

print("check the eig value of P0", numpy.linalg.eigvals(P0))
A=np.array([[1.0, 0.0, -0.15072664142860284, -0.0038424017710879006], 
            [0.0, 1.0, 0.13145904139180914, -0.004405572320044926], 
            [0.0, 0.0, 1.0, 0.0], 
            [0.0, 0.0, 0.0008723107079130039, 1.0]])
B=np.array([[0.0, 0.0], 
            [0.0, 0.0], 
            [0.0, 0.2], 
            [0.017606115703087965, 0.0]])
D=np.eye(nx)  # noise matrix
Q=np.eye(nx)  # cost matrix
R=np.eye(nu)  # cost matrix  for input
process_noise=np.eye(4)*0.05**2  # process noise
# noise here should check with erik
process_noise[0,0]=0.5  # x bound is [0, 3]
# process_noise[1,1]=0.01  # y bound is [0, 0.1]
process_noise[2,2]=1.8/6  # v bound is [0, 1.8]
# process_noise[3,3]=0.01  # psi bound is [0, 0.05]
# Predict Horizon
N=8
# calcuate the dlqr gain
_,K=calculate_Dlqr(A,B,Q,R)


# stochastic mpc parameters
Possibilty=np.ones([4,1])*0.8
#calculate the X_next_N
X_next_N=calculate_system_dynamics_N(A,B,x0,np.zeros([2,1]),N)
# initial upper bound of the constraints
h=np.eye(nx)
b=np.array([[5000],[1.75],[30],[3.14/8]])  #no limit to the x, y should be in [-1.75, 1.75], v should be in [0,30], psi should be in [-3.14/8,3.14/8]
#tighten the bound
tighten_bound=tighten_bound_N(P0,h,b,N)
print("the bound before tighten is:\n",b)
print("tightened bound is:\n", tighten_bound_N(P0,h,b,N))