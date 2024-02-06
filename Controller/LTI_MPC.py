""" Here is the LTI_Mpc for the Autonomous Vehicle to track the vehicle in front in the straight road. 
"""
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import sys
path_to_add='C:\\Users\\A490243\\Desktop\\Master_Thesis'
sys.path.append(path_to_add)
from Autonomous_Truck_Sim.helpers import *
from Controller.MPC_tighten_bound import MPC_tighten_bound
from util.utils import Param

class MPC:
    def __init__(self, A, B, D, Q, R, P0, process_noise, Possibility=0.99, N=12) -> None:
        
        # The number of MPC states, here include x, y, psi and v
        NUM_OF_STATES = A.shape[0]
        self.nx = NUM_OF_STATES
        # The number of MPC actions, including acc and steer_angle
        NUM_OF_ACTS = B.shape[1]
        self.nu = NUM_OF_ACTS
        self.N = N
        self.q = Q
        self.r = R
        self.P0=P0
        # self.first_state_index_ = FirstStateIndex(N)
        self.Param = Param()
        self.MPC_tighten_bound = MPC_tighten_bound(A, B, D, Q, R, P0, process_noise, Possibility) 
        # Create Opti Stack
        self.opti = Opti()

        # # Initialize opti stack
        self.x = self.opti.variable(self.nx,self.N+1)
        self.u = self.opti.variable(self.nu,self.N)
        self.x0 = self.opti.parameter(self.nx,1)  
        pass
    
    def compute_Dlqr(self):
        return self.MPC_tighten_bound.calculate_Dlqr()
    
    def IDM_constraint(self, p_leading, v_eg, d_s=10, L1=4, T_s=1.5, lambda_s=0):
        """Here is the IDM constrain for the vehicle to track the vehicle in front.
        p_leading: the x position of the vehicle in front
        v_eg: the velocity of the vehicle   
        d_s: the distance between the vehicle and the vehicle in front, default is 10m
        L1: the length of the vehicle in front, default is 4m
        T_s: the time headway, default is 1.5s
        lambda_s: is the slack variable
        
        return: the IDM constrain
        """
        return p_leading - L1 - d_s -  T_s * v_eg - lambda_s
    
    def calc_linear_discrete_model(self, v=10, phi=0, delta=0):
        """
        calc linear and discrete time dynamic model.
        :param v: speed: v_bar, when do the acc, the default is 10m/s
        :param phi: angle of vehicle: phi_bar, when do the acc, the default is 0
        :param delta: steering angle: delta_bar, when do the acc, the default is 0
        :return: A, B, C
        """

        A = DM([[1.0, 0.0, self.Param.dt * np.cos(phi), - self.Param.dt * v * np.sin(phi)],
                [0.0, 1.0, self.Param.dt * np.sin(phi), self.Param.dt * v * np.cos(phi)],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, self.Param.dt * np.tan(delta) / self.Param.WB, 1.0]])

        B = DM([[0.0, 0.0],
                [0.0, 0.0],
                [self.Param.dt, 0.0],
                [0.0, self.Param.dt * v / (self.Param.WB * np.cos(delta) ** 2)]])

        C = DM([self.Param.dt * v * np.sin(phi) * phi,
                -self.Param.dt * v * np.cos(phi) * phi,
                0.0,
                -self.Param.dt * v * delta / (self.Param.WB * np.cos(delta) ** 2)])

        return A, B, C
    
    def setStateEqconstraints(self, v=10, phi=0, delta=0):
        # Here is the jocobian A and B matrix of the vehicle kinematic model
         
        for i in range(self.N):
            A, B, C = self.calc_linear_discrete_model(v, phi, delta) 
            self.opti.subject_to(self.x[:,i+1] == A @ self.x[:,i] + B @ self.u[:,i] + C)
        self.opti.subject_to(self.x[:,0] == self.x0)
        
        
        
    def setInEqConstraints_val(self, H_up=None, upb=None, H_low=None, lwb=None):
        # Here is the tightened bound of the constraints
        if H_up is None:
            self.H_up=[np.array([[1],[0],[0],[0]]),np.array([[0],[1],[0],[0]]), np.array([[0],[0],[1],[0]]), np.array([[0],[0],[0],[1]])]
            self.upb=np.array([[5000],[500],[30],[3.14/8]])  #no limit to the x, y should be in [-1.75, 1.75], v should be in [0,30], psi should be in [-3.14/8,3.14/8]
            # Here is the lower bound of the constraints
            self.H_low=[np.array([[-1],[0],[0],[0]]),np.array([[0],[-1],[0],[0]]), np.array([[0],[0],[-1],[0]]), np.array([[0],[0],[0],[-1]])]
            self.lwb=np.array([[0],[500],[0],[3.14/8]])  #no limit to the x, y should be in [-1.75, 1.75], v should be in [0,30], psi should be in [-3.14/8,3.14/8]
        else:
            self.H_up = H_up
            self.upb = upb
            self.H_low = H_low
            self.lwb = lwb
    
    def setInEqConstraints(self):
        self.opti.subject_to(self.x[0,:] <= self.IDM_constraint(self.p_leading))
        # here A B matrix is constant, so the tightened bound is constant
        tightened_bound_N_list_up=self.MPC_tighten_bound.tighten_bound_N(self.P0,self.H_up,self.upb,self.N,1)
        tightened_bound_N_list_lw=self.MPC_tighten_bound.tighten_bound_N(self.P0,self.H_low,self.lwb,self.N,0)
        # using for loop to tighten the state constraints, [x, y ,v, psi]
        for i in range(self.N):
            self.opti.subject_to(self.x[0,i] >= tightened_bound_N_list_lw[i][0])
            self.opti.subject_to(self.x[1,i] >= tightened_bound_N_list_lw[i][1])
            self.opti.subject_to(self.x[2,i] >= tightened_bound_N_list_lw[i][2])
            self.opti.subject_to(self.x[3,i] >= tightened_bound_N_list_lw[i][3])
            self.opti.subject_to(self.x[0,i] <= tightened_bound_N_list_up[i][0])
            self.opti.subject_to(self.x[1,i] <= tightened_bound_N_list_up[i][1])
            self.opti.subject_to(self.x[2,i] <= tightened_bound_N_list_up[i][2])
            self.opti.subject_to(self.x[3,i] <= tightened_bound_N_list_up[i][3])
        # constraints for the input [-3.14/180,-0.7*9.81],[3.14/180,0.05*9.81]   steer angle and acc
        self.opti.subject_to(self.u[0,:] >= -3.14/180)
        self.opti.subject_to(self.u[0,:] <= 3.14/180)
        self.opti.subject_to(self.u[1,:] >= -0.7*9.81)
        self.opti.subject_to(self.u[1,:] <= 0.05*9.81)
    
    
    
    def Solve(self, state, vehicle_state_pred):
        '''
        state: the state of the vehicle
        vehicle_state_pred: the state of the vehicle in front
        '''
        self.p_leading = vehicle_state_pred[0]  
        
        # define the optimization variables 
        x = MX.sym('x', self.num_of_x_)
        
        #define the cost function
        cost = 0
        # TBD: the cost function is not clear
        
        # initial variables
        x_ = [0] * self.num_of_x_
        print(x_)












