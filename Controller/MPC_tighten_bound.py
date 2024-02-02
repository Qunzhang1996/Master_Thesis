"""Here, tighten the constraints for the system
"""
from casadi import *
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.stats import norm
class StateIndex:
    """Used to return the index of the state
       Args:
        A, B: system dynamic matrices
        N: the prediction horizon

    """
    def __init__(self, A, B, N):
        self.s = [0] * np.shape(A)[0]
        self.v = [0] * np.shape(B)[1]
        self.s[0] = 0
        self.v[0] = np.shape(A)[0] * N
        for i in range(np.shape(A)[0] - 1):
            self.s[i + 1] = self.s[i] + N
        for i in range(np.shape(B)[1] - 1):
            self.v[i + 1] = self.v[i] + N - 1   
    
class MPC_tighten_bound:
    """This class is used to calculate the stochastic mpc tightened bound
        Args:
            A,B,D: SYSTEM PARAMETERS
            Q,R:   Cost function parameters
            P0:    Covariance from kf
            Process_noise: the diff between CARLA env and the nominal model
            Possibilty: the possibility of the tightened bound
            nx,nu: the dimension of the state and input
    """
    def __init__(self, A, B, D, Q, R, P0, process_noise, Possibilty):
        self.A = A
        self.B = B
        self.D = D
        self.Q = Q
        self.R = R
        self.P0 = P0
        self.process_noise = process_noise
        self.Possibilty = Possibilty
        self.nx = A.shape[0]
        self.nu = B.shape[1]
    def calculate_Dlqr(self):
        P=solve_discrete_are(self.A,self.B,self.Q,self.R)
        temp=np.linalg.inv(self.B.T@P@self.B+self.R)
        k=-temp@(self.B.T@P@self.A)
        return P,k
    
    def calculate_system_dynamics(self,X,U):
        #check the dimension of x is 4X1 and u is 2X1
        assert self.A.shape==(self.nx,self.nx)
        assert self.B.shape==(self.nx,self.nu)
        assert X.shape==(self.nx,1)
        assert U.shape==(self.nx,1)
        X_next=self.A@X+self.B@U
        return X_next
    
    def calculate_system_dynamics_N(self,X,U):
        X_next=self.calculate_system_dynamics(X,U)
        for i in range(self.N-1):
            X_next=self.calculate_system_dynamics(X_next,U)
        return X_next
    
    def calculate_P_next(self,K):
        assert self.A.shape==(self.nx,self.nx)
        assert self.B.shape==(self.nx,self.nu)
        assert self.P0.shape==(self.nx,self.nx)
        assert self.process_noise.shape==(self.nx,self.nx)
        P_next=(self.A+self.B@K)@self.P0@(self.A+self.B@K).T+self.process_noise
        self.P0=P_next
        return P_next
    
    #calculate the Nth P_next
    def calculate_P_next_N(self,K,N):
        assert self.A.shape==(self.nx,self.nx)
        assert self.B.shape==(self.nx,self.nu)
        assert self.P0.shape==(self.nx,self.nx)
        assert self.process_noise.shape==(self.nx,self.nx)
        P_next_N_list=[]
        for i in range(N):
            P_next_N=self.calculate_P_next(K)
            P_next_N_list.append(P_next_N)
        return P_next_N,P_next_N_list
    
    def tightened_bound(self,sigma,h,b,upper_boundary):
        assert sigma.shape==(4,4)
        tighten_bound=[]
    # calculate the tightening term
        for i in range(len(h)):
            temp=np.sqrt(h[i].T@sigma@h[i])*norm.ppf(self.Possibilty)
            if upper_boundary:
                tighten_bound.append((b[i]-temp))
            else:
                tighten_bound.append((-b[i]+temp))
        # to 4X1 np array
        tighten_bound=np.array(tighten_bound).reshape(-1,1)
        return tighten_bound
    
    #calculate the Nth tighten bound
    def tighten_bound_N(self,sigma,h,b,N,upper_boundary):
        """calculate the Nth tighten bound

        Args:
            sigma (np.array): corvariance
            h,b (np.array): hx<=b
            N (int): predict horizon
        """
        tightened_bound_N_list=[]
        _,K=self.calculate_Dlqr()
        sigma,sigma_list=self.calculate_P_next_N(K,N)
        for s in sigma_list:
            tighten_bound=self.tightened_bound(s,h,b,upper_boundary)
            tightened_bound_N_list.append(tighten_bound)
        tightened_bound_N_list=np.array(tightened_bound_N_list).reshape(N,4)
        return tighten_bound,tightened_bound_N_list

    @property
    def get_corvariance(self):
        return self.P0
        
                         
