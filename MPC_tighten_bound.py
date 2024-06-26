from casadi import *
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.stats import norm
  
class MPC_tighten_bound:
    """Calculates the stochastic MPC tightened bound."""
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
        self.initial_P0 = P0.copy()  # Keep a copy of the initial P0
        self.P0 = P0
        self.process_noise = process_noise
        self.Possibilty = Possibilty
        self.nx = A.shape[0]
        self.nu = B.shape[1]

    def reset_P0(self):
        """Resets the covariance matrix to its initial value."""
        self.P0 = self.initial_P0.copy()

    def calculate_Dlqr(self):
        P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        K = -np.linalg.inv(self.B.T @ P @ self.B + self.R) @ (self.B.T @ P @ self.A)
        return P, K

    def calculate_system_dynamics(self, X, U):
        assert X.shape == (self.nx, 1)
        assert U.shape == (self.nu, 1)
        X_next = self.A @ X + self.B @ U
        return X_next

    def calculate_P_next(self, K):
        P_next = (self.A + self.B @ K) @ self.P0 @ (self.A + self.B @ K).T + self.process_noise
        self.P0 = P_next
        return P_next

    def calculate_P_next_N(self, K, N):
        P_next_N_list = [self.calculate_P_next(K) for _ in range(N)]
        return self.P0, P_next_N_list

    def tightened_bound(self, sigma, h, b, upper_boundary):
        tighten_bound = []
        for i in range(len(h)):
            temp = np.sqrt(h[i].T @ sigma @ h[i]) * norm.ppf(self.Possibilty)
            tighten_bound.append(b[i] - temp if upper_boundary else -b[i] + temp)
        return np.array(tighten_bound).reshape(-1, 1)
    
    
    def tighten_bound_U(self, sigma, h, b, upper_boundary):
        tightened_bound_U_list = []
        _, K = self.calculate_Dlqr()
        self.reset_P0()
        for i in range(len(h)):
            # print("sigma:", sigma)
            # print("h[i]:", h[i])
            # print("K:", K)
            temp = np.sqrt(h[i].T @ K @ sigma @ K.T @ h[i]) * norm.ppf(self.Possibilty)
            # print("temp:", temp)
            # exit()
            tightened_bound_U_list.append(b[i] - temp if upper_boundary else -b[i] + temp)
        
        return np.array(tightened_bound_U_list).reshape(-1, 1)
    
    

    def tighten_bound_N(self, sigma, h, b, N, upper_boundary):
        tightened_bound_N_list = []
        _, K = self.calculate_Dlqr()
        self.reset_P0()  # Reset P0 before the loop
        _, sigma_list = self.calculate_P_next_N(K, N)
        for s in sigma_list:
            tighten_bound = self.tightened_bound(s, h, b, upper_boundary)
            tightened_bound_N_list.append(tighten_bound)
            
        
        # add b at the beginning of the list
        tightened_bound_N_list.insert(0, b) if upper_boundary else tightened_bound_N_list.insert(0, -b)
        return np.array(tightened_bound_N_list).reshape(N+1, -1)
    
    
    def tighten_bound_N_U(self, sigma, h, b, N, upper_boundary):
        tightened_bound_N_U_list = []
        _, K = self.calculate_Dlqr()
        self.reset_P0()
        _, sigma_list = self.calculate_P_next_N(K, N)
        for s in sigma_list:
            tighten_bound = self.tighten_bound_U(s, h, b, upper_boundary)
            tightened_bound_N_U_list.append(tighten_bound)
        tightened_bound_N_U_list.insert(0, b) if upper_boundary else tightened_bound_N_U_list.insert(0, -b)
        # print(tightened_bound_N_U_list)
        # exit()
        return np.array(tightened_bound_N_U_list).reshape(N+1, -1)
    
    
    
    def tighten_bound_N_IDM(self, IDM_constraint, N):
        _, K = self.calculate_Dlqr()
        self.reset_P0()  # Reset P0 before the loop
        _, sigma_list = self.calculate_P_next_N(K, N)
        original_constraint = IDM_constraint # contains the 13X1 vector
        tightened_bound_N_IDM_list = []
        tightened_bound_N_IDM_list.append(original_constraint[0])
        temp_list = []
        temp_list.append(0)
        for i in range(0,N):
            current_sigma = sigma_list[i][0,0] # Only the first element: "x" of the sigma matrix is used
            temp = np.sqrt(current_sigma) * norm.ppf(self.Possibilty)
            tightened_bound_N_IDM_list.append(original_constraint[i+1] - temp)
            temp_list.append(temp)
            
        return np.array(tightened_bound_N_IDM_list).reshape(N+1, -1), temp_list
    
    def tighten_bound_N_laneChange(self, LC_constraint, N):
        _, K = self.calculate_Dlqr()
        self.reset_P0()  # Reset P0 before the loop
        _, sigma_list = self.calculate_P_next_N(K, N)
        original_constraint = LC_constraint # contains the 13X1 vector
        tightened_bound_N_LC_list = []
        tightened_bound_N_LC_list.append(original_constraint[0])
        temp_list = []
        temp_list.append(0)
        for i in range(0,N):
            current_sigma = sigma_list[i][1,1] # Only the second element: "y" of the sigma matrix is used
            temp = np.sqrt(current_sigma) * norm.ppf(self.Possibilty)
            tightened_bound_N_LC_list.append(original_constraint[i+1] + temp)
            temp_list.append(temp)
            
        return np.array(tightened_bound_N_LC_list).reshape(N+1, -1), temp_list
    #! return the tightened temp_y for the y axis   DM(1,N+1)
    def getXtemp(self, N):
        '''
        return the tightened temp_y for the y axis
        temp_y = DM(1,N+1)
        '''
        _, K = self.calculate_Dlqr()
        self.reset_P0()  # Reset P0 before the loop
        _, sigma_list = self.calculate_P_next_N(K, N)
        temp_x = DM(1,N+1)
        for i in range(0,N):
            current_sigma = sigma_list[i][0,0]
            temp = np.sqrt(current_sigma) * norm.ppf(self.Possibilty)
            temp_x[0,i+1] = temp
        return temp_x
    
    #! return the tightened temp_y for the y axis   DM(1,N+1)
    def getYtemp(self, N):
        '''
        return the tightened temp_y for the y axis
        temp_y = DM(1,N+1)
        '''
        _, K = self.calculate_Dlqr()
        self.reset_P0()  # Reset P0 before the loop
        _, sigma_list = self.calculate_P_next_N(K, N)
        temp_y = DM(1,N+1)
        for i in range(0,N):
            current_sigma = sigma_list[i][1,1]
            temp = np.sqrt(current_sigma) * norm.ppf(self.Possibilty)
            temp_y[0,i+1] = temp
        return temp_y
        
    
    def tighten_bound_N_y_upper(self, y_constrain, N):
        _, K = self.calculate_Dlqr()
        self.reset_P0()  # Reset P0 before the loop
        _, sigma_list = self.calculate_P_next_N(K, N)
        tightened_bound_N_y_list = []
        tightened_bound_N_y_list.append(y_constrain)
        tightened_bound_N_y_list = []
        tightened_bound_N_y_list.append(y_constrain[0])
        for i in range(0,N):
            current_sigma = sigma_list[i][1,1] # Only the second element: "y" of the sigma matrix is used
            temp = np.sqrt(current_sigma) * norm.ppf(self.Possibilty)
            tightened_bound_N_y_list.append(y_constrain[i+1] - temp)
        return np.array(tightened_bound_N_y_list).reshape(N+1, -1)
    
    def tighten_bound_N_y_lower(self, y_constrain, N):
        _, K = self.calculate_Dlqr()
        self.reset_P0()
        _, sigma_list = self.calculate_P_next_N(K, N)
        tightened_bound_N_y_list = []
        tightened_bound_N_y_list.append(y_constrain[0])
        for i in range(0,N):
            current_sigma = sigma_list[i][1,1]
            temp = np.sqrt(current_sigma) * norm.ppf(self.Possibilty)
            tightened_bound_N_y_list.append(-y_constrain[i+1] + temp)
        return np.array(tightened_bound_N_y_list).reshape(N+1, -1)
    
    
    
    def tighten_bound_N_vel_diff(self, vel_diff_constrain, N):
        _, K = self.calculate_Dlqr()
        self.reset_P0()  # Reset P0 before the loop
        _, sigma_list = self.calculate_P_next_N(K, N)
        tightened_bound_N_vel_diff_list = []
        tightened_bound_N_vel_diff_list.append(vel_diff_constrain[0])
        for i in range(0,N):
            current_sigma = sigma_list[i][2,2] # Only the second element: "v" of the sigma matrix is used
            temp = np.sqrt(current_sigma) * norm.ppf(self.Possibilty)
            tightened_bound_N_vel_diff_list.append(vel_diff_constrain[i+1] - temp)
            
        return np.array(tightened_bound_N_vel_diff_list).reshape(N+1, -1)
    
    
        

    @property
    def get_corvariance(self):
        return self.P0

