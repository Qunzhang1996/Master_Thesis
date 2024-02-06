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

    def tighten_bound_N(self, sigma, h, b, N, upper_boundary):
        tightened_bound_N_list = []
        _, K = self.calculate_Dlqr()
        self.reset_P0()  # Reset P0 before the loop
        sigma, sigma_list = self.calculate_P_next_N(K, N)
        for s in sigma_list:
            tighten_bound = self.tightened_bound(s, h, b, upper_boundary)
            tightened_bound_N_list.append(tighten_bound)
            
        
        # add b at the beginning of the list
        tightened_bound_N_list.insert(0, b) if upper_boundary else tightened_bound_N_list.insert(0, -b)
        return np.array(tightened_bound_N_list).reshape(N+1, -1)

    @property
    def get_corvariance(self):
        return self.P0

