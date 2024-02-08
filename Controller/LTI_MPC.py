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
from vehicleModel.vehicle_model import car_VehicleModel
from util.utils import Param
class MPC:
    def __init__(self, vehicle, Q, R, P0, process_noise, Possibility=0.95, N=12) -> None:
        # The number of MPC states, here include x, y, psi and v
        NUM_OF_STATES = 4
        self.nx = NUM_OF_STATES
        # The number of MPC actions, including acc and steer_angle
        NUM_OF_ACTS = 2
        self.nu = NUM_OF_ACTS
        
        self.vehicle = vehicle
        self.nx, self.nu, self.nrefx, self.nrefu = self.vehicle.getSystemDim()
        self.Q = Q
        self.R = R
        self.P0 = P0
        self.process_noise = process_noise
        self.Possibility = Possibility
        self.N = N
        self.Param = Param()
        
        # Create Opti Stack
        self.opti = Opti()
        
        # Initialize opti stack
        self.x = self.opti.variable(self.nx, self.N + 1)
        self.u = self.opti.variable(self.nu, self.N)
        self.refx = self.opti.parameter(self.nrefx, self.N + 1)
        self.refu = self.opti.parameter(self.nrefu, self.N)
        self.x0 = self.opti.parameter(self.nx, 1)
        
        # IDM leading vehicle position parameter
        self.p_leading = self.opti.parameter(1)
    
    def compute_Dlqr(self):
        return self.MPC_tighten_bound.calculate_Dlqr()
    
    def IDM_constraint(self, p_leading, v_eg, d_s=10, L1=4, T_s=1.5, lambda_s=0):
        """
        IDM constraint for tracking the vehicle in front.
        """
        return p_leading - L1 - d_s - T_s * v_eg - lambda_s
    
    def calc_linear_discrete_model(self, v=10, phi=0, delta=0):
        """
        Calculate linear and discrete time dynamic model.
        """
        A = DM([[1.0, 0.0, self.Param.dt * np.cos(phi), -self.Param.dt * v * np.sin(phi)],
                [0.0, 1.0, self.Param.dt * np.sin(phi), self.Param.dt * v * np.cos(phi)],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, self.Param.dt * np.tan(delta) / self.Param.WB, 1.0]])
        
        B = DM([[0.0, 0.0],
                [0.0, 0.0],
                [0.0, self.Param.dt],
                [self.Param.dt * v / (self.Param.WB * np.cos(delta) ** 2), 0.0]])
        
        
        C = DM([self.Param.dt * v * np.sin(phi) * phi,
                -self.Param.dt * v * np.cos(phi) * phi,
                0.0,
                -self.Param.dt * v * delta / (self.Param.WB * np.cos(delta) ** 2)])
        
        return A, B, C
    
    def setStateEqconstraints(self, v=10, phi=0, delta=0):
        """
        Set state equation constraints.
        """
        for i in range(self.N):
            A, B, C = self.calc_linear_discrete_model(v, phi, delta)  
            self.A, self.B, self.C = A, B, C
            self.opti.subject_to(self.x[:, i+1] == A @ self.x[:, i] + B @ self.u[:, i] + C)
        self.opti.subject_to(self.x[:, 0] == self.x0)
    
    def setInEqConstraints_val(self, H_up=None, upb=None, H_low=None, lwb=None):
        """
        Set inequality constraints values.
        """
        # Default or custom constraints
        self.H_up = H_up if H_up is not None else [np.array([[1], [0], [0], [0]]), np.array([[0], [1], [0], [0]]), np.array([[0], [0], [1], [0]]), np.array([[0], [0], [0], [1]])]
        self.upb = upb if upb is not None else np.array([[5000], [5000], [30], [3.14/8]])
        self.H_low = H_low if H_low is not None else [np.array([[-1], [0], [0], [0]]), np.array([[0], [-1], [0], [0]]), np.array([[0], [0], [-1], [0]]), np.array([[0], [0], [0], [-1]])]
        self.lwb = lwb if lwb is not None else np.array([[5000], [5000], [0], [3.14/8]])
    
    def setInEqConstraints(self):
        """
        Set inequality constraints.
        """
        v, phi, delta = 10, 0, 0  # Default values; adjust as necessary
        A, B, _ = self.calc_linear_discrete_model(v, phi, delta)
        D = np.eye(self.nx)  # Noise matrix
        
        self.setInEqConstraints_val()  # Set tightened bounds

        
        
        self.MPC_tighten_bound = MPC_tighten_bound(A, B, D, self.Q, self.R, self.P0, self.process_noise, self.Possibility)
                
        # Example tightened bound application (adjust according to actual implementation)
        tightened_bound_N_list_up = self.MPC_tighten_bound.tighten_bound_N(self.P0, self.H_up, self.upb, self.N, 1)
        tightened_bound_N_list_lw = self.MPC_tighten_bound.tighten_bound_N(self.P0, self.H_low, self.lwb, self.N, 0)
        
        # the tightened bound (up/lw) is N+1 X NUM_OF_STATES  [x, y, v, psi] 
        # according to the new bounded constraints set the constraints
        for i in range(self.N):
            for j in range(self.nx):
                self.opti.subject_to(self.x[j, i] <= tightened_bound_N_list_up[i][j].item())
            self.opti.subject_to(self.x[:, i] <= DM(tightened_bound_N_list_up[i].reshape(-1, 1)))
            self.opti.subject_to(self.x[:, i] >= DM(tightened_bound_N_list_lw[i].reshape(-1, 1)))
            
        # set the constraints for the input  [-3.14/180,-0.7*9.81],[3.14/180,0.05*9.81]
        self.opti.subject_to(self.u[0, :] >= -3.14 / 180)
        self.opti.subject_to(self.u[0, :] <= 3.14 / 180)
        self.opti.subject_to(self.u[1, :] >= -0.7 * 9.81)
        self.opti.subject_to(self.u[1, :] <= 0.5 * 9.81)
        
        
        # Set the IDM constraint
        self.opti.subject_to(self.x[0, :] <= self.IDM_constraint(self.p_leading, self.x[2, :]))
        
    def setCost(self):
        """
        Set cost function for the optimization problem.
        """
        L, Lf = self.vehicle.getCost()
        self.opti.minimize(getTotalCost(L, Lf, self.x, self.u, self.refx, self.refu, self.N))
    
    def setController(self):
        """
        Set constraints and cost function for the MPC controller.
        """
        self.setStateEqconstraints()
        self.setInEqConstraints()
        self.setCost()
    
    def solve(self, x0, ref_trajectory, ref_control, p_leading):
        """
        Solve the MPC problem.
        """
        # Set the initial condition and reference trajectories
        
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.refx, ref_trajectory)
        self.opti.set_value(self.refu, ref_control)
        self.opti.set_value(self.p_leading, p_leading)
        
        # Solver options
        opts = {"ipopt": {"print_level": 0, "tol": 1e-8}, "print_time": 0}
        self.opti.solver("ipopt", opts)
        # print("IDM_constraint", self.IDM_constraint(p_leading, x0[2]))
        
        try:
            sol = self.opti.solve()
            u_opt = sol.value(self.u)
            x_opt = sol.value(self.x)
            return u_opt, x_opt
        except Exception as e:
            print(f"An error occurred: {e}")
            return None, None
        








