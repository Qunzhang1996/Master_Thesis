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
from util.utils import *
class MPC:
    """
    ██████╗  ██████╗ ██████╗      ██████╗ ██╗     ███████╗███████╗███████╗    ███╗   ███╗██████╗  ██████╗            
    ██╔════╝ ██╔═══██╗██╔══██╗    ██╔══██╗██║     ██╔════╝██╔════╝██╔════╝    ████╗ ████║██╔══██╗██╔════╝            
    ██║  ███╗██║   ██║██║  ██║    ██████╔╝██║     █████╗  ███████╗███████╗    ██╔████╔██║██████╔╝██║                 
    ██║   ██║██║   ██║██║  ██║    ██╔══██╗██║     ██╔══╝  ╚════██║╚════██║    ██║╚██╔╝██║██╔═══╝ ██║                 
    ╚██████╔╝╚██████╔╝██████╔╝    ██████╔╝███████╗███████╗███████║███████║    ██║ ╚═╝ ██║██║     ╚██████╗            
     ╚═════╝  ╚═════╝ ╚═════╝     ╚═════╝ ╚══════╝╚══════╝╚══════╝╚══════╝    ╚═╝     ╚═╝╚═╝      ╚═════╝                                                                                                                                                                                                            
    """

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
        # ref val for the vehicle matrix
        self.v_ref = 15
        self.phi_ref = 0
        self.delta_ref = 0
        
        # Create Opti Stack
        self.opti = Opti()
        # Initialize opti stack
        self.x = self.opti.variable(self.nx, self.N + 1)
        self.u = self.opti.variable(self.nu, self.N)
        self.refx = self.opti.parameter(self.nrefx, self.N + 1)
        self.refu = self.opti.parameter(self.nrefu, self.N)
        self.x0 = self.opti.parameter(self.nx, 1)
        # slack variable for the IDM constraint
        self.lambda_s= self.opti.variable(1,self.N + 1)
        # slack variable for y 
        self.slack_y = self.opti.variable(1,self.N + 1)
        # ! here is a test for sigma
        self.sigma = self.opti.variable(1,self.N + 1)

        # IDM leading vehicle position parameter
        self.p_leading = self.opti.parameter(2,1)
        # set car velocity 
        self.leading_velocity = self.opti.parameter(1)
        self.vel_diff = self.opti.parameter(1)
    
    def compute_Dlqr(self):
        return self.MPC_tighten_bound.calculate_Dlqr()
    
    def IDM_constraint(self, p_leading_x, v_eg, d_s=1, L1=8.4, T_s=1, lambda_s=0):
        """
        IDM constraint for tracking the vehicle in front.
        """
        return p_leading_x - L1 - d_s - T_s * v_eg - lambda_s
    
    def calc_linear_discrete_model(self, v=15, phi=0, delta=0):
        """
        Calculate linear and discrete time dynamic model.
        """
        # ! REMEMBER TO CHANGE THE VEHICLE MODEL
        A_d = DM([[1.0, 0.0, self.Param.dt * np.cos(phi), -self.Param.dt * v * np.sin(phi)],
                [0.0, 1.0, self.Param.dt * np.sin(phi), self.Param.dt * v * np.cos(phi)],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, self.Param.dt * np.tan(delta) / self.Param.WB, 1.0]])
        
        B_d = DM([[0.0, 0.0],
                [0.0, 0.0],
                [0.0, self.Param.dt],
                [self.Param.dt * v / (self.Param.WB * np.cos(delta) ** 2), 0.0]])
        
        
        g_d = DM([self.Param.dt * v * np.sin(phi) * phi,
                -self.Param.dt * v * np.cos(phi) * phi,
                0.0,
                -self.Param.dt * v * delta / (self.Param.WB * np.cos(delta) ** 2)])
        
        return A_d, B_d, g_d
    
    def setStateEqconstraints(self, v=15, phi=0, delta=0):
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
        v, phi, delta = self.v_ref, self.phi_ref, self.delta_ref  # Default values; adjust as necessary
        A, B, _ = self.calc_linear_discrete_model(v, phi, delta)
        D = np.eye(self.nx)  # Noise matrix
        
        self.setInEqConstraints_val()  # Set tightened bounds

        self.MPC_tighten_bound = MPC_tighten_bound(A, B, D, self.Q, self.R, self.P0, self.process_noise, self.Possibility)
        # Set the IDM constraint
        self.IDM_constraint_list = []
        for i in range(self.N+1):
            self.IDM_constraint_list.append(self.IDM_constraint(self.p_leading[0] + self.leading_velocity*self.Param.dt*i, 
                                                                self.x[2, i],self.lambda_s[0,i]))
        #! Set the y constraint 13
        self.y_upper = [143.318146+1.75-2.54/2] * (self.N+1)
        self.y_lower = [-143.318146+1.75-2.54/2] * (self.N+1)
        for i in range(self.N+1):
            self.y_upper[i] += self.slack_y[0,i]
            self.y_lower[i] -= self.slack_y[0,i]
        
        # Set the vel_diff constraint 
        vel_diff_constrain_list = [self.vel_diff] * (self.N+1)

        # Example tightened bound application (adjust according to actual implementation)
        self.tightened_bound_N_list_up = self.MPC_tighten_bound.tighten_bound_N(self.P0, self.H_up, self.upb, self.N, 1)
        self.tightened_bound_N_list_lw = self.MPC_tighten_bound.tighten_bound_N(self.P0, self.H_low, self.lwb, self.N, 0)
        self.tightened_bound_N_IDM_list, self.tighten_val_list = self.MPC_tighten_bound.tighten_bound_N_IDM(self.IDM_constraint_list, self.N)
        self.tightened_bound_N_vel_diff_list = self.MPC_tighten_bound.tighten_bound_N_vel_diff(vel_diff_constrain_list, self.N)
        self.tightened_bound_N_y_upper_list = self.MPC_tighten_bound.tighten_bound_N_y_upper(self.y_upper, self.N)
        self.tightened_bound_N_y_lower_list = self.MPC_tighten_bound.tighten_bound_N_y_lower(self.y_lower, self.N)
        # print(f"tighten_val_list: {self.tighten_val_list}")
        # exit()

        # the tightened bound (up/lw) is N+1 X NUM_OF_STATES  [x, y, v, psi] 
        # according to the new bounded constraints set the constraints
        for i in range(self.N+1):
            self.opti.subject_to(self.x[:, i] <= DM(self.tightened_bound_N_list_up[i].reshape(-1, 1)))
            self.opti.subject_to(self.x[:, i] >= DM(self.tightened_bound_N_list_lw[i].reshape(-1, 1)))
            # ! Set the IDM constraint
            self.opti.subject_to(self.x[0, i] - self.tightened_bound_N_IDM_list[i].item() <= 0)
            
            # ! Set the vel_diff constraint
            # self.opti.subject_to(self.x[2, i] - self.leading_velocity <= self.tightened_bound_N_vel_diff_list[i].item())
            # self.opti.subject_to(self.x[2, i] - self.leading_velocity >= -self.tightened_bound_N_vel_diff_list[i].item())
            # ! set the y constraint
            # self.opti.subject_to(self.x[1, i] <= self.tightened_bound_N_y_upper_list[i].item())
            # self.opti.subject_to(self.x[1, i] >= self.tightened_bound_N_y_lower_list[i].item())
            
        # set the constraints for the input  [-3.14/8,-0.7*9.81],[3.14/8,0.05*9.81]
        self.opti.subject_to(self.u[0, :] >= -3.14 / 8)
        self.opti.subject_to(self.u[0, :] <= 3.14 / 8)
        self.opti.subject_to(self.u[1, :] >= -0.5 * 9.81)
        self.opti.subject_to(self.u[1, :] <= 0.5 * 9.81)
        self.opti.subject_to(self.lambda_s <= 0)
        # self.opti.subject_to(self.slack_y >= 0)
        # ! here is test for sigma
        for i in range(self.N+1):
            self.opti.subject_to(self.sigma[0,i] >= -3e4*self.lambda_s[0,i])
            self.opti.subject_to(self.sigma[0,i] >= 3e4*self.lambda_s[0,i]**2)
        
        
        
        
    def setCost(self):
        """
        Set cost function for the optimization problem.
        """
        L, Lf = self.vehicle.getCost()
        cost=getTotalCost(L, Lf, self.x, self.u, self.refx, self.refu, self.N)
        # Add slack variable cost
        cost += 3e4*self.lambda_s@ self.lambda_s.T
            
        # [0, DM(2.32743), DM(3.3005), DM(4.05798), DM(4.70298), DM(5.27452), DM(5.79271), 
            #  DM(6.26978), DM(6.7139), DM(7.13089), DM(7.52508), DM(7.89977), DM(8.25757)]
        # for i in range(self.N+1):
        #     cost +=if_else(self.lambda_s[0,i]<=self.lambda_s[0,i]**2, 3e4*self.lambda_s[0,i]**2, -3e4*self.lambda_s[0,i]) 
        # ! here test the sigma
        for i in range(self.N+1):
            cost += self.sigma[0,i]
        for i in range(self.N-1):
            cost += 1e2*(self.u[1,i+1]-self.u[1,i])@(self.u[1,i+1]-self.u[1,i]).T
            self.cost=cost
        # Add slack variable cost for y
        # cost += 5e5*self.slack_y@ self.slack_y.T
        self.opti.minimize(self.cost)
    
    def setController(self):
        """
        Set constraints and cost function for the MPC controller.
        """
        self.setStateEqconstraints()
        self.setInEqConstraints()
        self.setCost()
    
    def solve(self, x0, ref_trajectory, ref_control, p_leading, leading_velocity=10, vel_diff=6):
        """
        Solve the MPC problem.
        """
        # Set the initial condition and reference trajectories
        
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.refx, ref_trajectory)
        self.opti.set_value(self.refu, ref_control)
        self.opti.set_value(self.p_leading, p_leading)
        self.opti.set_value(self.leading_velocity, leading_velocity)
        self.opti.set_value(self.vel_diff, vel_diff)
        # Solver options
        opts = {"ipopt": {"print_level": 0, "tol": 1e-8}, "print_time": 0}
        self.opti.solver("ipopt", opts)
        
        try:
            sol = self.opti.solve()
            u_opt = sol.value(self.u)
            x_opt = sol.value(self.x)
            lambda_s = sol.value(self.lambda_s)
            # lambda_y = sol.value(self.slack_y)
            # [0, DM(2.32743), DM(3.3005), DM(4.05798), DM(4.70298), DM(5.27452), DM(5.79271), 
            #  DM(6.26978), DM(6.7139), DM(7.13089), DM(7.52508), DM(7.89977), DM(8.25757)]
            # print(f"lambda_y: {lambda_y}")
            # also return tightened IDM constraint with solved op
            tightened_IDM_constraints = [sol.value(constraint) for constraint in self.IDM_constraint_list]
            print(f"Cost value: {sol.value(self.cost)}")
            return u_opt, x_opt, lambda_s, tightened_IDM_constraints
        except Exception as e:
            print(f"An error occurred: {e}")
            self.opti.debug.value(self.x)
            return None, None
        
        
    def get_dynammic_model(self):
        """
        Return the dynamic model of the vehicle.
        """
        return self.A, self.B, self.C








