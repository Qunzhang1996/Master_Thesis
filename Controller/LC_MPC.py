"""Here is controller for lane change 
"""
# ! here is the lane change mpc controller
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
class LC_MPC:
    """
    ██████╗  ██████╗ ██████╗      ██████╗ ██╗     ███████╗███████╗███████╗    ███╗   ███╗██████╗  ██████╗            
    ██╔════╝ ██╔═══██╗██╔══██╗    ██╔══██╗██║     ██╔════╝██╔════╝██╔════╝    ████╗ ████║██╔══██╗██╔════╝            
    ██║  ███╗██║   ██║██║  ██║    ██████╔╝██║     █████╗  ███████╗███████╗    ██╔████╔██║██████╔╝██║                 
    ██║   ██║██║   ██║██║  ██║    ██╔══██╗██║     ██╔══╝  ╚════██║╚════██║    ██║╚██╔╝██║██╔═══╝ ██║                 
    ╚██████╔╝╚██████╔╝██████╔╝    ██████╔╝███████╗███████╗███████║███████║    ██║ ╚═╝ ██║██║     ╚██████╗            
     ╚═════╝  ╚═════╝ ╚═════╝     ╚═════╝ ╚══════╝╚══════╝╚══════╝╚══════╝    ╚═╝     ╚═╝╚═╝      ╚═════╝                                                                                                                                                                                                            
    """
    def __init__(self, vehicle, Traffic, Q, R, P0, process_noise, Possibility=0.95, N=12) -> None:
        # The number of MPC states, here include x, y, psi and v
        NUM_OF_STATES = 4
        self.nx = NUM_OF_STATES
        # The number of MPC actions, including acc and steer_angle
        NUM_OF_ACTS = 2
        self.nu = NUM_OF_ACTS
        self.vehicle = vehicle
        self.L_tract, self.L_trail, self.egoWidth = self.vehicle.get_vehicle_size()
        self.nx, self.nu, self.nrefx, self.nrefu = self.vehicle.getSystemDim()
        self.Traffic = Traffic   # ! notice that the traffic is the object of the class Traffic
        self.laneWidth = self.Traffic.get_laneWidth()
        self.leadWidth, self.leadLength = self.Traffic.get_size() # ! the size of the leading vehicle
        self.Q = Q
        self.R = R
        self.P0 = P0
        self.process_noise = process_noise
        self.Possibility = Possibility
        self.N = N
        self.dt = self.vehicle.get_dt()
        # ref val for the vehicle matrix
        self.v_ref = 15
        self.phi_ref = 0
        self.delta_ref = 0
        self.Time_headway = 0.5
        self.min_distx = 5
        
        # Create Opti Stack
        self.opti = Opti()
        # Initialize opti stack
        self.x = self.opti.variable(self.nx, self.N + 1)
        self.u = self.opti.variable(self.nu, self.N)
        self.refx = self.opti.parameter(self.nrefx, self.N + 1)
        self.refu = self.opti.parameter(self.nrefu, self.N)
        self.x0 = self.opti.parameter(self.nx, 1)
        #! here is only one test for the right lane change
        # slack variable for the lane change constraint for  y direction
        self.slack_y = self.opti.variable(1, self.N)
        # create a Opti stack for traffic_x, traffic_y, px
        self.traffic_x_all = self.opti.variable(1, self.N+1)
        self.traffic_y_all = self.opti.variable(1, self.N+1)
        self.px_all = self.opti.variable(1, self.N+1)
        self.leading_velocity = self.opti.parameter(1)
        self.p_leading = self.opti.parameter(2,1)
        # add slack for the y direction
        self.slack_y = self.opti.variable(1, self.N+1)
    
    def compute_Dlqr(self):
        return self.MPC_tighten_bound.calculate_Dlqr()
    
    def IDM_constraint(self, p_leading, v_eg, d_s=1, L1=6, T_s=1, lambda_s=0):
        """
        IDM constraint for tracking the vehicle in front.
        """
        return p_leading - L1 - d_s - T_s * v_eg - lambda_s
    
    def calc_linear_discrete_model(self, v=15, phi=0, delta=0):
        """
        Calculate linear and discrete time dynamic model.
        """
        return self.vehicle.vehicle_linear_discrete_model(v, phi, delta)
    
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
        self.H_up = H_up if H_up is not None else [np.array([[1], [0], [0], [0]]), np.array([[0], [1], [0], [0]]), \
                                                   np.array([[0], [0], [1], [0]]), np.array([[0], [0], [0], [1]])]
        self.upb = upb if upb is not None else np.array([[5000], [5000], [30], [3.14/8]])
        
        self.H_low = H_low if H_low is not None else [np.array([[-1], [0], [0], [0]]), np.array([[0], [-1], [0], [0]]),\
                                                      np.array([[0], [0], [-1], [0]]), np.array([[0], [0], [0], [-1]])]
        self.lwb = lwb if lwb is not None else np.array([[5000], [5000], [0], [3.14/8]])
    
    def lane_change_constraint_test(self):
        '''
        This is the test for the lane change constraint
        '''
        leadWidth, leadLength = self.leadWidth, self.leadLength
        v0_i = self.leading_velocity
        # !traffic_shift=-laneWidth, traffic_sign=1, for the right lane change
        self.traffic_sign = 1
        self.traffic_shift = -self.laneWidth
        func1 = self.traffic_sign * (self.traffic_sign*(self.traffic_y-self.traffic_shift) + self.egoWidth + leadWidth) / 2 * \
                    tanh(self.px - self.traffic_x + leadLength/2 + self.L_tract + v0_i * self.Time_headway + self.min_distx )  + self.traffic_shift/2
        func2 = self.traffic_sign * (self.traffic_sign*(self.traffic_y-self.traffic_shift) + self.egoWidth + leadWidth) / 2 * \
                tanh( - (self.px - self.traffic_x)  + leadLength/2 + self.L_trail + v0_i * self.Time_headway+ self.min_distx )  + self.traffic_shift/2
        constraint_shift = func1 + func2 + 143.318146 -self.laneWidth/2
        
        return constraint_shift
    
    
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
      

        # Example tightened bound application (adjust according to actual implementation)
        self.tightened_bound_N_list_up = self.MPC_tighten_bound.tighten_bound_N(self.P0, self.H_up, self.upb, self.N, 1)
        self.tightened_bound_N_list_lw = self.MPC_tighten_bound.tighten_bound_N(self.P0, self.H_low, self.lwb, self.N, 0)

        # the tightened bound (up/lw) is N+1 X NUM_OF_STATES  [x, y, v, psi] 
        # according to the new bounded constraints set the constraints
        for i in range(self.N+1):
            self.opti.subject_to(self.x[:, i] <= DM(self.tightened_bound_N_list_up[i].reshape(-1, 1)))
            self.opti.subject_to(self.x[:, i] >= DM(self.tightened_bound_N_list_lw[i].reshape(-1, 1)))
        self.lane_change_constraint_all = []
        for i in range(self.N+1):
            # ! calculate the px, according to the vehicle state and time
            self.px_all[0,i] = self.x[0,i] + self.x[2,i] * self.dt
            # pleading here include the initial x, y of the leading vehicle
            # ! calculate the self.traffic_x and self.traffic_y according to the leading velocity
            self.traffic_x_all[0,i] = self.p_leading[0] + self.leading_velocity * self.dt * i
            # ! notice! traffic_y should be scaled according to the map, the center of the map is func2 + 143.318146 -self.laneWidth/2
            self.traffic_y_all[0,i] = self.p_leading[1]-(143.318146 -self.laneWidth/2)
            
        # ! here is the lane change constraint
        for i in range(self.N+1):
            self.traffic_x = self.traffic_x_all[i]
            self.traffic_y = self.traffic_y_all[i]
            self.px = self.px_all[i]
            constraint_shift = self.lane_change_constraint_test()
            self.lane_change_constraint_all.append(constraint_shift+self.slack_y[0,i])
        
        self.tightened_LC_list,_=self.MPC_tighten_bound.tighten_bound_N_laneChange(self.lane_change_constraint_all, self.N)
        # ! here is the lane change constraint, the y direction
        for i in range(self.N+1):
            self.opti.subject_to(self.x[1,i] >= self.lane_change_constraint_all[i])
        self.opti.subject_to(self.opti.bounded(143.318146 -self.laneWidth*3/2+2.7/2, self.x[1,:], 143.318146 + self.laneWidth/2-2.7/2))
        # set the constraints for the input  [-3.14/8,-0.7*9.81],[3 .14/8,0.05*9.81]
        self.opti.subject_to(self.u[0, :] >= -3.14 / 8)
        self.opti.subject_to(self.u[0, :] <= 3.14 / 8)
        self.opti.subject_to(self.u[1, :] >= -0.5 * 9.81)
        self.opti.subject_to(self.u[1, :] <= 0.5 * 9.81)
    
    def setCost(self):
        """
        Set cost function for the optimization problem.
        """
        L, Lf = self.vehicle.getCost()
        cost=getTotalCost(L, Lf, self.x, self.u, self.refx, self.refu, self.N)
        # Add slack variable cost
        for i in range(self.N-1):
            # cost += 1e2*(self.u[0,i+1]-self.u[0,i])@(self.u[0,i+1]-self.u[0,i]).T
            # cost += 1e2*(self.u[1,i+1]-self.u[1,i])@(self.u[1,i+1]-self.u[1,i]).T
            cost += 1e2*(self.u[:,i+1]-self.u[:,i]).T@(self.u[:,i+1]-self.u[:,i])
        # Add slack variable cost for y
        cost += 1e4*self.slack_y@ self.slack_y.T
        self.cost=cost
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
        # Solver options
        opts = {"ipopt": {"print_level": 0, "tol": 1e-8}, "print_time": 0}
        self.opti.solver("ipopt", opts)
        
        try:
            sol = self.opti.solve()
            u_opt = sol.value(self.u)
            x_opt = sol.value(self.x)
            lambda_y = sol.value(self.slack_y)
            # try to visualize the constraints of the y direction, not the slack variable, lane_change_constraint_all
            lane_change_constraint_all=[sol.value(self.lane_change_constraint_all[i]) for i in range(self.N+1)]

            print(f"Cost value: {sol.value(self.cost)}")
            return u_opt, x_opt, lambda_y,lane_change_constraint_all
        except Exception as e:
            print(f"An error occurred: {e}")
            self.opti.debug.value(self.x)
            return None, None
        
        
    def get_dynammic_model(self):
        """
        Return the dynamic model of the vehicle.
        """
        return self.A, self.B, self.C