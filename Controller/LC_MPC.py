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
        self.Param = Param()
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
        constraint_shift = func1 + func2 + + 143.318146 -self.laneWidth/2
        
        return constraint_shift
    
    # def lane_change_constraint(self):
    #     constraints = []
    #     leadWidth, leadLength =self.leadWidth, self.leadLength
    #     for i in range(self.Traffic.getDim()):
    #         # !  ignore the ego vehicle itself, the 1th is the ego vehicle
    #         if i == 1:
    #             continue
    #         # !  ignore the ego vehicle itself, the 1th is the ego vehicle
    #         v0_i = self.Traffic.get_velocity()[i]
            
    #         func1 = self.traffic_sign * (self.traffic_sign*(self.traffic_y-self.traffic_shift) + self.egoWidth + leadWidth) / 2 * \
    #                 tanh(self.px - self.traffic_x + leadLength/2 + self.L_tract + v0_i * self.Time_headway + self.min_distx )  + self.traffic_shift/2
    #         func2 = self.traffic_sign * (self.traffic_sign*(self.traffic_y-self.traffic_shift) + self.egoWidth + leadWidth) / 2 * \
    #                 tanh( - (self.px - self.traffic_x)  + leadLength/2 + self.L_trail + v0_i * self.Time_headway+ self.min_distx )  + self.traffic_shift/2
    #         constraint_shift = func1 + func2 + + 143.318146 -self.laneWidth/2
    #         constraints.append(Function('S',[self.px,self.traffic_x,self.traffic_y,
    #                                 self.traffic_sign,self.traffic_shift,],
    #                                 [constraint_shift],['px','t_x','t_y','t_sign','t_shift'],['y_cons']))
    #     return constraints
    
    def setInEqConstraints(self):
        """
        Set inequality constraints.
        """
        lane_change_constraint_all = []
        for i in range(self.N+1):
            # ! calculate the px, according to the vehicle state and time
            self.px_all[0,i] = self.x[0,i] + self.x[2,i] * self.Param.dt
            # pleading here include the initial x, y of the leading vehicle
            # ! calculate the self.traffic_x and self.traffic_y according to the leading velocity
            self.traffic_x_all[0,i] = self.p_leading[0] + self.leading_velocity * self.Param.dt * i
            self.traffic_y_all[0,i] = self.p_leading[1]
            
        # ! here is the lane change constraint
        for i in range(self.N+1):
            self.traffic_x = self.traffic_x_all[0,i]
            self.traffic_y = self.traffic_y_all[0,i]
            self.px = self.px_all[0,i]
            constraint_shift = self.lane_change_constraint_test()
            lane_change_constraint_all.append(constraint_shift)
        
        # ! here is the lane change constraint, the y direction
        # for i in range(self.N+1):
        #     self.opti.subject_to(self.x[1,i] <= lane_change_constraint_all[i]+self.slack_y[0,i])
            
        # set the constraints for the input  [-3.14/8,-0.7*9.81],[3.14/8,0.05*9.81]
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
            cost += 1e2*(self.u[1,i+1]-self.u[1,i])@(self.u[1,i+1]-self.u[1,i]).T
        # Add slack variable cost for y
        cost += 5e5*self.slack_y@ self.slack_y.T
        self.opti.minimize(cost)
        
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
            return u_opt, x_opt
        except Exception as e:
            print(f"An error occurred: {e}")
            self.opti.debug.value(self.x)
            return None, None
        
        
    def get_dynammic_model(self):
        """
        Return the dynamic model of the vehicle.
        """
        return self.A, self.B, self.C