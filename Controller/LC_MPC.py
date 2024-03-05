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
    
    
    def lane_change_constraint(self):
        constraints = []
        leadWidth, leadLength =self.leadWidth, self.leadLength
        for i in range(self.Traffic.getDim()):
            # !  ignore the ego vehicle itself, the 1th is the ego vehicle
            if i == 1:
                continue
            # !  ignore the ego vehicle itself, the 1th is the ego vehicle
            v0_i = self.Traffic.get_velocity()[i]
            
            func1 = self.traffic_sign * (self.traffic_sign*(self.traffic_y-self.traffic_shift) + self.egoWidth + leadWidth) / 2 * \
                    tanh(self.px - self.traffic_x + leadLength/2 + self.L_tract + v0_i * self.Time_headway + self.min_distx )  + self.traffic_shift/2
            func2 = self.traffic_sign * (self.traffic_sign*(self.traffic_y-self.traffic_shift) + self.egoWidth + leadWidth) / 2 * \
                    tanh( - (self.px - self.traffic_x)  + leadLength/2 + self.L_trail + v0_i * self.Time_headway+ self.min_distx )  + self.traffic_shift/2
            constraint_shift = func1 + func2 + + 143.318146 -self.laneWidth/2
            constraints.append(Function('S',[self.px,self.traffic_x,self.traffic_y,
                                    self.traffic_sign,self.traffic_shift,],
                                    [constraint_shift],['px','t_x','t_y','t_sign','t_shift'],['y_cons']))
        return constraints
            
            
    
    
        
    # def lane_change_constraint():
    #     constraints = []
    #     leadWidth, leadLength = traffic.getVehicles()[0].getSize()
    #     for i in range(traffic.getDim()):
    #         v0_i = traffic.vehicles[i].v0
    #         func1 = self.traffic_sign * (self.traffic_sign*(self.traffic_y-self.traffic_shift)/2 + self.egoWidth + leadWidth) / 2 * \
    #                 tanh(self.px - self.traffic_x + leadLength/2 + self.L_tract + v0_i * self.Time_headway + self.min_distx )  + self.traffic_shift/2
    #         func2 = self.traffic_sign * (self.traffic_sign*(self.traffic_y-self.traffic_shift)/2 + self.egoWidth + leadWidth) / 2 * \
    #                 tanh( - (self.px - self.traffic_x)  + leadLength/2 + self.L_trail + v0_i * self.Time_headway+ self.min_distx )  + self.traffic_shift/2

    #         constraints.append(Function('S',[self.px,self.traffic_x,self.traffic_y,
    #                                 self.traffic_sign,self.traffic_shift,],
    #                                 [func1+func2],['px','t_x','t_y','t_sign','t_shift'],['y_cons']))
    #     return constraints