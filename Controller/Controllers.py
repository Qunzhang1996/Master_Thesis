"""
This one the the controller that contains trailing and lane change controller.
"""
import sys
path_to_add='C:\\Users\\A490243\\Desktop\\Master_Thesis'
sys.path.append(path_to_add)
from casadi import *
import numpy as np
from matplotlib import pyplot as plt
from Controller.MPC_tighten_bound import MPC_tighten_bound
from util.utils import *
class makeController:
    """
    Creates a MPC based on current vehicle, traffic and scenario
    """
    def __init__(self, vehicle,traffic,scenario,N,opts,dt):
        self.vehicle = vehicle
        self.traffic = traffic
        self.scenario = scenario
        self.opts = opts 
        
        # Get constraints and road information
        self.Nveh = self.traffic.getNveh() # here, get the number of vehicles of the traffic scenario
        self.N = N
        self.vehocleWidth,_,_,_ = self.vehicle.getSize()
        self.roadMin, self.roadMax, self.laneCenters = self.scenario.getRoad()
        self.opts = opts
        # ! get ref velocity 
        self.Vmax = scenario.getVmax()
        # ! set the LTI model from the vehicle model
        self.A, self.B, self.C = self.vehicle.vehicle_linear_discrete_model(v=15, phi=0, delta=0)
        # ! get the cost param from the vehicle model
        self.Q, self.R = self.vehicle.getCostParam()
        
        #! P0, process_noise, possibility will be obtained from set_stochastic_mpc_params
        #! Used for tighten the MPC bound
        self.P0, self.process_noise, self.Possibility = set_stochastic_mpc_params()
        
        # ! create opti stack
         # Create Opti Stack
        self.opti = Opti()

        # # Initialize opti stack
        self.x = self.opti.variable(self.nx,self.N+1)
        self.u = self.opti.variable(self.nu,self.N)
        self.refx = self.opti.parameter(self.nrefx,self.N+1)
        self.refu = self.opti.parameter(self.nrefu,self.N)
        self.x0 = self.opti.parameter(self.nx,1)
        
        
        # ! change this according to the LC_MPC AND TRAILING_MPC
        if opts["version"] == "trailing":
            self.lead = self.opti.parameter(1,self.N+1)
            # slack variable for the IDM constraint
            self.lambda_s= self.opti.variable(1,self.N + 1)
        else:
            self.lead = self.opti.parameter(self.Nveh,self.N+1)
            self.traffic_x = self.opti.parameter(self.Nveh,self.N+1)
            self.traffic_y = self.opti.parameter(self.Nveh,self.N+1)
            self.traffic_sign = self.opti.parameter(self.Nveh,self.N+1)
            self.traffic_shift = self.opti.parameter(self.Nveh,self.N+1)
            self.traffic_flip = self.opti.parameter(self.Nveh,self.N+1)
            # add slack for the y direction
            self.slack_y = self.opti.variable(1, self.N+1)
            
        #! NEED TO CHANGE THIS
        
        
    def setStateEqconstraints(self):
        """
        Set state equation constraints, using the LTI model,  ref_v=15, ref_phi=0, ref_delta=0
        """
        for i in range(self.N):
            A_d, B_d, G_d = self.A, self.B, self.C
            self.opti.subject_to(self.x[:, i+1] == A_d @ self.x[:, i] + B_d @ self.u[:, i] + G_d)
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
    
    
    # def IDM_constraint(self, p_leading_x, v_eg, d_s=1, L1=8.4, T_s=1, lambda_s=0):
    #     """
    #     IDM constraint for tracking the vehicle in front.
    #     """
    #     return p_leading_x - L1 - d_s - T_s * v_eg - lambda_s
    def setTrafficConstraints(self):
        self.S = self.scenario.constraint(self.traffic,self.opts)

        if self.scenario.name == 'simpleOvertake':
            for i in range(self.Nveh):
                self.opti.subject_to(self.traffic_flip[i,:] * self.x[1,:] 
                                    >= self.traffic_flip[i,:] * self.S[i](self.x[0,:], self.traffic_x[i,:], self.traffic_y[i,:],
                                        self.traffic_sign[i,:], self.traffic_shift[i,:]))

        elif self.scenario.name == 'trailing':
            self.scenario.setEgoLane()
            self.scenario.getLeadVehicle(self.traffic)
            self.opti.subject_to(self.x[0,:] <= self.S(self.lead)) 
    
    
    
    def setInEqConstraints(self):
        """
        Set inequality constraints, only for default constraints and tihgtened constraints
        
        """
        D = np.eye(self.nx)  # Noise matrix
        lbx,ubx = self.vehicle.xConstraints()
        self.setInEqConstraints_val(H_up=lbx, upb=None, H_low=ubx, lwb=None)  # Set default constraints
        lbu,ubu = self.vehicle.uConstraints()
        #! initial MPC_tighten_bound CLASS for the STATE CONSTRAINTS
        self.MPC_tighten_bound = MPC_tighten_bound(self.A, self.B, D, self.Q, self.R, self.P0, self.process_noise, self.Possibility)
        self.tightened_bound_N_list_up = self.MPC_tighten_bound.tighten_bound_N(self.P0, self.H_up, self.upb, self.N, 1)
        self.tightened_bound_N_list_lw = self.MPC_tighten_bound.tighten_bound_N(self.P0, self.H_low, self.lwb, self.N, 0)
        for i in range(self.N+1):
            self.opti.subject_to(self.x[:, i] <= DM(self.tightened_bound_N_list_up[i].reshape(-1, 1)))
            self.opti.subject_to(self.x[:, i] >= DM(self.tightened_bound_N_list_lw[i].reshape(-1, 1)))
        # set the constraints for the input  [-3.14/8,-0.7*9.81],[3.14/8,0.05*9.81]
        #! set the constraints for the INPUT
        self.opti.subject_to(self.opti.bounded(lbu, self.u, ubu))
        #! extra constraints for the V_MAX
        self.opti.subject_to(self.opti.bounded(0,self.x[2,:],self.scenario.vmax))
        
        
        ######! just put IDM in to test it
        