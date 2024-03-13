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
    ██████╗  ██████╗ ██████╗      ██████╗ ██╗     ███████╗███████╗███████╗    ███╗   ███╗██████╗  ██████╗            
    ██╔════╝ ██╔═══██╗██╔══██╗    ██╔══██╗██║     ██╔════╝██╔════╝██╔════╝    ████╗ ████║██╔══██╗██╔════╝            
    ██║  ███╗██║   ██║██║  ██║    ██████╔╝██║     █████╗  ███████╗███████╗    ██╔████╔██║██████╔╝██║                 
    ██║   ██║██║   ██║██║  ██║    ██╔══██╗██║     ██╔══╝  ╚════██║╚════██║    ██║╚██╔╝██║██╔═══╝ ██║                 
    ╚██████╔╝╚██████╔╝██████╔╝    ██████╔╝███████╗███████╗███████║███████║    ██║ ╚═╝ ██║██║     ╚██████╗            
     ╚═════╝  ╚═════╝ ╚═════╝     ╚═════╝ ╚══════╝╚══════╝╚══════╝╚══════╝    ╚═╝     ╚═╝╚═╝      ╚═════╝                                                                                                                                                                                                            
    """
    """
    #! Creates a MPC based on current vehicle, traffic and scenario
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
        self.nx,self.nu,self.nrefx,self.nrefu = self.vehicle.getSystemDim()
        self.opts = opts
        # ! get ref velocity 
        self.Vmax = scenario.getVmax()
        
        
        if self.opts["integrator"] == "rk":
            self.vehicle.integrator(opts["integrator"],dt)
            self.F_x  = self.vehicle.getIntegrator()
        else:
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
            self.traffic_slack = self.opti.variable(1,self.N+1)
        else:
            self.traffic_slack = self.opti.variable(self.Nveh,self.N+1)
            self.lead = self.opti.parameter(self.Nveh,self.N+1)
            self.traffic_x = self.opti.parameter(self.Nveh,self.N+1)
            self.traffic_y = self.opti.parameter(self.Nveh,self.N+1)
            self.traffic_sign = self.opti.parameter(self.Nveh,self.N+1)
            self.traffic_shift = self.opti.parameter(self.Nveh,self.N+1)
            self.traffic_flip = self.opti.parameter(self.Nveh,self.N+1)
            
        #! NEED TO CHANGE THIS
        # # solver
        if opts["version"] == "trailing":
            opts_trailing = {"ipopt": {"print_level": 0, "tol": 1e-8}, "print_time": 0}
            self.opti.solver("ipopt", opts_trailing)
        
        
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
    
    def setTrafficConstraints(self):
        self.S = self.scenario.constraint(self.traffic,self.opts)

        if self.scenario.name == 'simpleOvertake':
            for i in range(self.Nveh):
                self.opti.subject_to(self.traffic_flip[i,:] * self.x[1,:] 
                                    >= self.traffic_flip[i,:] * self.S[i](self.x[0,:], self.traffic_x[i,:], self.traffic_y[i,:],
                                        self.traffic_sign[i,:], self.traffic_shift[i,:]) +  self.traffic_slack[i,:])
            
            # Set default road boundries, given that there is a "phantom vehicle" in the lane we can not enter
            d_lat_spread =  self.L_trail* np.tan(self.egoTheta_max)
            if self.opts["version"] == "leftChange":
                self.y_lanes = [self.vehWidth/2+d_lat_spread,2*self.laneWidth-self.vehWidth/2-d_lat_spread]
            elif self.opts["version"] == "rightChange":
                self.y_lanes = [-self.laneWidth + self.vehWidth/2+d_lat_spread,self.laneWidth-self.vehWidth/2-d_lat_spread]
            self.opti.subject_to(self.opti.bounded(self.y_lanes[0],self.x[1,:],self.y_lanes[1]))

        elif self.scenario.name == 'trailing':
            T = self.scenario.Time_headway
            self.scenario.setEgoLane(self.traffic)
            self.scenario.getLeadVehicle(self.traffic)
            self.IDM_constraint_list = self.S(self.lead) + self.traffic_slack[0,:]-T * self.x[2,:]
            self.opti.subject_to(self.x[0,:]  <= self.S(self.lead) + self.traffic_slack[0,:]-T * self.x[2,:])
            # ! tighten the TRAILING CONSTRAINTS  
            self.tightened_bound_N_IDM_list, _ = self.MPC_tighten_bound.tighten_bound_N_IDM(self.IDM_constraint_list, self.N)
            for i in range(self.N+1):
                self.opti.subject_to(self.x[0,i] <= self.tightened_bound_N_IDM_list[i].item())

    def setInEqConstraints(self):
        """
        Set inequality constraints, only for default constraints and tihgtened  default  constraints
        
        """
        D = np.eye(self.nx)  # Noise matrix
        lbx,ubx = self.vehicle.xConstraints()
        # ! element in lbx and ubx should be positive
        lbx = [abs(x) for x in lbx]
        # print("this is the lbx",lbx)
        # print("this is the ubx",ubx)
        # print((np.array([[5000], [5000], [30], [3.14/8]]).shape))
        self.setInEqConstraints_val(H_up=None, upb=np.array(ubx).reshape(4,1), H_low=None, lwb=np.array(lbx).reshape(4,1))  # Set default constraints
        lbu,ubu = self.vehicle.uConstraints()
        #! initial MPC_tighten_bound CLASS for the STATE CONSTRAINTS
        self.MPC_tighten_bound = MPC_tighten_bound(self.A, self.B, D, np.diag(self.Q), np.diag(self.R), self.P0, self.process_noise, self.Possibility)
        self.tightened_bound_N_list_up = self.MPC_tighten_bound.tighten_bound_N(self.P0, self.H_up, self.upb, self.N, 1)
        self.tightened_bound_N_list_lw = self.MPC_tighten_bound.tighten_bound_N(self.P0, self.H_low, self.lwb, self.N, 0)
        
        for i in range(self.N+1):
            # print(DM(self.tightened_bound_N_list_up[i].reshape(-1, 1)))
            # print(DM(self.tightened_bound_N_list_lw[i].reshape(-1, 1)))
            self.opti.subject_to(self.x[:, i] <= DM(self.tightened_bound_N_list_up[i].reshape(-1, 1)))
            self.opti.subject_to(self.x[:, i] >= DM(self.tightened_bound_N_list_lw[i].reshape(-1, 1))) 
        # set the constraints for the input  [-3.14/8,-0.7*9.81],[3.14/8,0.05*9.81]
        #! set the constraints for the INPUT
        self.opti.subject_to(self.opti.bounded(lbu, self.u, ubu))
        #! extra constraints for the V_MAX
        self.opti.subject_to(self.opti.bounded(0,self.x[2,:],self.scenario.vmax))
        
    def setCost(self):
        L,Lf = self.vehicle.getCost()
        Ls = self.scenario.getSlackCost()
        costMain = getTotalCost(L,Lf,self.x,self.u,self.refx,self.refu,self.N)
        costSlack = getSlackCost(Ls,self.traffic_slack)
        self.total_cost = costMain + costSlack
        self.opti.minimize(self.total_cost)    
 
    def setController(self):
        """
        Sets all constraints and cost
        """
        # Constraints
        self.setStateEqconstraints()
        self.setInEqConstraints()
        self.setTrafficConstraints()

        # Cost
        self.setCost()
        
    def solve(self, x_iter, refxT_out, refu_out, x_traffic):
        """
        Solve the optimization problem
        """
        # Solve the optimization problem
        if self.opts["version"] == "trailing":
            self.opti.set_value(self.x0, x_iter)
            self.opti.set_value(self.refx, refxT_out)
            self.opti.set_value(self.refu, refu_out) 
            self.opti.set_value(self.lead, x_traffic)
        try:
            sol = self.opti.solve()
            u_opt = sol.value(self.u)
            x_opt = sol.value(self.x)
            cost = sol.value(self.total_cost)
            return u_opt, x_opt, cost
        except Exception as e:
            print(f"An error occurred: {e}")
            self.opti.debug.value(self.x)
            return None, None
class makeDecisionMaster:
    """
    #! This is the Decision Master used for decision making
    """
    def __init__(self,vehicle,traffic,controllers,scenarios,changeHorizon = 10,forgettingFact = 0.90): 
        self.vehicle = vehicle
        self.traffic = traffic
        self.controllers = controllers
        self.scenarios = scenarios
        
    def storeInput(self,input):
        """
        Stores the current states sent from main file
        """
        self.x_iter, self.refxL_out, self.refxR_out, self.refxT_out, self.refu_out, \
                                                    self.x_lead,self.traffic_state = input
    def setDecisionCost(self,q):
        """
        Sets costs of changing a decisions
        """
        self.decisionQ = q
        
    def chooseController(self):
        """
        Main function, finds optimal choice of controller for the current step
        """
        self.egoLane = self.scenarios[0].getEgoLane()
        idx = self.scenarios[0].getLeadVehicle(self.traffic)  
        if len(idx) == 0:         #No leading vehicle,  Move barrier very far forward
                x_traffic = DM(1,self.N+1)
                x_traffic[0,:] = self.x_iter[0] + 200
        else:
            x_traffic = self.x_lead[idx[0],:]
        u_opt, x_opt, cost=self.controllers.solve(self.x_iter, self.refxT_out, self.refu_out, x_traffic)
        return u_opt, x_opt, cost