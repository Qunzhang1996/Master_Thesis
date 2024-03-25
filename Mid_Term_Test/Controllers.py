"""
This one the the controller that contains trailing and lane change controller.
"""
import sys
path_to_add='C:\\Users\\A490243\\Desktop\\Master_Thesis'
sys.path.append(path_to_add)
from casadi import *
import numpy as np
from matplotlib import pyplot as plt
from MPC_tighten_bound import MPC_tighten_bound
from util.utils import *
class makeController:   
    """
    #! Creates a MPC based on current vehicle, traffic and scenario
    """
    """
    ██████╗  ██████╗ ██████╗      ██████╗ ██╗     ███████╗███████╗███████╗    ███╗   ███╗██████╗  ██████╗            
    ██╔════╝ ██╔═══██╗██╔══██╗    ██╔══██╗██║     ██╔════╝██╔════╝██╔════╝    ████╗ ████║██╔══██╗██╔════╝            
    ██║  ███╗██║   ██║██║  ██║    ██████╔╝██║     █████╗  ███████╗███████╗    ██╔████╔██║██████╔╝██║                 
    ██║   ██║██║   ██║██║  ██║    ██╔══██╗██║     ██╔══╝  ╚════██║╚════██║    ██║╚██╔╝██║██╔═══╝ ██║                 
    ╚██████╔╝╚██████╔╝██████╔╝    ██████╔╝███████╗███████╗███████║███████║    ██║ ╚═╝ ██║██║     ╚██████╗            
     ╚═════╝  ╚═════╝ ╚═════╝     ╚═════╝ ╚══════╝╚══════╝╚══════╝╚══════╝    ╚═╝     ╚═╝╚═╝      ╚═════╝                                                                                                                                                                                                            
    """
    
    def __init__(self, vehicle,traffic,scenario,N,opts,dt):
        self.vehicle = vehicle
        self.traffic = traffic
        self.scenario = scenario
        self.opts = opts 
        
        # Get constraints and road information
        self.N = N
        self.Nveh = self.traffic.getNveh() # here, get the number of vehicles of the traffic scenario
        self.laneWidth = self.traffic.get_laneWidth()
        self.vehWidth,self.egoLength,self.L_tract, self.L_trail = self.vehicle.getSize()
        self.nx,self.nu,self.nrefx,self.nrefu = self.vehicle.getSystemDim()
        self.init_bound = self.vehicle.getInitBound()
        self.P0, self.process_noise, self.Possibility = self.vehicle.P0, self.vehicle.process_noise, self.vehicle.Possibility
        self.roadMin, self.roadMax, self.laneCenters, _ = self.scenario.getRoad()
        self.egoTheta_max  = vehicle.xConstraints()[1][3]  #! In this situation, we do not have egoTheta_max. no trailor
        self.opts = opts
        # ! get ref velocity 
        self.Vmax = scenario.getVmax()
        
        
        if self.opts["integrator"] == "rk":
            self.vehicle.integrator(opts["integrator"],dt)
            self.F_x  = self.vehicle.getIntegrator()
        else:
            
            # self.vehicle.integrator("rk",dt)
            # self.F_x  = self.vehicle.getIntegrator()
            # ! set the LTI model from the vehicle model
            self.A, self.B, self.C = self.vehicle.vehicle_linear_discrete_model(v=15, phi=0, delta=0)
        self.D = np.eye(self.nx)  # Noise matrix
        # ! get the cost param from the vehicle model
        self.Q, self.R = self.vehicle.getCostParam()
            
        #! P0, process_noise, possibility will be obtained from set_stochastic_mpc_params
        #! Used for tighten the MPC bound
        #! initial MPC_tighten_bound CLASS for the STATE CONSTRAINTS
    
        self.MPC_tighten_bound = MPC_tighten_bound(self.A, self.B, self.D, np.diag(self.Q), np.diag(self.R), self.P0, self.process_noise, self.Possibility)
        
        # ! create opti stack
         # Create Opti Stack
        self.opti = Opti()

        # # Initialize opti stack
        self.x = self.opti.variable(self.nx,self.N+1)
        self.u = self.opti.variable(self.nu,self.N)
        self.refx = self.opti.parameter(self.nrefx,self.N+1)
        self.refu = self.opti.parameter(self.nrefu,self.N)
        self.x0 = self.opti.parameter(self.nx,1)
        
        
        #! turn on/off the stochastic MPC
        self.stochasticMPC=1
        
        
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
        p_opts = dict(print_time=False, verbose=False)
        s_opts = dict(print_level=0)
        self.opti.solver(self.opts["solver"], p_opts,s_opts)
        
        
    def setStateEqconstraints(self):
        """
        Set state equation constraints, using the LTI model,  ref_v=15, ref_phi=0, ref_delta=0
        """
        for i in range(self.N):
            A_d, B_d, G_d = self.A, self.B, self.C
            self.opti.subject_to(self.x[:, i+1] == A_d @ self.x[:, i] + B_d @ self.u[:, i] + G_d)
            # self.opti.subject_to(self.x[:, i+1] == self.F_x(self.x[:,i],self.u[:,i]))
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
        
        if self.stochasticMPC:
            self.temp_x, self.tempt_y = self.MPC_tighten_bound.getXtemp(self.N ), self.MPC_tighten_bound.getYtemp(self.N )
            print("INFO: temp_x is:", self.temp_x)
            print("INFO: temp_y is:", self.tempt_y)
            self.S =self.scenario.constrain_tightened(self.traffic,self.opts,self.temp_x, self.tempt_y)
        else:
            self.S = self.scenario.constraint(self.traffic,self.opts)
            

        if self.scenario.name == 'simpleOvertake':
            
            #! DO NOT TAKE EGO VEHICLE INTO ACCOUNT
            for i in range(self.Nveh):
                if i ==1: continue #! avoid putting the ego vehicle in the list
                self.opti.subject_to(self.traffic_flip[i,:] * self.x[1,:] 
                                    >= self.traffic_flip[i,:] * self.S[i](self.x[0,:], self.traffic_x[i,:], self.traffic_y[i,:],
                                        self.traffic_sign[i,:], self.traffic_shift[i,:]) +  self.traffic_slack[i,:])
            #TODO: BE CAREFUL ABOUT THE SHIFT "143.318146 -self.laneWidth/2" -- self.init_bound, IT IS DEFINED IN THE SCENERIO
            # ! ASK ERIK, TOO STRICT CONSTRAINTS
            #TODO: DO The Tighten Finally
            # Set default road boundries, given that there is a "phantom vehicle" in the lane we can not enter
            d_lat_spread =  self.L_trail* np.tan(self.egoTheta_max)
            if self.opts["version"] == "leftChange":
                self.y_lanes = [self.init_bound + self.vehWidth/2+d_lat_spread, self.init_bound + 2*self.laneWidth-self.vehWidth/2-d_lat_spread]
            elif self.opts["version"] == "rightChange":
                self.y_lanes = [self.init_bound -self.laneWidth + self.vehWidth/2+d_lat_spread, self.init_bound + self.laneWidth-self.vehWidth/2-d_lat_spread]
            # self.opti.subject_to(self.opti.bounded(self.y_lanes[0],self.x[1,:],self.y_lanes[1]))

        elif self.scenario.name == 'trailing':
            T = self.scenario.Time_headway
            self.scenario.setEgoLane(self.traffic)
            self.scenario.getLeadVehicle(self.traffic)
            self.IDM_constraint_list = self.S(self.lead) + self.traffic_slack[0,:]-T * self.x[2,:]
            self.opti.subject_to(self.x[0,:]  <= self.S(self.lead) + self.traffic_slack[0,:]-T * self.x[2,:])
            # ! tighten the TRAILING CONSTRAINTS  
            # self.tightened_bound_N_IDM_list, _ = self.MPC_tighten_bound.tighten_bound_N_IDM(self.IDM_constraint_list, self.N)
            # for i in range(self.N+1):
            #     self.opti.subject_to(self.x[0,i] <= self.tightened_bound_N_IDM_list[i].item())

    def setInEqConstraints(self):
        """
        Set inequality constraints, only for default constraints and tihgtened  default  constraints
        
        """
        lbx,ubx = self.vehicle.xConstraints()
        # ! element in lbx and ubx should be positive
        lbx = [abs(x) for x in lbx]
        self.setInEqConstraints_val(H_up=None, upb=np.array(ubx).reshape(4,1), H_low=None, lwb=np.array(lbx).reshape(4,1))  # Set default constraints
        lbu,ubu = self.vehicle.uConstraints()
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
        self.costMain = getTotalCost(L,Lf,self.x,self.u,self.refx,self.refu,self.N)
        self.costSlack = getSlackCost(Ls,self.traffic_slack)
        self.total_cost = self.costMain + self.costSlack
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
        
    def solve(self, *args, **kwargs):
        """
        Solve the optimization problem with flexible inputs based on configuration in self.opts.
        """
        if self.opts["version"] == "trailing":
            x_iter, refxT_out, refu_out, x_traffic = args[:4]
            self.opti.set_value(self.x0, x_iter)
            self.opti.set_value(self.refx, refxT_out)
            self.opti.set_value(self.refu, refu_out) 
            self.opti.set_value(self.lead, x_traffic)
        else:
            '''
            used for the overtake issue
            '''
            x_iter, refx_out, refu_out, traffic_x, traffic_y, traffic_sign, traffic_shift, traffic_flip = args
            self.opti.set_value(self.x0, x_iter)
            self.opti.set_value(self.refx, refx_out)
            self.opti.set_value(self.refu, refu_out)
            self.opti.set_value(self.traffic_x, traffic_x)
            self.opti.set_value(self.traffic_y, traffic_y)
            self.opti.set_value(self.traffic_sign, traffic_sign) 
            self.opti.set_value(self.traffic_shift, traffic_shift)
            self.opti.set_value(self.traffic_flip, traffic_flip)
            #! check if this is useful
            #! self.lead = self.opti.parameter(self.Nveh,self.N+1)

        try:
            sol = self.opti.solve()
            u_opt = sol.value(self.u)
            x_opt = sol.value(self.x)
            costMain = sol.value(self.costMain)
            costSlack = sol.value(self.costSlack)
            return u_opt, x_opt, costMain, costSlack
        except Exception as e:
            print(f"An error occurred: {e}")
            self.opti.debug()
            return None, None, None
        
        
class makeDecisionMaster:
    """
    #! This is the Decision Master used for decision making
    """
    def __init__(self,vehicle,traffic,controllers,scenarios,changeHorizon = 10, forgettingFact = 0.90): 
        self.vehicle = vehicle
        self.traffic = traffic
        self.controllers = controllers
        self.scenarios = scenarios
        
        self.laneCenters = self.scenarios[1].getRoad()[2]
        self.laneWidth = self.scenarios[1].laneWidth
        
        self.egoLane = self.scenarios[0].getEgoLane()
        self.egoPx = self.vehicle.getPosition()[0]
        
        self.init_bound = self.traffic.getInitBound()
        
        self.nx,self.nu,self.nrefx,self.nrefu = vehicle.getSystemDim()
        self.N = vehicle.N
        self.Nveh = self.traffic.getDim()
        
        self.doRouteGoalScenario = 0
        
        self.state = np.zeros((self.nx,))
        self.x_pred = np.zeros((self.nx,self.N))
        self.u_pred = np.zeros((self.nu,self.N))
        self.tol = 0.1
        self.consecutiveErrors = 0
        self.changeHorizon = changeHorizon
        self.forgettingFact = forgettingFact
        
        self.MPCs = []
        for i in range(len(self.controllers)):
            self.MPCs.append(controllers[i])

        self.errors = 0

        self.decisionLog = []
        #! TEST, ADD COST AND DECISION IN THE LOG
        

    def checkSolution(self, x_pred_new, u_pred_new):
        """
        Adjusted to ensure at least a 10-step prediction horizon is maintained in the outputs.
        Checks if the MPC returned a strange solution:
        - If that is the case, fall back to the previous solution.
        - The number of times this occurs is presented as "Error count".
        """
        cond1 = (x_pred_new[0,0]) < (self.x_pred[0,0] - self.tol)
        cond2 = ((x_pred_new[1,0] - x_pred_new[1,1]) > 1)
        if (cond1 or cond2) and (self.consecutiveErrors < self.N-1):
            self.consecutiveErrors += 1
            self.errors += 1
            # Adjusting to ensure at least 10-step horizon is included in the output.
            # Here we assume self.x_pred and self.u_pred are already maintaining this horizon.
            x_pred_output = self.x_pred[:, :11]  # Ensure 10-step horizon for x_pred
            u_pred_output = self.u_pred[:, :11]  # Ensure 10-step horizon for u_pred
        else:
            self.consecutiveErrors = 0
            self.x_pred = x_pred_new
            self.u_pred = u_pred_new
            x_pred_output = x_pred_new[:, :11]  # Ensure 10-step horizon for x_pred
            u_pred_output = u_pred_new[:, :11]  # Ensure 10-step horizon for u_pred
        
        return x_pred_output, u_pred_output, self.x_pred

 
    def storeInput(self,input):
        """
        Stores the current states sent from main file
        """
        self.x_iter, self.refxL_out, self.refxR_out, self.refxT_out, self.refu_out, \
                                                    self.x_lead,self.traffic_state = input
                                                    
        # print("INFO: Ego position in storeInput is:", self.x_iter[0])
    
    
    def costRouteGoal(self,i):
        # i == 0 -> left change, i == 1 -> right change, i == 2 -> trail
        if self.doRouteGoalScenario and (self.goalP_x - self.egoPx < self.goalD_xmax):
            # Check if goal is reached
            if self.goalP_x - self.egoPx < 0:
                self.goalD_xmax = -1e5             # Deactivates the goal cost
                if self.egoLane == self.goalLane:
                    self.goalAccomplished = 1

            currentLane = self.egoLane
            # Find the best action in terms of reaching the goal
            if self.goalLane == currentLane:
                # If the goal lane is the current lane, dont change lane
                bestChoice = 2          
            elif currentLane == 1:
                # We are left and the goal is not left, change
                bestChoice = 0
            elif currentLane == -1:
                # We are right and the goal is not right, change
                bestChoice = 1
            else:
                # We are center and the gol is not center, change
                if self.goalLane == 1:
                    # If goal is left change left
                    bestChoice = 0
                elif self.goalLane == -1:
                    # If the goal is right change right
                    bestChoice = 1
            cost = (1 - ((self.goalP_x-self.egoPx) / self.goalD_xmax )** 0.4 ) * np.minimum(np.abs(i-bestChoice),1)
            return self.goalCost * cost
        else:
            # We dont consider any goal
            return 0
    
    
    
      
    def setDecisionCost(self,q):
        """
        Sets costs of changing a decisions
        """
        self.decisionQ = q
        
        
    def costDecision(self,decision):
        """
        Returns cost of the current decision based on the past (i == changeHorizon) decisions
        """
        cost = 0
        for i in range(self.changeHorizon):
            cost += self.decisionQ * (self.forgettingFact ** i) * (decision - self.decisionLog[i]) ** 2
        return cost

    def getDecision(self,costs):
        """
        Find optimal choice out of the three controllers
        """
        costMPC = np.array(costs)
        self.costTotal = np.zeros((3,))

        for i in range(3):
            self.costTotal[i] = self.costDecision(i) + costMPC[i] + self.costRouteGoal(i)
        return np.argmin(self.costTotal)   
        
        
        
    def updateReference(self, r=np.zeros((4,1))):
        """
        Updates the y position reference for each controller based on the current lane
        """
        self.scenarios[0].setEgoLane(self.traffic)
        #! Here add the noise 
        # py_ego = self.vehicle.getPosition()[1] + r[1]
        # self.egoPx = self.vehicle.getPosition()[0]+ r[0]
        py_ego =self.x_iter[1]
        self.egoPx = self.x_iter[0]
        
        print("INFO:  Ego position Measurement is:", self.egoPx)
        refu_in = [0,0,0]                                     # To work with function reference (update?)

        refxT_in,refxL_in,refxR_in = self.vehicle.getReferences()

        tol = 0.2
        if py_ego >= self.laneCenters[1]:
            # Set left reference to mid lane
            # Set trailing reference to left lane
            refxT_in[1] = self.laneCenters[1]
            refxL_in[1] = self.laneCenters[0]
            refxR_in[1] = self.laneCenters[2]

        elif py_ego <  self.laneCenters[2]:
            # Set right reference to right lane
            # Set trailing reference to right lane
            refxT_in[1] = self.laneCenters[2]
            refxL_in[1] = self.laneCenters[0]
            refxR_in[1] = self.laneCenters[1]

        elif abs(py_ego - self.laneCenters[0]) < tol:
            # Set left reference to left lane
            # Set right reference to right lane
            # Set trailing reference to middle lane
            # refxT_in[1] = self.laneCenters[0]
            refxL_in[1] = self.laneCenters[1]
            refxR_in[1] = self.laneCenters[2]

        # Trailing reference should always be the current Lane!
        refxT_in[1] = self.laneCenters[self.egoLane]
        
        self.refxT,_ = self.scenarios[0].getReference(refxT_in,refu_in)
        self.refxL,_ = self.scenarios[0].getReference(refxL_in,refu_in)
        self.refxR,_ = self.scenarios[0].getReference(refxR_in,refu_in)

        
        return self.refxL,self.refxR,self.refxT  
        
    
        
    def removeDeviation(self):
        """
        Centers the x-position around 0 (to fix nummerical issues)
        """
        # Store current values of changes
        self.egoPx = float(self.x_iter[0])
        # Alter initialization of MPC
        # # X-position
        self.x_iter[0] = 0
        for i in range(self.Nveh):
            self.x_lead[i,:] = self.x_lead[i,:] - self.egoPx
        # print("INFO: Ego position before removing deviation is:", self.traffic_state[0,:,:])
        self.traffic_state[0,:,:] = self.traffic_state[0,:,:] - self.egoPx
        self.traffic_state[0,:,:] = np.clip(self.traffic_state[0,:,:],-800,800)  
        # print("INFO: Ego position after removing deviation is:", self.traffic_state[0,:,:])
        
    def removeDeviation_y(self):
        self.traffic_state[1,:,:] = self.traffic_state[1,:,:] - self.init_bound
    
    def returnDeviation(self,X,U):
        """
        # Adds back the deviations that where removed in the above function
        """
        self.x_iter[0] = self.egoPx
        X[0,:] = X[0,:] + self.egoPx

        return X, U
    
    
    
    def setControllerParameters(self,version):
        """
        Sets traffic parameters, to be used in the MPC controllers
        """

        sign = np.zeros((self.Nveh,self.N+1))
        shift = np.zeros((self.Nveh,self.N+1))
        flip = np.ones((self.Nveh,self.N+1))

        if version == "leftChange":
            for ii in range(self.Nveh):
                for jj in range(self.N+1):
                    if self.traffic_state[1,jj,ii] > self.laneWidth :
                        sign[ii,jj] = -1
                        shift[ii,jj] = 2 * self.laneWidth
                        flip[ii,jj] = -1
                    elif self.traffic_state[1,jj,ii] < 0 :
                        sign[ii,jj] = 1
                        shift[ii,jj] = -self.laneWidth
                    else:
                        sign[ii,jj] = 1
                        shift[ii,jj] = 0

        elif version == "rightChange":
            for ii in range(self.Nveh):
                for jj in range(self.N+1):
                    if self.traffic_state[1,jj,ii] > self.laneWidth :
                        sign[ii,jj] = -1
                        shift[ii,jj] = 2 * self.laneWidth
                        flip[ii,jj] = -1
                    elif self.traffic_state[1,jj,ii] < 0 :
                        sign[ii,jj] = 1
                        shift[ii,jj] = -self.laneWidth
                    else:
                        sign[ii,jj] = -1
                        shift[ii,jj] = self.laneWidth
                        flip[ii,jj] = -1

        self.traffic_state[2,:,:] = sign.T
        self.traffic_state[3,:,:] = shift.T
        self.traffic_state[4,:,:] = flip.T
        
    def chooseController(self):
        """
        Main function, finds optimal choice of controller for the current step
        """
        self.egoLane = self.scenarios[0].getEgoLane()

        # Revoke controller usage if initialized unfeasible
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # ? ASK ERIK ABOUT THIS
        self.doTrailing = 1
        self.doLeft = 1
        self.doRight = 1
        
        
        self.doTrailing = 1
        if self.egoLane == 0:
            self.doLeft = 1
            self.doRight = 1
        elif self.egoLane == 1:
            self.doLeft = 1
            self.doRight = 0
        elif self.egoLane == -1:
            self.doLeft = 0
            self.doRight = 1

        self.paramLog = np.zeros((5,self.N+1,self.Nveh,3))
        # Initialize costs as very large number
        costT,costT_slack = 1e10,1e10
        costL,costL_slack = 1e10,1e10
        costR,costR_slack = 1e10,1e10
   
        self.removeDeviation()
        self.removeDeviation_y()
        
        if self.doTrailing:
            idx = self.scenarios[0].getLeadVehicle(self.traffic)  
            if len(idx) == 0:         #No leading vehicle,  Move barrier very far forward
                    x_traffic = DM(1,self.N+1)
                    x_traffic[0,:] = self.x_iter[0] + 200
            else:
                x_traffic = self.x_lead[idx[0],:]
                
            # print(self.Nveh)
            self.paramLog[0,:,idx,2] = x_traffic.full()
            u_testT, x_testT, costT, costT_slack=self.MPCs[2].solve(self.x_iter, self.refxT_out, self.refu_out, x_traffic)
            
        if self.doLeft:
            self.setControllerParameters(self.controllers[0].opts["version"])
            self.paramLog[:,:,:,0] = self.traffic_state
            # x_iter, refx_out, refu_out, traffic_x, traffic_y, traffic_sign, traffic_shift, traffic_flip  self.refxL_out
            u_testL, x_testL, costL, costL_slack=self.MPCs[0].solve(self.x_iter, self.refxL_out , self.refu_out, \
                                            self.traffic_state[0,:,:].T,self.traffic_state[1,:,:].T,self.traffic_state[2,:,:].T,
                                            self.traffic_state[3,:,:].T,self.traffic_state[4,:,:].T)
            
        if self.doRight:
            self.setControllerParameters(self.controllers[1].opts["version"])
            self.paramLog[:,:,:,1] = self.traffic_state
            # x_iter, refx_out, refu_out, traffic_x, traffic_y, traffic_sign, traffic_shift, traffic_flip  self.refxR_out
            u_testR, x_testR, costR, costR_slack=self.MPCs[1].solve(self.x_iter, self.refxR_out , self.refu_out, \
                                            self.traffic_state[0,:,:].T,self.traffic_state[1,:,:].T,self.traffic_state[2,:,:].T,
                                            self.traffic_state[3,:,:].T,self.traffic_state[4,:,:].T)
            
        #TODO: SIMPLE CHOICE OF THE CONTROLLER BASED ON THE COST
        # print("this is x_iter in controller", self.x_iter)
        
        # compare with the cost before and cose the best decision
        decision_i = np.argmin(np.array([costL+costL_slack,costR+costR_slack,costT+costT_slack]))
        # #! JUST FOR TEST:
        
        if len(self.decisionLog) >= self.changeHorizon:
            decision_i = self.getDecision([costL+costL_slack,costR+costR_slack,costT+costT_slack])
            self.decisionLog.insert(0,decision_i)
            self.decisionLog.pop()
        else:
            decision_i = np.argmin(np.array([costL+costL_slack,costR+costR_slack,costT+costT_slack]))
            self.decisionLog.insert(0,decision_i)
            
        print("INFO:  Controller cost",costL+ costL_slack,costR+costR_slack,costT+costT_slack,
              "Slack:",costL_slack,costR_slack,costT_slack,")")

        if decision_i == 0:
            X = x_testL
            U = u_testL
            print("INFO:  Optimal cost:",costL+ costL_slack)
        elif decision_i == 1:
            X = x_testR
            U = u_testR
            print("INFO:  Optimal cost:",costR+costR_slack)
        else:
            X = x_testT
            U = u_testT
            print("INFO:  Optimal cost:", [costT+costT_slack])
            
        print('INFO:  Decision: ',self.controllers[decision_i].opts["version"])

        X, U = self.returnDeviation(X,U)


        x_ok, u_ok, X = self.checkSolution(X,U)
        
            
        return u_ok, x_ok, X, decision_i
    
    
    
    def getTrafficState(self):
        return self.paramLog[:,0,:,:]

    def getErrorCount(self):
        """
        Returns the amount of strange solutions encountered
        """
        return self.errors

    def getGoalStatus(self):
        if self.doRouteGoalScenario == 1:
            if self.goalAccomplished == 1:
                return "Succesfully reached"
            else:
                return "Not reached"
        else:
            return "Was not considered"