"""
This one the the controller that contains trailing and lane change controller.
"""
import sys
import timeit
from casadi import *
import numpy as np
import shutil
import errno
from matplotlib import pyplot as plt
from MPC_tighten_bound import MPC_tighten_bound
from acados_template import *
from utils import *
import scipy.linalg

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
    
    def __init__(self, vehicle,traffic,scenario,N,opts,dt,controller_type="casadi"):
        self.controller_type = controller_type  #! casadi or acados
        self.vehicle = vehicle
        self.traffic = traffic
        self.scenario = scenario
        if opts["version"] == "trailing":
            self.scenario.setEgoLane(self.traffic)
            self.scenario.getLeadVehicle(self.traffic)
        # ! Get constraints and road information
        self.N = N
        self.Nveh = self.traffic.getNveh() # here, get the number of vehicles of the traffic scenario
        self.laneWidth = self.traffic.get_laneWidth()
        self.vehWidth,self.egoLength,self.L_tract, self.L_trail = self.vehicle.getSize()
        self.nx,self.nu,self.nrefx,self.nrefu = self.vehicle.getSystemDim()
        self.init_bound = self.vehicle.getInitBound()
        self.P0, self.process_noise, self.Possibility = self.vehicle.P0, self.vehicle.process_noise, self.vehicle.Possibility
        self.LQR_P, self.LQR_K = self.vehicle.calculate_Dlqr()
        self.roadMin, self.roadMax, self.laneCenters, _ = self.scenario.getRoad()
        self.egoTheta_max  = vehicle.xConstraints()[1][3]  #! In this situation, we do not have egoTheta_max. no trailor
        # ! get ref velocity 
        self.Vmax = scenario.getVmax()
        # ! get the cost param from the vehicle model
        self.Q, self.R = self.vehicle.getCostParam()
            
        #! P0, process_noise, possibility will be obtained from set_stochastic_mpc_params
        #! Used for tighten the MPC bound
        #! initial MPC_tighten_bound CLASS for the STATE CONSTRAINTS
        # self.A, self.B, self.C = self.vehicle.vehicle_linear_discrete_model(v=15, phi=0, delta=0)
        self.A, self.B, self.C = self.vehicle.calculate_AB_cog()
        self.D = np.eye(self.nx)
        self.MPC_tighten_bound = MPC_tighten_bound(self.A, self.B, self.D, np.diag(self.Q), 
                                                    np.diag(self.R), self.P0, self.process_noise, self.Possibility)
        if controller_type == "casadi":
            self.opts = opts 
            self.opti = Opti()
            # # Initialize opti stack
            self.x = self.opti.variable(self.nx,self.N+1)
            # self.u = self.opti.variable(self.nu,self.N)
            self.refx = self.opti.parameter(self.nrefx,self.N+1)
            self.refu = self.opti.parameter(self.nrefu,self.N)
            self.x0 = self.opti.parameter(self.nx,1)
            
            # ! here is a test for miu
            self.mu = self.opti.variable(self.nu,self.N)
            self.u = self.LQR_K @ self.x[:,:N] + self.mu
            # print(self.u.shape)
            # exit()
            
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
            
            
            
            
        elif controller_type == "acados":
            

            #! using opti.parameter to set the parameters
            self.opts = opts 
            self.opti = Opti()
            # # Initialize opti stack
            
            # # Initialize opti stack
            # self.u = self.opti.variable(self.nu,self.N)
            self.refx = self.opti.parameter(self.nrefx,self.N+1)
            self.refu = self.opti.parameter(self.nrefu,self.N)
            self.x0 = self.opti.parameter(self.nx,1)
            
            #! turn on/off the stochastic MPC
            self.stochasticMPC=1

            # ! change this according to the LC_MPC AND TRAILING_MPC
            if opts["version"] == "trailing":
                self.lead = self.opti.parameter(1,self.N+1)
            else:
                self.lead = self.opti.parameter(self.Nveh,self.N+1)
                self.traffic_x = self.opti.parameter(self.Nveh,self.N+1)
                self.traffic_y = self.opti.parameter(self.Nveh,self.N+1)
                self.traffic_sign = self.opti.parameter(self.Nveh,self.N+1)
                self.traffic_shift = self.opti.parameter(self.Nveh,self.N+1)
                self.traffic_flip = self.opti.parameter(self.Nveh,self.N+1)
            
            self.N = N
            #! acados model
            m_model = self.vehicle.model_acados(opts, self.Nveh)
            self.model = m_model
            
                
            #! ensure current working directory is current folder
            os.chdir(os.path.dirname(os.path.realpath(__file__)))
            self.acados_models_dir = './acados_models'
            self.safe_mkdir_recursive(os.path.join(os.getcwd(), self.acados_models_dir))
            acados_source_path = os.environ['ACADOS_SOURCE_DIR']
            sys.path.insert(0, acados_source_path)
            
            
            
            #! define nx, nu, ny, n_params
            self.ny = self.nx + self.nu
            n_params = len(self.model.p)
            
            
            
            #!create acados ocp
            self.ocp = AcadosOcp()
            self.ocp.acados_include_path = acados_source_path + '/include'
            self.ocp.acados_lib_path = acados_source_path + '/lib'
            self.ocp.model = self.model
            
            
            #! becareful! this is the defination of the dynamics
            if opts["version"] == "trailing":
                self.x_acados = self.model.x[0]
                self.y_acados = self.model.x[1]
                self.v_acados = self.model.x[2]
                self.theta_acados = self.model.x[3]
                self.x_lead_acados = self.model.x[4]
                self.temptX_acados = self.model.x[5]
                print("INFO:  x_lead_acados is:", self.model.x)
            elif opts["version"] == "leftChange" or opts["version"] == "rightChange":
                self.x_acados = self.model.x[0]
                self.y_acados = self.model.x[1]
                self.v_acados = self.model.x[2]
                self.theta_acados = self.model.x[3]
                self.surrounding_vehicle_states = self.model.x[4:-2]  # Adjust indices as necessary based on your model's structure
                self.temptX_acados = self.model.x[-2]  # Second to last state assuming 'temptX' is here
                self.temptY_acados = self.model.x[-1]  # Last state assuming 'temptY' is here
                # print("INFO:  x_acados is:", self.model.x)
                # print("INFO:  u_acados is:", self.model.u)


            
            self.ocp.dims.N = self.N
            self.ocp.solver_options.tf = self.N*dt
            #! cost type
 
            #! get len of x and u of the acados model
            self.nx_acados = self.model.x.size()[0]
            self.nu_acados = self.model.u.size()[0]


            Q_reduced = np.zeros((self.nx_acados, self.nx_acados))  # Initialize a zero matrix for Q
            np.fill_diagonal(Q_reduced[:self.nx, :self.nx], self.Q[:self.nx])  # Fill the diagonal for the first four states
            #fill the diagonal for the first two inputs
            
            R_reduced = np.zeros((self.nu_acados, self.nu_acados))  # Initialize a zero matrix for R
            np.fill_diagonal(R_reduced[:self.nu, :self.nu], self.R[:self.nu])  # Fill the diagonal for the first two inputs
            # print("INFO:  R_reduced is:", R_reduced)
            
            R_du_reduced = np.zeros((self.nu_acados, self.nu_acados))  # Initialize a zero matrix for R_du
            np.fill_diagonal(R_du_reduced[:self.nu, :self.nu], self.R[:self.nu])  # Fill the diagonal for the first two inputs
            du_penalty = self.vehicle.setAcadosPenal()
            R_du_reduced = du_penalty*R_du_reduced
            # print("INFO:  R_du_reduced is:", R_du_reduced)
    
            # matrix for Q_reduced, R and R_du
            # w= diag([Q_reduced, R_reduced, R_du_reduced])
            W = np.block([
                    [Q_reduced, np.zeros((self.nx_acados, self.nu_acados)), np.zeros((self.nx_acados, self.nu_acados))], 
                    [np.zeros((self.nu_acados, self.nx_acados)), R_reduced, np.zeros((self.nu_acados, self.nu_acados))],
                    [np.zeros((self.nu_acados, self.nx_acados)), np.zeros((self.nu_acados, self.nu_acados)), R_du_reduced]])

            # np.set_printoptions(threshold=np.inf, linewidth=np.inf)
            # print("INFO:  ocp.cost.W is:")
            # print(W)
            #! cost matrix finished!

            self.ocp.cost.W = W
            
            self.ocp.cost.W_e = Q_reduced  # For the terminal cost

            self.ocp.cost.Vx = np.zeros((self.nx_acados+2*self.nu_acados, self.nx_acados))
            self.ocp.cost.Vx[:self.nx, :self.nx] = np.eye(self.nx)  # Map only the first four states
            
            
            
            self.ocp.cost.Vu = np.zeros((self.nx_acados+2*self.nu_acados, self.nu_acados))
            self.ocp.cost.Vu[self.nx_acados:self.nx_acados+self.nu, :self.nu] = np.eye(self.nu)
            
            self.ocp.cost.Vx_e = np.zeros((self.nx_acados, self.nx_acados))
            self.ocp.cost.Vx_e[:self.nx, :self.nx] = np.eye(self.nx)  # Map only the first four states
            


            #! define initial constraints
            lbx,ubx = self.vehicle.xConstraints()
            lbu,ubu = self.vehicle.uConstraints()
            # print("INFO: lbx is:", lbx)
            # print("INFO: ubx is:", ubx)
            # print("INFO: lbu is:", lbu)
            # print("INFO: ubu is:", ubu)
            lbx_full = np.zeros((self.nx_acados,))
            lbx_full[:self.nx] = lbx
            lbx_full[self.nx:] = -1e10
            
            # print("INFO: lbx_full is:", lbx_full)
            
            ubx_full = np.zeros((self.nx_acados,))
            ubx_full[:self.nx] = ubx
            ubx_full[self.nx:] = 1e10
            
            lbu_full = np.zeros((self.nu_acados,))
            lbu_full[:self.nu] = lbu
            lbu_full[self.nu:] = -1e10
            
            ubu_full = np.zeros((self.nu_acados,))
            ubu_full[:self.nu] = ubu
            ubu_full[self.nu:] = 1e10
            
            
            # print("INFO: lbx_full is:", lbx_full)
            
            
            
            self.ocp.constraints.lbu = np.array(lbu_full)
            self.ocp.constraints.ubu = np.array(ubu_full)
            self.ocp.constraints.lbx = np.array(lbx_full)
            self.ocp.constraints.ubx = np.array(ubx_full)
            # print("INFO: lbu is:", ocp.constraints.lbu.shape)
            # print("INFO: ubu is:", ocp.constraints.ubu)
            #! add penalty for the slack variable
            ns = self.nx_acados
            self.ocp.constraints.idxsbx = np.array(range(self.nx_acados))
            
            penalty_utils = self.vehicle.setAcadosSlack()
            self.ocp.cost.zl = penalty_utils * np.ones((ns,))
            self.ocp.cost.zu = penalty_utils * np.ones((ns,))
            self.ocp.cost.Zl = 1e0 * np.ones((ns,))
            self.ocp.cost.Zu = 1e0 * np.ones((ns,))
            
            # idxbu =  np.array([0,1,2,.....i]) for i in range(self.nu_acados)
            self.ocp.constraints.idxbu = np.array(range(self.nu_acados))
            self.ocp.constraints.idxbx = np.array(range(self.nx_acados))
            
            
            #! define the reference
            x_ref = np.zeros(self.nx_acados)
            u_ref = np.zeros(self.nu_acados)
            # initial state
            self.ocp.constraints.x0 = x_ref
            self.ocp.cost.yref = np.concatenate((x_ref, u_ref, np.zeros(self.nu_acados)))
            self.ocp.cost.yref_e = x_ref
            
            
            #! solver options
            # integrator option
            self.ocp.solver_options.integrator_type = 'ERK'
        
            # nlp solver options
            self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
            self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'
            self.ocp.solver_options.nlp_solver_max_iter = 400 
            
            # qp solver options
            self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
            self.ocp.solver_options.qp_solver_iter_max = 100  
            self.ocp.solver_options.print_level = 0
                    
            # compile acados ocp
            json_file = os.path.join('./'+m_model.name+'_acados_ocp.json')
            self.solver = AcadosOcpSolver(self.ocp, json_file=json_file)
            self.integrator = AcadosSimSolver(self.ocp, json_file=json_file)
            print("INFO:  Acados Controller is created !!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            
            #! Here finised temporarily  TO Check Later
            
            
##################################################################################################
        
    def setStateEqconstraints(self):
        """
        Set state equation constraints, using the LTI model,  ref_v=15, ref_phi=0, ref_delta=0
        """
        for i in range(self.N):
            A_d, B_d, G_d = self.A, self.B, self.C
            self.opti.subject_to(self.x[:, i+1] == A_d @ self.x[:, i] + B_d @ self.u[:, i] + G_d)
            # self.opti.subject_to(self.x[:, i+1] == self.F_x(self.x[:,i],self.u[:,i]))
        self.opti.subject_to(self.x[:, 0] == self.x0)
        # self.opti.subject_to(self.u == self.LQR_K @ self.x + self.mu) 
        
        #! check withe ERIK
        # for i in range(self.N):
        #     self.opti.subject_to(self.u[:,i] == self.LQR_K @ self.x[:,i] + self.mu[:,i])
        
    def setStateEqconstraints_acados(self):
        """
        Set state equation constraints, using the ACAODS model
        """
        
        #TODO:CHANGE THIS LATER
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # self.opti.set_value(self.x0, np.zeros((self.nx,1)))
        x_current = self.opti.value(self.x0)
        print(x_current)
        print(x_current.shape)
        print("###################################################")
        self.solver.set(0, 'lbx', x_current)
        self.solver.set(0, 'ubx', x_current)

    def setInEqConstraints_val(self, H_up=None, upb=None, H_low=None, lwb=None):
        """
        Set inequality constraints values.
        """
        # Default or custom constraints
        self.H_up = H_up if H_up is not None else [np.array([[1], [0], [0], [0]]), np.array([[0], [1], [0], [0]]), np.array([[0], [0], [1], [0]]), np.array([[0], [0], [0], [1]])]
        self.upb = upb if upb is not None else np.array([[5000], [5000], [30], [3.14/8]])
        
        self.H_low = H_low if H_low is not None else [np.array([[-1], [0], [0], [0]]), np.array([[0], [-1], [0], [0]]), np.array([[0], [0], [-1], [0]]), np.array([[0], [0], [0], [-1]])]
        self.lwb = lwb if lwb is not None else np.array([[5000], [5000], [0], [3.14/8]])
        
        
        
#################################################        setInEqConstraints           #################################################


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
            self.opti.subject_to(self.x[:, i] <= DM(self.tightened_bound_N_list_up[i].reshape(-1, 1)))
            self.opti.subject_to(self.x[:, i] >= DM(self.tightened_bound_N_list_lw[i].reshape(-1, 1))) 
        # set the constraints for the input  [-3.14/8,-0.7*9.81],[3.14/8,0.05*9.81]
        #! set the constraints for the INPUT
        self.opti.subject_to(self.opti.bounded(lbu, self.u, ubu))
        #! extra constraints for the V_MAX
        self.opti.subject_to(self.opti.bounded(0,self.x[2,:],self.scenario.vmax))
        
        
    def setInEqConstraints_acados(self):
        """
        Set inequality constraints, only for default constraints and tihgtened  default  constraints
        
        """
        lbx,ubx = self.vehicle.xConstraints()   
        # ! element in lbx and ubx should be positive
        lbx = [abs(x) for x in lbx]
        self.setInEqConstraints_val(H_up=None, upb=np.array(ubx).reshape(4,1), H_low=None, lwb=np.array(lbx).reshape(4,1))  # Set default constraints
        lbu,ubu = self.vehicle.uConstraints()
        self.tightened_bound_N_list_up = self.MPC_tighten_bound.tighten_bound_N(self.P0, self.H_up, self.upb, self.N, 1)
        print("INFO:  tightened_bound_N_list_up is:", self.tightened_bound_N_list_up)
        self.tightened_bound_N_list_lw = self.MPC_tighten_bound.tighten_bound_N(self.P0, self.H_low, self.lwb, self.N, 0)
        #! create a np array for store the v and delta constraints
        
        self.constraintStore = np.zeros((self.N,2*self.nx))  #(lbx, ubx) for each step
        
        for i in range(1, self.N):
            #! lbx and ubx  shape  should be the (nx,)
            lbx_array = np.array(self.tightened_bound_N_list_lw[i]).reshape(4,)
            self.constraintStore[i, :self.nx] = lbx_array
            # self.solver.constraints_set(i, 'lbx', lbx_array)
            ubx_array = np.array(self.tightened_bound_N_list_up[i]).reshape(4,)
            # self.solver.constraints_set(i, 'ubx', ubx_array)
            self.constraintStore[i, self.nx:] = ubx_array
            
            
            # print("INFO: lbx_array is:", lbx_array)
            # print("INFO: ubx_array is:", ubx_array)
        # print("INFO:  setInEqConstraints_acados is done")
        # print("constraintStore is:", self.constraintStore)

##################################################          setTrafficConstraints            ################################################

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

            # Set default road boundries, given that there is a "phantom vehicle" in the lane we can not enter
            d_lat_spread =  self.L_trail* np.tan(self.egoTheta_max)
            d_lat_spread = 0
            if self.opts["version"] == "leftChange":
                self.y_lanes = [self.init_bound + self.vehWidth/3+d_lat_spread, self.init_bound + 2*self.laneWidth-self.vehWidth/3-d_lat_spread]
            elif self.opts["version"] == "rightChange":
                self.y_lanes = [self.init_bound -self.laneWidth + self.vehWidth/3+d_lat_spread, self.init_bound + self.laneWidth-self.vehWidth/3-d_lat_spread]
            # self.vehWidth/2 is  too strict, to be self.vehWidth/3
            # self.opti.subject_to(self.opti.bounded(self.y_lanes[0],self.x[1,:],self.y_lanes[1]))

        elif self.scenario.name == 'trailing':
            T = self.scenario.Time_headway
            self.scenario.setEgoLane(self.traffic)
            self.scenario.getLeadVehicle(self.traffic)
            self.IDM_constraint_list = self.S(self.lead) + self.traffic_slack[0,:]-T * self.x[2,:]
            self.opti.subject_to(self.x[0,:]  <= self.S(self.lead) + self.traffic_slack[0,:]-T * self.x[2,:])
            # tighten the TRAILING CONSTRAINTS  
            
            
    
    #TODO: set the traffic constraints for the acados carefully
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!       
    
    #! Adding the change of it as states
    #! 1. the state of the model should have two part: trailing and lane change
    #! 2. the constraints should be set for the trailing and lane change
    
    
    
    def setTrafficConstraints_acados(self):
        """
        Set the traffic constraints for the acados
        #! Saving the minimal boundary into the self.constraintStore as the constraints of the solver!
        """
        # if self.stochasticMPC:
        #     self.temp_x, self.tempt_y = self.MPC_tighten_bound.getXtemp(self.N ), self.MPC_tighten_bound.getYtemp(self.N )
        #     # print("INFO: temp_x is:", self.temp_x)
        #     # print("INFO: temp_y is:", self.tempt_y)
        #     self.S =self.scenario.constrain_tightened(self.traffic,self.opts,self.temp_x, self.tempt_y)
        # else:
        #     self.S = self.scenario.constraint(self.traffic,self.opts)
        
        if self.scenario.name == 'simpleOvertake':
            
            
            
            
            
            
            #! DO NOT TAKE EGO VEHICLE INTO ACCOUNT
            for i in range(self.Nveh):
                if i ==1: continue #! avoid putting the ego vehicle in the list
                self.opti.subject_to(self.traffic_flip[i,:] * self.x[1,:] 
                                    >= self.traffic_flip[i,:] * self.S[i](self.x[0,:], self.traffic_x[i,:], self.traffic_y[i,:],
                                        self.traffic_sign[i,:], self.traffic_shift[i,:]) +  self.traffic_slack[i,:])
            
            
            
            
            
            
            #! DO NOT TAKE EGO VEHICLE INTO ACCOUNT
            for i in range(self.Nveh):
                #constr_h should be Nveh-1 vertcat
                constr_h_container = []
                if i ==1: continue #! avoid putting the ego vehicle in the list
                # constr_h_container.append(self.traffic_flip[i,:] * self.S[i](self.x[0,:], self.traffic_x[i,:], self.traffic_y[i,:],
                #                         self.traffic_sign[i,:], self.traffic_shift[i,:]) -self.traffic_flip[i,:] * self.x[1,:] )
                
            # h_max = np.zeros((self.Nveh-1))
            # h_min = np.zeros((self.Nveh-1))
            
            
                

                
                
                constraintDirection = np.sign(self.opti.value(self.traffic_flip[i,:]))  #! 1 or -1
                for j in range(1,self.N):
                    if constraintDirection > 0:
                        self.constraintStore[j,1] = max(self.constraintStore[j,1], self.opti.value(self.S[i](self.x[0,j], self.traffic_x[i,j], self.traffic_y[i,j], self.traffic_sign[i,j], self.traffic_shift[i,j])) )
                    elif constraintDirection < 0:
                        self.constraintStore[j,5] = min(self.constraintStore[j,0], self.opti.value(self.S[i](self.x[0,j], self.traffic_x[i,j], self.traffic_y[i,j], self.traffic_sign[i,j], self.traffic_shift[i,j])) )
               
               
         #TODO: FIND SOME WAY TO APPROACH THIS SELF.X  ISSUE!!!!!!!!!!!!       SELF.X IS OPTI.VARIABLE!!!!!!!!!!!!!!!!!!!
                
        elif self.scenario.name == 'trailing':
            # self.opti.subject_to(self.x[0,:]  <= self.S(self.lead) + self.traffic_slack[0,:]-T * self.x[2,:])
            
            T = self.scenario.Time_headway
            self.scenario.setEgoLane(self.traffic)
            self.scenario.getLeadVehicle(self.traffic)

            self.min_distx = self.scenario.min_distx
            leadWidth, leadLength = self.traffic.getVehicles()[0].getSize()
            self.L_tract = self.scenario.L_tract
            
            safeDist = self.min_distx + leadLength + self.L_tract + self.temptX_acados
            S_func = self.x_lead_acados - safeDist
            constr_h = self.x_acados - S_func + T * self.v_acados   

            self.ocp.model.con_h_expr = constr_h

            lh =-1e3*np.ones((1,1))
            uh =np.zeros((1,1))
            
            self.ocp.constraints.lh = lh
            self.ocp.constraints.uh = uh
            
    def OverTakeConstraints(self, px, v0_i, traffic_x, traffic_y, traffic_sign, traffic_shift):
        """
        Set the constraints for the overtake issue
        """
        # v0_i = traffic.getVehicles()[i].v0
        # traffic.getVehicles
        
        L_tract = 8.46  
        leadWidth = 2.032
        T = self.scenario.Time_headway
        l_front,l_rear = 4.78536/2 , 4.78536/2
        self.min_distx = self.scenario.min_distx
        init_bound = self.vehicle.getInitBound()

        # Define vehicle specific constants
        alpha_0 = traffic_sign * (traffic_sign*(traffic_y-traffic_shift)+leadWidth/2)
        alpha_1 = l_front+ L_tract/2 + v0_i * self.scenario.Time_headway + self.min_distx 
        alpha_2 = l_rear + L_tract/2+ v0_i * self.scenario.Time_headway+ self.min_distx 
        alpha_3 = traffic_shift
        d_w_e = (self.vehWidth/2)*traffic_sign
        # Construct function
        func1 = alpha_0 / 2 * tanh(px - traffic_x + alpha_1)+alpha_3/2
        func2 = alpha_0 / 2 * tanh(traffic_x - px + alpha_2)+alpha_3/2
        S = func1+func2 + d_w_e
        # !SHIFT ACCORDING TO THE INIT_BOUND
        S = S + init_bound 
        pass



        


############################################            COST            ######################################################
    
        
    def setCost(self):
        L,Lf = self.vehicle.getCost()
        Ls = self.scenario.getSlackCost()
        self.costMain = getTotalCost(L,Lf,self.x,self.u,self.refx,self.refu,self.N)
        self.costSlack = getSlackCost(Ls,self.traffic_slack)
        self.total_cost = self.costMain + self.costSlack
        self.opti.minimize(self.total_cost)    
        
    def setCost_acados(self):
        """
        Set the cost function for the acados
        #! for now, no need to set the cost function.....it is defined in the initialization
        """
        
        pass
 
 
 ###############################################################################################################
 
 
 
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
    
    def setController_acados(self):
        """
        Sets all constraints
        """
        #TODO: becareful about the constraints!!!!!!!!!!!!!!!!!!!!!!!!!
        #! add equality constraints later
        # self.setStateEqconstraints_acados()
        self.setInEqConstraints_acados()
        self.setTrafficConstraints_acados()  #! execute the function sequentially!!!!!!!!!!!!!
        
        
        #use self.constraintStore to set the constraints
        print("INFO:  self.constraintStore", self.constraintStore)
        for i in range(1, self.N):
            self.solver.constraints_set(i, 'lbx', self.constraintStore[i, :self.nx])
            self.solver.constraints_set(i, 'ubx', self.constraintStore[i, self.nx:])
        
        print("INFO:  setConstraints_acados is done")
        pass
    
    
    
##################################################################################################
    
    
    
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
            # print("this is traffic_x", traffic_x)
            # print("this is traffic_y", traffic_y)
            # print("this is traffic_sign", traffic_sign)
            # print("this is traffic_shift", traffic_shift)
            

            
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
            mu_opt = sol.value(self.mu)
            u_opt = sol.value(self.u)
            x_opt = sol.value(self.x)
            #! get u_opt from the LQR_K and mu_opt
            u_opt = self.LQR_K @ x_opt[:,:self.N] + mu_opt
            costMain = sol.value(self.costMain)
            costSlack = sol.value(self.costSlack)
            return u_opt, x_opt, costMain, costSlack
        except Exception as e:
            print(f"An error occurred: {e}")
            self.opti.debug()
            return None, None, None
    
    
    
    def solve_acados(self, *args, **kwargs):
        """
        Solve the optimization problem with flexible inputs based on configuration in self.opts.
        """
        simX = np.zeros((self.nx, self.N+1))
        simU = np.zeros((self.nu,self.N))
        
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
            
        
        #! set constraints
        ##########################################################
        self.setController_acados()
        exit()
        ##########################################################
        
            
        xs = self.opti.value(self.refx)
        xs_between = np.concatenate((xs, np.zeros(4)))
        
        #close loop
        self.solver.set(self.N, 'yref', xs)
        for i in range(self.N):
            self.solver.set(i, 'yref', xs_between)
            
            
        status = self.solver.solve()

        if status != 0 :
            raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status))
        
        #! save total X and U
        for i in range(self.N):
            simX[:,i] = self.solver.get(i, 'x')
            simU[:,i] = self.solver.get(i, 'u')
        cost_value = self.solver.get_cost()
        
        return simU, simX, cost_value
        
        
        
##################################################################################################
      
    def safe_mkdir_recursive(self,directory, overwrite=False):
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError as exc:
                if exc.errno == errno.EEXIST and os.path.isdir(directory):
                    pass
                else:
                    raise
        else:
            if overwrite:
                try:
                    shutil.rmtree(directory)
                except:
                    print('Error while removing directory {}'.format(directory))       
        

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
            
            
        self.doTrailing = 1
        self.doLeft = 0
        self.doRight = 0

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
            if self.MPCs[2].controller_type == "casadi":
                u_testT, x_testT, costT, costT_slack=self.MPCs[2].solve(self.x_iter, self.refxT_out, self.refu_out, x_traffic)
            elif self.MPCs[2].controller_type == "acados":
                u_testT, x_testT, costT=self.MPCs[2].solve_acados(self.x_iter, self.refxT_out, self.refu_out, x_traffic)
            
        if self.doLeft:
            self.setControllerParameters(self.controllers[0].opts["version"])
            self.paramLog[:,:,:,0] = self.traffic_state
            # x_iter, refx_out, refu_out, traffic_x, traffic_y, traffic_sign, traffic_shift, traffic_flip  self.refxL_out
            if self.MPCs[0].controller_type == "casadi":
                u_testL, x_testL, costL, costL_slack=self.MPCs[0].solve(self.x_iter, self.refxL_out , self.refu_out, \
                                                self.traffic_state[0,:,:].T,self.traffic_state[1,:,:].T,self.traffic_state[2,:,:].T,
                                                self.traffic_state[3,:,:].T,self.traffic_state[4,:,:].T)
            elif self.MPCs[0].controller_type == "acados":
                u_testL, x_testL, costL=self.MPCs[0].solve_acados(self.x_iter, self.refxL_out , self.refu_out, \
                                                self.traffic_state[0,:,:].T,self.traffic_state[1,:,:].T,self.traffic_state[2,:,:].T,
                                                self.traffic_state[3,:,:].T,self.traffic_state[4,:,:].T)
            
        if self.doRight:
            self.setControllerParameters(self.controllers[1].opts["version"])
            self.paramLog[:,:,:,1] = self.traffic_state
            # x_iter, refx_out, refu_out, traffic_x, traffic_y, traffic_sign, traffic_shift, traffic_flip  self.refxR_out
            if self.MPCs[1].controller_type == "casadi":
                u_testR, x_testR, costR, costR_slack=self.MPCs[1].solve(self.x_iter, self.refxR_out , self.refu_out, \
                                                self.traffic_state[0,:,:].T,self.traffic_state[1,:,:].T,self.traffic_state[2,:,:].T,
                                                self.traffic_state[3,:,:].T,self.traffic_state[4,:,:].T)
            elif self.MPCs[1].controller_type == "acados":
                u_testR, x_testR, costR=self.MPCs[1].solve_acados(self.x_iter, self.refxR_out , self.refu_out, \
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