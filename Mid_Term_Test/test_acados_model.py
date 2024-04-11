import time
import sys
from casadi import *
from Controllers import makeController, makeDecisionMaster #, makeControllerAcados
from vehicle_model import car_VehicleModel
from Traffic import Traffic
from Scenarios import trailing, simpleOvertake
from utils import *
from acados_template import *
import scipy.linalg
import timeit
import shutil
import errno


sys.path.append(r'/mnt/c/Users/A490243/CARLA/CARLA_Latest/WindowsNoEditor/PythonAPI/carla')
from agents.navigation.controller import VehiclePIDController
# import helpers
from util.utils import *
makeMovie = True
directory = r"C:\Users\A490243\Desktop\Master_Thesis\Figure\crazy_traffic_mix3.gif"


'''
#! run this every time
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/mnt/c/Users/A490243/acados/lib"
export ACADOS_SOURCE_DIR="/mnt/c/Users/A490243/acados"
'''



## ! --------------------------------------System initialization--------------------------------------------
dt = 0.02               # Simulation time step (Impacts traffic model accuracy)
desired_interval = dt
dt_PID = dt/5              # Time step for the PID controller
f_controller = 10            # Controller update frequency, i.e updates at each t = dt*f_controller
N =  12        # MPC Horizon length
laneWidth = 3.5

ref_vx = 54/3.6             # Higway speed limit in (m/s)
ref_velocity=ref_vx
q_traffic_slack = 1e5
traffic = Traffic(N,dt)
#! for now ignore traffic settings
# velocities = {
#         'normal': carla.Vector3D(0.75 * ref_vx, 0, 0),
#         'passive': carla.Vector3D(0.65 * ref_vx, 0, 0),
#         'aggressive': carla.Vector3D(0.9*ref_vx, 0, 0),  # Equal to 1.0 * ref_velocity for clarity
#         'reference': carla.Vector3D(ref_vx, 0, 0)  # Specifically for the truck
#     }
# spawned_vehicles, center_line = traffic.setup_complex_carla_environment()
# traffic.set_velocity(velocities)
# Nveh = traffic.getDim()
# vehList = traffic.getVehicles()
# time.sleep(1)
# px_init,py_init,vx_init=traffic.getStates()[:3,1] # get the initial position of the truck
# truck = traffic.getEgo()  # get the ego vehicle


## ! -----------------------------------initialize the local controller-----------------------------------------
# local_controller = VehiclePIDController(truck, 
#                                         args_lateral = {'K_P': 1.2, 'K_I': 0.2, 'K_D': 0.5, 'dt': dt_PID}, 
#                                         args_longitudinal = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.5, 'dt': dt_PID})

'''The following code is to test the controller with the carla environment'''
# while(True):
#     control_Truck = local_controller.run_step(ref_vx*3.6-50, 143.318146, False)
#     truck.apply_control(control_Truck)


## ! -----------------------------------initialize the VehicleModel-----------------------------------------
vehicleADV = car_VehicleModel(dt,N)
vehWidth,vehLength,L_tract,L_trail = vehicleADV.getSize()
nx,nu,nrefx,nrefu = vehicleADV.getSystemDim()
# Set Cost parameters
Q_ADV = [0,40,3e2,5]                            # State cost, Entries in diagonal matrix
R_ADV = [5, 5]                                   # Input cost, Entries in diagonal matrix
q_ADV_decision = 100
# vehicleADV.cost(Q_ADV,R_ADV)
# vehicleADV.costf(Q_ADV)
# L_ADV,Lf_ADV = vehicleADV.getCost()
vehicleADV.cost_new(Q_ADV,R_ADV)
LQR_P, LQR_K = vehicleADV.calculate_Dlqr()
vehicleADV.costf(Q_ADV)
L_ADV,Lf_ADV = vehicleADV.getCost()


## ! --------------------------------------- Problem definition ---------------------------------------------
scenarioTrailADV = trailing(vehicleADV,N,lanes = 3,v_legal = ref_vx, laneWidth=laneWidth)
scenarioTrailADV.slackCost(q_traffic_slack)
#TODO: ADD scenarioADV LATTER
scenarioADV = simpleOvertake(vehicleADV,N,lanes = 3,v_legal = ref_vx,laneWidth=laneWidth)
scenarioADV.slackCost(q_traffic_slack)

#! get road INFOS
roadMin, roadMax, laneCenters, _ = scenarioTrailADV.getRoad()
#! initilize the ego vehicle
vehicleADV.setRoad(roadMin,roadMax,laneCenters)
px_init,py_init= 0,0
vehicleADV.setInit([px_init,py_init],ref_vx )

sigma_process=0.01
sigma_measurement=0.01
Q_0=np.eye(nx)*sigma_process**2
Q_0[0,0]=0.3  # x bound is [0, 3]
Q_0[1,1]=0.05  # y bound is [0, 0.1]
Q_0[2,2]=0.5  # v bound is [0, 1.8]
Q_0[3,3]=0.01**2  # psi bound is [0, 0.05]
# ! get the param for the stochastic mpc
P0, _, possibility = set_stochastic_mpc_params()
vehicleADV.setStochasticMPCParams(P0, Q_0, possibility)
#! -----------------------------------------------------------------
#! -----------------------------------------------------------------
#!      Formulate optimal control problem using opti framework
#! -----------------------------------------------------------------
#! -----------------------------------------------------------------


def safe_mkdir_recursive(directory, overwrite=False):
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


class model_acados(object):
        """
        define the model for acados
        """
        
        def __init__(self,):
            self.length=1
            self.name = "Truck_bicycle"
            x = SX.sym('x')
            y = SX.sym('y')
            v = SX.sym('v')
            theta = SX.sym('theta')
            states = vertcat(x,y,v,theta)
            a = SX.sym('a')
            delta = SX.sym('delta')
            controls = vertcat(delta,a)
            rhs = [v*cos(theta),v*sin(theta),a,v*tan(delta)/self.length]

            #function
            f = Function('f', [states, controls], [vcat(rhs)], ['state', 'control_input'], ['rhs'])
            #acasdo model
            x_dot = SX.sym('x_dot', len(rhs))
            f_impl = x_dot - f(states, controls)
            
            model = AcadosModel()
            model.f_expl_expr = f(states, controls)
            model.f_impl_expr = f_impl
            model.x = states
            model.u = controls
            model.xdot = x_dot
            model.p = []
            model.name = self.name
            self.model = model

class makeControllerAcados:
    """
    #! Creates a MPC using acados based on current vehicle, traffic and scenario
    """
    """
    ██████╗  ██████╗ ██████╗      ██████╗ ██╗     ███████╗███████╗███████╗    ███╗   ███╗██████╗  ██████╗            
    ██╔════╝ ██╔═══██╗██╔══██╗    ██╔══██╗██║     ██╔════╝██╔════╝██╔════╝    ████╗ ████║██╔══██╗██╔════╝            
    ██║  ███╗██║   ██║██║  ██║    ██████╔╝██║     █████╗  ███████╗███████╗    ██╔████╔██║██████╔╝██║                 
    ██║   ██║██║   ██║██║  ██║    ██╔══██╗██║     ██╔══╝  ╚════██║╚════██║    ██║╚██╔╝██║██╔═══╝ ██║                 
    ╚██████╔╝╚██████╔╝██████╔╝    ██████╔╝███████╗███████╗███████║███████║    ██║ ╚═╝ ██║██║     ╚██████╗            
     ╚═════╝  ╚═════╝ ╚═════╝     ╚═════╝ ╚══════╝╚══════╝╚══════╝╚══════╝    ╚═╝     ╚═╝╚═╝      ╚═════╝                                                                                                                                                                                                            
    """
    
    def __init__(self,vehicle,traffic,scenario,N,opts,dt):
        self.vehicle = vehicle
        self.traffic = traffic
        self.scenario = scenario
        self.opts = opts 
        
        # Get constraints and road information
        self.N = N
        #! for now comment this
        # self.Nveh = self.traffic.getNveh() # here, get the number of vehicles of the traffic scenario
        # self.laneWidth = self.traffic.get_laneWidth()

        
        #! acados model
        m_model = model_acados()
        model = m_model.model
        # Ensure current working directory is current folder
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        self.acados_models_dir = './acados_models'
        safe_mkdir_recursive(os.path.join(os.getcwd(), self.acados_models_dir))
        acados_source_path = os.environ['ACADOS_SOURCE_DIR']
        sys.path.insert(0, acados_source_path)
        self.nx = model.x.size()[0]
        print(self.nx)
        self.nu = model.u.size()[0]
        print(self.nu)
        self.ny = self.nx + self.nu
        print(self.ny)
        n_params = len(model.p)
        print(n_params)
        
        #!create acados ocp
        ocp = AcadosOcp()
        ocp.acados_include_path = acados_source_path + '/include'
        ocp.acados_lib_path = acados_source_path + '/lib'
        ocp.model = model
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.N*dt
        
        #! initialize parameters
        ocp.dims.np = n_params
        ocp.parameter_values = np.zeros(n_params)
        
        #! cost type
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        Q = np.diag(self.vehicle.Q)
        R = np.diag(self.vehicle.R)
        R_du = 1e2*np.diag(self.vehicle.R)
        # print(Q)
        # print(R)
        # ocp.cost.W = np.block([[Q, np.zeros((self.nx, self.nu))], [np.zeros((self.nu, self.nx)), R]])
        ocp.cost.W = np.block([
                    [Q, np.zeros((self.nx, self.nu)), np.zeros((self.nx, self.nu))], 
                    [np.zeros((self.nu, self.nx)), R, np.zeros((self.nu, self.nu))],
                    [np.zeros((self.nu, self.nx)), np.zeros((self.nu, self.nu)), R_du]])
        print(ocp.cost.W)

        ocp.cost.W_e = Q
        
        ocp.cost.Vx = np.zeros((self.ny+self.nu, self.nx))
        matrix_Q = np.eye(self.nx)
        # matrix_Q[0, 0] = 0
        ocp.cost.Vx[:self.nx, :self.nx] = matrix_Q
        print(ocp.cost.Vx)
        ocp.cost.Vu = np.zeros((self.ny+self.nu, self.nu))
        ocp.cost.Vu[self.nx:self.nx+self.nu, :self.nu] = np.eye(self.nu)
        print(ocp.cost.Vu)
        ocp.cost.Vx_e = np.eye(self.nx)
        
        #! set constraints
        
        #! for now, all of them are infinity
        # large_value = 5000
        ocp.constraints.lbu = -np.array([np.pi/4, 0.5*9.8])
        ocp.constraints.ubu = 1 * np.array([np.pi/4, 0.5*9.8])
        original_ubx_2 = 6  # Original upper bound before softening
        ocp.constraints.lbx = -1 * np.ones((self.nx, ))
        ocp.constraints.ubx = np.array([2000, 2000, original_ubx_2, 1])
        #! add penalty to the slack
        # TODO: add slack variables
        nx = self.nx
        ocp.constraints.idxsbx = np.array(range(nx))
        ns = nx
        penalty_utils = 1e5
        ocp.cost.zl = penalty_utils * np.ones((ns,))
        ocp.cost.zu = penalty_utils * np.ones((ns,))
        ocp.cost.Zl = 1e0 * np.ones((ns,))
        ocp.cost.Zu = 1e0 * np.ones((ns,))
        # ocp.constraints.lsbx = -slack_bounds
        # ocp.constraints.usbx = slack_bounds
        # ocp.constraints.idxsbx = np.array([2])



        
        print(ocp.constraints.ubx)

        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.idxbx = np.array([0, 1, 2, 3])
        
    

        
        x_ref = np.zeros(self.nx)
        u_ref = np.zeros(self.nu)
        # initial state
        ocp.constraints.x0 = x_ref
        # ocp.cost.yref = np.concatenate((x_ref, u_ref))
        ocp.cost.yref = np.concatenate((x_ref, u_ref, np.zeros(self.nu)))
        ocp.cost.yref_e = x_ref
        
        # solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.print_level = 0
                
        # compile acados ocp
        json_file = os.path.join('./'+model.name+'_acados_ocp.json')
        self.solver = AcadosOcpSolver(ocp, json_file=json_file)
        self.integrator = AcadosSimSolver(ocp, json_file=json_file)
        # self.integrator.set('x', np.array([0, 0, 1, 0]))
        # self.integrator.set('u', np.array([0, 10]))
        # status_s = self.integrator.solve()
        # x_current = self.integrator.get('x')
        # print("!!!!!!!!!!!x_current is:", x_current)
        
    def simulation(self, x0, xs):
        simX = np.zeros((self.N+1, self.nx))
        simU = np.zeros((self.N, self.nu))
        x_current = x0
        print("x0 is:", x_current)
        simX[0, :] = x0.reshape(1, -1)
        xs_between = np.concatenate((xs, np.zeros(4)))
        time_record = np.zeros(self.N)

        # closed loop
        self.solver.set(self.N, 'yref', xs)
        for i in range(self.N):
            self.solver.set(i, 'yref', xs_between)
        #! print the cost setting
        for i in range(self.N):
            # solve ocp
            start = timeit.default_timer()
            ##  set inertial (stage 0)
            self.solver.constraints_set(0, 'lbx', x_current)
            self.solver.constraints_set(0, 'ubx', x_current)
            
            
            status = self.solver.solve()

            if status != 0 :
                raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status))

            simU[i, :] = self.solver.get(0, 'u')
            # print("simU is:", simU[i, :])
            cost_value = self.solver.get_cost()
            print("Optimization cost:", cost_value)
            time_record[i] =  timeit.default_timer() - start
            
            
            # simulate system
            self.integrator.set('x', x_current)
            self.integrator.set('u', simU[i, :])

            status_s = self.integrator.solve()
            if status_s != 0:
                raise Exception('acados integrator returned status {}. Exiting.'.format(status))

            # update
            x_current = self.integrator.get('x')
            # print("X_current is:", i, x_current)
            
            simX[i+1, :] = x_current
        print(simX)
        print("average estimation time is {}".format(time_record.mean()))
        print("max estimation time is {}".format(time_record.max()))
        print("min estimation time is {}".format(time_record.min()))

opts1 = {"version" : "leftChange", "solver": "ipopt", "integrator":"LTI"}
N=30
dt=0.2
Acados_MPC = makeControllerAcados(vehicleADV,traffic,scenarioADV,N,opts1,dt)
Acados_MPC.simulation(x0=np.array([0, 1, 1, 0]), xs=np.array([100., 1., 10, 0]))
