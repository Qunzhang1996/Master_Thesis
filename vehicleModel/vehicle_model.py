# Vehicle model setup for casadi interface
#! include bicycle model for car and truck
from casadi import*
import sys
path_to_add='C:\\Users\\A490243\\Desktop\\Master_Thesis'
sys.path.append(path_to_add)
from Autonomous_Truck_Sim.vehicleModelGarage import vehBicycleKinematic
from enum import IntEnum


class Kinematic(IntEnum):
    X_km, Y_km, V_km, THETA_km = range(4)

class car_VehicleModel(vehBicycleKinematic):
    """Here, inheriting from vehBicycleKinematic class ( 2 degree truck)

    Kinematic bicycle model with car
    x = [p_x p_y v_x theta]
    u = [steer_ang, acc_v]
    """
    def __init__(self, *args, **kwargs):
        #__init__(self,dt,N, width = 2.54, length = 16.1544, scaling = [0.1,1,1,1,1])
        super().__init__(*args, **kwargs)
        self.name = "car_bicycle"
        self.nx = 4                     # State dimensions
        #! rewrite the refx
        self.nrefx = self.nx
        self.nrefu = self.nu
        self.refxT = [0,0,60/3.6,0]
        self.refxL = [0,0,60/3.6,0]
        self.refxR = [0,0,60/3.6,0]
        
         # System model variables
        self.x = SX.sym('x',self.nx)             # self.x = [p_x p_y v_x v_y]
        self.u = SX.sym('u',self.nu)             # self.u = [a_x a_y]

        self.refx = SX.sym('refx',self.nx)
        self.refu = SX.sym('refu',self.nu)

         # Standard choices for reference and initialization
        self.x_init = [0,0,45/3.6,0]
        self.p = self.x_init[:2]
        self.v = self.x_init[2]


        self.refxT = [0,0,60/3.6,0]
        self.refxL = [0,0,60/3.6,0]
        self.refxR = [0,0,60/3.6,0]

        # System model variables
        self.x = SX.sym('x',self.nx)             # self.x = [p_x p_y v_x v_y]
        self.u = SX.sym('u',self.nu)             # self.u = [a_x a_y]

        self.refx = SX.sym('refx',self.nx)
        self.refu = SX.sym('refu',self.nu)

        
    #! rewrite the model from vehBicycleKinematic to car_VehicleModel
    def model(self):
        # System dynamics model
        # x = [x_a y_a v_vx theta]
        # u = [steer acc]
        dp_xb = self.x[2] *cos(self.x[3])
        dp_yb = self.x[2] * sin(self.x[3])
        dv_x = self.u[1] 
        dtheta = self.x[2] / self.length* tan(self.u[0]) 
        dx = vertcat(dp_xb,dp_yb,dv_x,dtheta)
        self.dx = dx
        return {'x':self.x,'p':self.u,'ode':dx}
    
    
    def integrator(self,opts,dt):
        self.dt = dt
        ode = self.model()
        if opts == 'rk':
            int_opts = {'simplify':True,'number_of_finite_elements': 4}
        else:
            int_opts = {}
        t0 = 0  
        tf = self.dt
        int = integrator('int', opts, ode, t0, tf, int_opts)

        x_res = int(x0 = self.x,p=self.u)
        x_next = x_res['xf']

        self.F_x = Function('F_x',[self.x,self.u],[x_next],['x','u'],['x_next'])
    
    #! rewrite the xconstraint from vehBicycleKinematic to car_VehicleModel
    def xConstraints(self):
        # State constraints based on internal dynamics
        # 10000 arbitrarilly large number
        inf = 50000
        lower = [0,-inf,0,-3.14/8]
        upper = [inf,inf,inf,3.14/8]
        return lower, upper
    
    #! add the calculate the A and B matrix
    def calculate_AB(self, dt_sim=0.2,init_flag=False):
        """Calculate the A and B matrix for the linearized system

        Args:
            dt_sim:Defaults to 0.2.

        Returns:
            newA: linearized A matrix
            newB: linearized B matrix
        """
        #! rewrite the calculate_AB from vehBicycleKinematic to car_VehicleModel
        # # Define state and control symbolic variables
        # Get the state dynamics from the kinematic car model
        state_dynamics = self.dx
        # Calculate the Jacobians for linearization
        A = jacobian(state_dynamics, self.x)
        B = jacobian(state_dynamics, self.u)
        # Create CasADi functions for the linearized matrices
        f_A = Function('A', [self.x, self.u], [A])
        f_B = Function('B', [self.x, self.u], [B])
        #set the operating point
        if init_flag:
            x_op=self.x_init
            u_op=[0,0]
        else:
            x_op=self.state
            u_op=self.control
        # Evaluate the Jacobians at the operating point
        A_op = f_A(x_op, u_op)
        B_op = f_B(x_op, u_op)
        g_op= self.F_x(x_op,u_op)-A_op@x_op-B_op@u_op

        # Discretize the linearized matrices
        newA = A_op * dt_sim + np.eye(self.nx)
        newB = B_op * dt_sim
        newG=  g_op * dt_sim

        return newA, newB,newG

# # System initialization 
# dt = 0.1
# N=10
# end_time = 15
# t_axis = np.arange(0, end_time, dt)
# car_model = car_VehicleModel(dt,N, width = 2, length = 4)
# nx,nu,nrefx,nrefu = car_model.getSystemDim()
# int_opt = 'rk'
# car_model.integrator(int_opt,dt)
# F_x_ADV  = car_model.getIntegrator()
# vx_init_ego = 10   
# car_model.setInit([124-75,143.318146],vx_init_ego)
# x_iter = DM(int(nx),1)
# # get initial state and input
# x_iter[:],u_iter = car_model.getInit()

# # ----------------- Ego Vehicle obsver(kalman filter) Settings ------------------------
# init_flag = True
# F,B,G=car_model.calculate_AB(dt,init_flag=1)
# print("F is:",F,"","B is:",B,"G is:",G)