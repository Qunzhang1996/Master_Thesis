'''
Here is the truck model, including the kinematic model (for nmpc and LTI_MPC)
In this file, we will rewrite the vehicle model from vehBicycleKinematic to car_VehicleModel
In this project, we will use the LTI_MPC to control the truck
'''
from casadi import*
import sys
path_to_add='C:\\Users\\A490243\\Desktop\\Master_Thesis'
sys.path.append(path_to_add)
from Autonomous_Truck_Sim.vehicleModelGarage import vehBicycleKinematic
from enum import IntEnum

# ████████╗██████╗ ██╗   ██╗ ██████╗██╗  ██╗        ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗     
# ╚══██╔══╝██╔══██╗██║   ██║██╔════╝██║ ██╔╝        ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║     
#    ██║   ██████╔╝██║   ██║██║     █████╔╝         ██╔████╔██║██║   ██║██║  ██║█████╗  ██║     
#    ██║   ██╔══██╗██║   ██║██║     ██╔═██╗         ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║     
#    ██║   ██║  ██║╚██████╔╝╚██████╗██║  ██╗███████╗██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗
#    ╚═╝   ╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝                                                                                            
class Kinematic(IntEnum):
    X_km, Y_km, V_km, THETA_km = range(4)

class car_VehicleModel(vehBicycleKinematic):
    """Here, inheriting from vehBicycleKinematic class ( 2 degree truck)

    Kinematic bicycle model with car
    x = [p_x p_y v_x theta]
    u = [steer_ang, acc_v]
    """
    def __init__(self,dt,N, width = 2.54, length = 8.4, scaling = [0.1,1,1,1,1]):
        self.name = "Truck_bicycle"
        self.laneCenters = 143.318146
        self.laneWidth = 3.5
        self.nx = 4                    # State dimensions
        self.nu = 2                     # Input dimensions
        self.np = self.nx
        self.nrefx = self.nx
        self.nrefu = self.nu
        self.dt = dt                    # Time step
        self.N = N
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
        
        # This should really change based on scenario
        if self.x_init[1] > 143.318146+self.laneWidth/2:
            self.lane = 1
        elif self.x_init[1] < 143.318146-self.laneWidth/2:
            self.lane = -1
        else:
            self.lane = 0
            
        # ! parameters for the car
        self.width = width
        self.ego_width = 2.54
        self.length = length
        self.L_tract = length                     # ! in the simulation, only have the tractor
        self.L_trail = self.length-self.L_tract
        self.WB = 3/4*self.length  
        
        # Energy efficiency parameters
        self.Cd = 0.31                  # []
        self.Area = 10                  # [m2]
        self.Air_rho = 1.225            # [kg/m3]
        self.mass = 31000               # [kg]
        self.C_roll = 0.005             # []
        self.r_whl = 0.056              # [m]

    
    
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
        
    def uConstraints(self):
        return [-3.14/8,-0.5*9.81],[3.14/8,0.5*9.81]
    
    #! rewrite the xconstraint from vehBicycleKinematic to car_VehicleModel
    def xConstraints(self):
        # State constraints based on internal dynamics
        # 10000 arbitrarilly large number
        inf = 50000
        lower = [0,-inf,0,-3.14/8]
        upper = [inf,inf,inf,3.14/8]
        return lower, upper
    
    #! this is the function to calculate the linear and discrete time dynamic model using casadi symbolic
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
    
    
    #! this is the function to calculate the linear and discrete time dynamic model
    def vehicle_linear_discrete_model(self, v=15, phi=0, delta=0):
        """
        Calculate linear and discrete time dynamic model.
        """
        A_d = DM([[1.0, 0.0, self.dt * np.cos(phi), -self.dt * v * np.sin(phi)],
                [0.0, 1.0, self.dt * np.sin(phi), self.dt * v * np.cos(phi)],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, self.dt * np.tan(delta) / self.WB, 1.0]])
        
        B_d = DM([[0.0, 0.0],
                [0.0, 0.0],
                [0.0, self.dt],
                [self.dt * v / (self.WB * np.cos(delta) ** 2), 0.0]])
        
        
        g_d = DM([self.dt * v * np.sin(phi) * phi,
                -self.dt * v * np.cos(phi) * phi,
                0.0,
                -self.dt * v * delta / (self.WB * np.cos(delta) ** 2)])
        
        return A_d, B_d, g_d
    
    
    def setReferences(self,vx,center_line=143.318146):
        self.laneCenters = [center_line,center_line+self.laneWidth,center_line-self.laneWidth]
        self.refxT[1] = self.laneCenters[0]
        self.refxL[1] = self.laneCenters[1]
        self.refxR[1] = self.laneCenters[2]

        self.refxT[2] = vx
        self.refxL[2] = vx
        self.refxR[2] = vx
        return self.refxT, self.refxL, self.refxR
    


    def cost(self,Q,R):
        self.Q = Q
        self.R = R
        l = 0
        for i in range(0,self.nx):
                l += Q[i]*(self.x[i]-self.refx[i]) ** 2
        
        for i in range(0,self.nu):
            l += R[i]*(self.u[i]-self.refu[i]) ** 2
        self.L = Function('L',[self.x,self.u,self.refx,self.refu],[l],['x','u','refx','refu'],['Loss'])

    def costf(self,Q):
        lf = 0
        for i in range(0,self.nx):
            lf += Q[i]*(self.x[i]-self.refx[i]) ** 2
        
        self.Lf =  Function('Lf',[self.x,self.refx],[lf],['x','refx'],['Lossf'])    
    
    def get_vehicle_size(self):
        return self.L_tract, self.L_trail, self.ego_width
    
    def get_dt(self):
        return self.dt
    
    
    def getSize(self):
        return self.width, self.length, self.L_tract, self.L_trail
    
    def getSystemDim(self):
        return self.nx,self.nu,self.nrefx,self.nrefu
    
    def getCostParam(self):
        return self.Q, self.R

    def getPosition(self):
        return self.p
    
    def setRoad(self,roadMin,roadMax,laneCenters):
        self.roadMin = roadMin
        self.roadMax = roadMax
        self.laneCenters = laneCenters
        return self.roadMin, self.roadMax, self.laneCenters
    
    
        
    def setInit(self,px,vx):
        self.x_init[0] = px[0]
        self.x_init[1] = px[1]
        self.x_init[2] = vx

        self.u_init = [0,0]

        if self.x_init[1] > self.laneWidth:
            self.lane = 1
        elif self.x_init[1] < 0:
            self.lane = -1
        else:
            self.lane = 0
            
        return self.x_init, self.u_init
    
    

