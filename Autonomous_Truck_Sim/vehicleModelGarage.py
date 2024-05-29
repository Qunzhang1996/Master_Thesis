# Vehicle model setup for casadi interface
from casadi import*

class vehBicycleKinematic:
    '''
    Kinematic bicycle model with trailer
    x = [p_x p_y v_x theta_1 theta_2]
    u = [steer_ang, acc_v,x]
    '''
    def __init__(self,dt,N, width = 2.54, length = 16.1544, scaling = [0.1,1,1,1,1]):
        # Initialize parameters
        self.name = "truck_trailer_bicycle"
        self.nx = 5                     # State dimensions
        self.nu = 2                     # Input dimensions
        self.np = self.nx
        self.nrefx = self.nx
        self.nrefu = self.nu

        # self.dt = dt                    # Time step
        self.N = N

        self.width = width
        self.length = length
        self.L_tract = length/3                     # Tractor is a third of the trailer length
        self.L_trail = self.length-self.L_tract

         # Standard choices for reference and initialization
        self.x_init = [0,0,45/3.6,0,0]
        self.p = self.x_init[:2]
        self.v = self.x_init[2]

        self.scaling = scaling

        self.refxT = [0,0,60/3.6,0,0]
        self.refxL = [0,0,60/3.6,0,0]
        self.refxR = [0,0,60/3.6,0,0]

        # System model variables
        self.x = SX.sym('x',self.nx)             # self.x = [p_x p_y v_x v_y]
        self.u = SX.sym('u',self.nu)             # self.u = [a_x a_y]

        self.refx = SX.sym('refx',self.nx)
        self.refu = SX.sym('refu',self.nu)

        # This should really change based on scenario
        if self.x_init[1] > 6.5:
            self.lane = 1
        elif self.x_init[1] < 0:
            self.lane = -1
        else:
            self.lane = 0

        # Energy efficiency parameters
        self.Cd = 0.31                  # []
        self.Area = 10                  # [m2]
        self.Air_rho = 1.225            # [kg/m3]
        self.mass = 31000               # [kg]
        self.C_roll = 0.005             # []
        self.r_whl = 0.056              # [m]

    def model(self):
        # System dynamics model
        # x = [x_a y_a v_vx theta_1 theta_2]
        # u = [steer acc]
        dp_xb = self.x[2] 
        dp_yb = self.x[2] * tan(self.x[3])
        dv_x = self.u[1] * cos(self.x[3])
        dtheta_1 = self.x[2] / self.L_tract * tan(self.u[0]) / cos(self.x[3])
        dtheta_2 = self.x[2] / self.L_trail* sin(self.x[3] - self.x[4]) / cos(self.x[3])
        dx = vertcat(dp_xb,dp_yb,dv_x,dtheta_1,dtheta_2)
        
        return {'x':self.x,'p':self.u,'ode':dx}

    def integrator(self,opts,dt):
        self.dt = dt
        ode = self.model()
        if opts == 'rk':
            int_opts = {'tf':self.dt,'simplify':True,'number_of_finite_elements': 4}
        else:
            int_opts = {}

        int = integrator('int',opts,ode,int_opts)

        x_res = int(x0 = self.x,p=self.u)
        x_next = x_res['xf']

        self.F_x = Function('F_x',[self.x,self.u],[x_next],['x','u'],['x_next'])

    def uConstraints(self):
        return [-3.14/180,-0.7*9.81],[3.14/180,0.05*9.81]

    def xConstraints(self):
        # State constraints based on internal dynamics
        # 10000 arbitrarilly large number
        inf = 50000
        lower = [0,-inf,0,-3.14/8,-3.14/8]
        upper = [inf,inf,inf,3.14/8,3.14/8]
        return lower, upper

    def cost(self,Q,R):
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
    
    def getState(self):
        return self.state.full()

    def getControl(self):
        return self.control

    def getSystemDim(self):
        return self.nx,self.nu,self.nrefx,self.nrefu

    def getSize(self):
        return self.width, self.length, self.L_tract, self.L_trail

    def getSizeLorry(self):
        return self.L_tract, self.L_trail

    def getIntegrator(self):
        return self.F_x

    def getCost(self):
        return self.L, self.Lf
    
    def update(self,x_new,u_new):
        self.state = x_new
        self.control = u_new
        self.p = x_new[:2]

    def getPosition(self):
        return self.p

    def setInit(self,px,vx):
        self.x_init[0] = px[0]
        self.x_init[1] = px[1]
        self.x_init[2] = vx

        self.u_init = [0,0]

    def getInit(self):
        return self.x_init,self.u_init

    def setReferences(self,laneCenters):
        self.laneCenters = laneCenters
        self.refxT[1] = self.laneCenters[0]
        self.refxL[1] = self.laneCenters[1]
        self.refxR[1] = self.laneCenters[2]

        return self.refxT, self.refxL, self.refxR
    
    def getReferences(self):
        return self.refxT,self.refxL,self.refxR

    def getScaling(self):
        return self.scaling

    def getLane(self):
        if self.state[1] > 6.5:
            self.lane = 1
        elif self.state[1] < 0:
            self.lane = -1
        else:
            self.lane = 0

        return self.lane

    def getEconsParams(self):
        return self.Cd, self.Area, self.Air_rho, self.mass, self.C_roll, self.r_whl