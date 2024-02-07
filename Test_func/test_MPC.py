from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import sys
path_to_add='C:\\Users\\A490243\\Desktop\\Master_Thesis'
sys.path.append(path_to_add)
from Controller.MPC_tighten_bound import MPC_tighten_bound
from vehicleModel.vehicle_model import car_VehicleModel
from util.utils import *


# Assuming the necessary path additions and imports have been done correctly as you've shown

class MPC:
    def __init__(self, vehicle, Q, R, P0, process_noise, Possibility=0.99, N=12) -> None:
        # The number of MPC states, here include x, y, psi and v
        NUM_OF_STATES = 4
        self.nx = NUM_OF_STATES
        # The number of MPC actions, including acc and steer_angle
        NUM_OF_ACTS = 2
        self.nu = NUM_OF_ACTS
        
        self.vehicle = vehicle
        self.nx, self.nu, self.nrefx, self.nrefu = self.vehicle.getSystemDim()
        self.Q = Q
        self.R = R
        self.P0 = P0
        self.process_noise = process_noise
        self.Possibility = Possibility
        self.N = N
        self.Param = Param()
        
        # Create Opti Stack
        self.opti = Opti()
        
        # Initialize opti stack
        self.x = self.opti.variable(self.nx, self.N + 1)
        self.u = self.opti.variable(self.nu, self.N)
        self.refx = self.opti.parameter(self.nrefx, self.N + 1)
        self.refu = self.opti.parameter(self.nrefu, self.N)
        self.x0 = self.opti.parameter(self.nx, 1)
        
        # IDM leading vehicle position parameter
        self.p_leading = self.opti.parameter(1)
    
    def compute_Dlqr(self):
        return self.MPC_tighten_bound.calculate_Dlqr()
    
    def IDM_constraint(self, p_leading, v_eg, d_s=10, L1=4, T_s=1.5, lambda_s=0):
        """
        IDM constraint for tracking the vehicle in front.
        """
        return p_leading - L1 - d_s - T_s * v_eg - lambda_s
    
    def calc_linear_discrete_model(self, v=10, phi=0, delta=0):
        """
        Calculate linear and discrete time dynamic model.
        """
        A = DM([[1.0, 0.0, self.Param.dt * np.cos(phi), -self.Param.dt * v * np.sin(phi)],
                [0.0, 1.0, self.Param.dt * np.sin(phi), self.Param.dt * v * np.cos(phi)],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, self.Param.dt * np.tan(delta) / self.Param.WB, 1.0]])
        
        B = DM([[0.0, 0.0],
                [0.0, 0.0],
                [0.0, self.Param.dt],
                [self.Param.dt * v / (self.Param.WB * np.cos(delta) ** 2), 0.0]])
        
        
        C = DM([self.Param.dt * v * np.sin(phi) * phi,
                -self.Param.dt * v * np.cos(phi) * phi,
                0.0,
                -self.Param.dt * v * delta / (self.Param.WB * np.cos(delta) ** 2)])
        
        return A, B, C
    
    def setStateEqconstraints(self, v=10, phi=0, delta=0):
        """
        Set state equation constraints.
        """
        for i in range(self.N):
            A, B, C = self.calc_linear_discrete_model(v, phi, delta)
            self.opti.subject_to(self.x[:, i+1] == A @ self.x[:, i] + B @ self.u[:, i] + C)
        self.opti.subject_to(self.x[:, 0] == self.x0)
    
    def setInEqConstraints_val(self, H_up=None, upb=None, H_low=None, lwb=None):
        """
        Set inequality constraints values.
        """
        # Default or custom constraints
        self.H_up = H_up if H_up is not None else [np.array([[1], [0], [0], [0]]), np.array([[0], [1], [0], [0]]), np.array([[0], [0], [1], [0]]), np.array([[0], [0], [0], [1]])]
        self.upb = upb if upb is not None else np.array([[5000], [500], [30], [np.pi/8]])
        self.H_low = H_low if H_low is not None else [np.array([[-1], [0], [0], [0]]), np.array([[0], [-1], [0], [0]]), np.array([[0], [0], [-1], [0]]), np.array([[0], [0], [0], [-1]])]
        self.lwb = lwb if lwb is not None else np.array([[0], [500], [0], [np.pi/8]])
    
    def setInEqConstraints(self):
        """
        Set inequality constraints.
        """
        v, phi, delta = 10, 0, 0  # Default values; adjust as necessary
        A, B, _ = self.calc_linear_discrete_model(v, phi, delta)
        D = DM.eye(self.nx)  # Noise matrix
        self.MPC_tighten_bound = MPC_tighten_bound(A, B, D, self.Q, self.R, self.P0, self.process_noise, self.Possibility)
        
        self.setInEqConstraints_val()  # Set tightened bounds
        
        # Example tightened bound application (adjust according to actual implementation)
        # Here you would apply the IDM constraints and any other inequality constraints
    
    def setCost(self):
        """
        Set cost function for the optimization problem.
        """
        L, Lf = self.vehicle.getCost()
        self.opti.minimize(getTotalCost(L, Lf, self.x, self.u, self.refx, self.refu, self.N))
    
    def setController(self):
        """
        Set constraints and cost function for the MPC controller.
        """
        self.setStateEqconstraints()
        self.setInEqConstraints()
        self.setCost()
    
    def solve(self, x0, ref_trajectory, ref_control, p_leading):
        """
        Solve the MPC problem.
        """
        # Set the initial condition and reference trajectories
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.refx, ref_trajectory)
        self.opti.set_value(self.refu, ref_control)
        self.opti.set_value(self.p_leading, p_leading)
        
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
            return None, None
        
        
dt = 0.1
N=12
vehicleADV = car_VehicleModel(dt,N, width = 2, length = 4)
nx,nu,nrefx,nrefu = vehicleADV.getSystemDim()
int_opt = 'rk'
vehicleADV.integrator(int_opt,dt)
F_x_ADV  = vehicleADV.getIntegrator()
vx_init_ego = 10   
vehicleADV.setInit([0,0],vx_init_ego)
#__init__(self, vehicle, Q, R, P0, process_noise, Possibility=0.99, N=12)
Q_ADV = [0,40,3e2,5,5]                            # State cost, Entries in diagonal matrix
R_ADV = [5,5]                                    # Input cost, Entries in diagonal matrix
vehicleADV.cost(Q_ADV,R_ADV)
vehicleADV.costf(Q_ADV)
L_ADV,Lf_ADV = vehicleADV.getCost()
mpc_controller = MPC(vehicleADV, np.eye(nx), np.eye(nu), np.eye(nx), np.eye(nx), 0.99, N)
# Set initial conditions for the ego vehicle
x0 = np.array([[0], [0], [10], [0]])  # Initial state: [x, y, psi, v]. Example values provided

# Define reference trajectory and control for N steps
# For simplicity, setting reference states and inputs to zero or desired states.
# In practice, these should be calculated based on the desired trajectory.
ref_trajectory = np.zeros((nx, N + 1)) # Reference trajectory (states)
# ref x should be 100
ref_trajectory[0,:] = 100
ref_control = np.ones((nu, N))  # Reference control inputs

# Position of the leading vehicle (for IDM constraint)
# Assuming the leading vehicle is at 20 meters ahead initially
p_leading = 20

# Set the controller (this step initializes the optimization problem with cost and constraints)
mpc_controller.setController()

# Solve the MPC problem
u_opt, x_opt = mpc_controller.solve(x0, ref_trajectory, ref_control, p_leading)

# Print the optimized control input for the first step
print("Optimized control input (steer_angle,acc) for the first step:", u_opt[:, 0])
