""" Here is the LTI_Mpc for the Autonomous Vehicle to track the vehicle in front in the straight road. 
"""
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import sys
sys.path.append(r'/mnt/c/Users/A490242/acados/Master_Thesis-main')
from Autonomous_Truck_Sim.helpers import *
from Controller.MPC_tighten_bound import MPC_tighten_bound
from vehicleModel.vehicle_model import car_VehicleModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel
from util.utils import *
import os
import sys
import shutil
import errno
import timeit

class MPC:
    def __init__(self, vehicle, Q, R, P0, process_noise, Possibility=0.95, N=12) -> None:
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
        # ref val for the vehicle matrix
        self.v_ref = 15
        self.phi_ref = 0
        self.delta_ref = 0
        
        # ! create ocp model
        self.ocp = AcadosOcp()    
        self.ocp.model.name = 'vehicle_running_acados'
        self.ocp.model.x = self.vehicle.x
        self.ocp.model.xdot = SX.sym('x_dot', self.nx)
        self.ocp.model.u = self.vehicle.u

        self.ocp.dims.N = self.N # shooting node
        self.ocp.solver_options.tf = self.N # prediction horizon
        self.ny = self.nx + self.nu

        
            
        # ! set the LTI model from the vehicle model
        self.A, self.B, self.C = self.vehicle.vehicle_linear_discrete_model(v=15, phi=0, delta=0)
        self.x_next = self.A@self.vehicle.x +  self.B@self.vehicle.u
        self.ocp.model.disc_dyn_expr = self.x_next
        self.ocp.solver_options.integrator_type = 'DISCRETE'
            # self.MPC_tighten_bound = MPC_tighten_bound(self.A, self.B, self.D, np.diag(self.Q), np.diag(self.R), self.P0, self.process_noise, self.Possibility)
            
        #! Create OCP solver 
        # solver options
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'   
        self.ocp.solver_options.print_level = 0
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'

       

        self.constraint = types.SimpleNamespace()
     
    def setInEqConstraints_val(self, H_up=None, upb=None, H_low=None, lwb=None):
        """
        Set inequality constraints values.
        """
        # Default or custom constraints
        self.H_up = H_up if H_up is not None else [np.array([[1], [0], [0], [0]]), np.array([[0], [1], [0], [0]]), np.array([[0], [0], [1], [0]]), np.array([[0], [0], [0], [1]])]
        self.upb = upb if upb is not None else np.array([[5000], [5000], [30], [3.14/8]])
        
        self.H_low = H_low if H_low is not None else [np.array([[-1], [0], [0], [0]]), np.array([[0], [-1], [0], [0]]), np.array([[0], [0], [-1], [0]]), np.array([[0], [0], [0], [-1]])]
        self.lwb = lwb if lwb is not None else np.array([[5000], [5000], [0], [3.14/8]])
    
    def setInEqConstraints_acados(self,p_leading, leading_velocity=10, vel_diff=6, lambda_s=0):
        """
        Set inequality constraints.
        """
        v, phi, delta = self.v_ref, self.phi_ref, self.delta_ref  # Default values; adjust as necessary
        
        self.setInEqConstraints_val() 
        self.constraint.lbx = -self.lwb
        self.constraint.ubx = self.upb
        

        # set the constraints for the input  [-3.14/8,-0.7*9.81],[3.14/8,0.05*9.81]
        self.constraint.delta_min = -3.14 / 8
        self.constraint.delta_max = 3.14 / 8
        self.constraint.acc_min = -0.5 * 9.81
        self.constraint.acc_max = 0.5 * 9.81

    
    def solve(self, x0, ref_trajectory, ref_control, p_leading, leading_velocity=10, vel_diff=6):
        """
        Solve the MPC problem.
        """

        #! set the constraints
        self.setInEqConstraints_acados(p_leading, leading_velocity=10, vel_diff=6, lambda_s=0)
        # control constraints
        self.ocp.constraints.lbu = np.array([self.constraint.delta_min, self.constraint.acc_min])
        self.ocp.constraints.ubu = np.array([self.constraint.delta_max, self.constraint.acc_max])
        self.ocp.constraints.idxbu = np.array([0, 1])
        # state constraints
        self.ocp.constraints.lbx = self.constraint.lbx
        self.ocp.constraints.ubx = self.constraint.ubx
        self.ocp.constraints.idxbx = np.array([0, 1, 2, 3])
        # initial state
        # self.ocp.constraints.x0 = np.array(x0)
        self.ocp.constraints.x0 = np.array([36.16552004667315, 143.3179729227714,0.29016719521143663, -0.011114551359435901])

        #! set the cost
        self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.cost_type_e = 'LINEAR_LS'
        self.ocp.cost.W = scipy.linalg.block_diag(self.Q, self.R)
        self.ocp.cost.W_e = self.Q
        self.ocp.cost.Vx = np.zeros((self.ny, self.nx))
        self.ocp.cost.Vx[:self.nx, :self.nx] = np.eye(self.nx)
        self.ocp.cost.Vu = np.zeros((self.ny, self.nu))
        self.ocp.cost.Vu[-self.nu:, -self.nu:] = np.eye(self.nu)
        self.ocp.cost.Vx_e = np.eye(self.nx)
        self.ocp.cost.yref = np.array([36.16552004667315, 143.3179729227714,0.29016719521143663, -0.011114551359435901, 0, 0])
        self.ocp.cost.yref_e = np.array([36.16552004667315, 143.3179729227714,0.29016719521143663, -0.011114551359435901])
        json_file = os.path.join('./'+self.ocp.model.name+'_acados_ocp.json')
        self.solver = AcadosOcpSolver(self.ocp, json_file)

       
        

        self.simX = np.zeros((self.nx, self.N+1))
        self.simU = np.zeros((self.nu, self.N))
        for i in range(self.N):
            self.solver.cost_set(i, 'yref', np.concatenate((ref_trajectory[:,i], ref_control[:,i])) )
            
        for i in range(self.N):
            #  set inetial (stage 0)
            x0 = np.array([36.16552004667315, 143.3179729227714,0.29016719521143663, -0.011114551359435901])
            self.solver.constraints_set(0, 'lbx', x0)
            self.solver.constraints_set(0, 'ubx', x0)
            try:
                status = self.solver.solve()
                self.simU[:,i] = self.solver.get(0, 'u')
            except Exception as e:
                print(f"An error occurred: {e}")
                break

            if status != 0 :
                raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status, i))
            

        return self.simU
        
        
    def get_dynammic_model(self):
        """
        Return the dynamic model of the vehicle.
        """
        return self.A, self.B, self.C








