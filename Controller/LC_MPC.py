""" Here is the LTI_Mpc for the Autonomous egoVehicle to track the egoVehicle in front in the straight road. 
"""
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import sys
path_to_add='C:\\Users\\A490242\\Desktop\\Master_Thesis'
sys.path.append(path_to_add)
from Autonomous_Truck_Sim.helpers import *
from Controller.MPC_tighten_bound import MPC_tighten_bound
from vehicleModel.vehicle_model import car_VehicleModel
from util.utils import Param
class LC_MPC:
    """
    ██████╗  ██████╗ ██████╗      ██████╗ ██╗     ███████╗███████╗███████╗    ███╗   ███╗██████╗  ██████╗            
    ██╔════╝ ██╔═══██╗██╔══██╗    ██╔══██╗██║     ██╔════╝██╔════╝██╔════╝    ████╗ ████║██╔══██╗██╔════╝            
    ██║  ███╗██║   ██║██║  ██║    ██████╔╝██║     █████╗  ███████╗███████╗    ██╔████╔██║██████╔╝██║                 
    ██║   ██║██║   ██║██║  ██║    ██╔══██╗██║     ██╔══╝  ╚════██║╚════██║    ██║╚██╔╝██║██╔═══╝ ██║                 
    ╚██████╔╝╚██████╔╝██████╔╝    ██████╔╝███████╗███████╗███████║███████║    ██║ ╚═╝ ██║██║     ╚██████╗        
     ╚═════╝  ╚═════╝ ╚═════╝     ╚═════╝ ╚══════╝╚══════╝╚══════╝╚══════╝    ╚═╝     ╚═╝╚═╝      ╚═════╝                                                                                                                                                                                                            
    """

    def __init__(self, egoVehicle, trafficVehicles, Q, R, P0, process_noise, Possibility=0.95, N=12, version='trailing', scenario=None) -> None:
        # The number of MPC states, here include x, y, psi and v
        NUM_OF_STATES = 4
        self.nx = NUM_OF_STATES
        # The number of MPC actions, including acc and steer_angle
        NUM_OF_ACTS = 2
        self.nu = NUM_OF_ACTS
        self.egoVehicle = egoVehicle
        self.trafficVehicles = trafficVehicles
        self.nx, self.nu, self.nrefx, self.nrefu = self.egoVehicle.getSystemDim()
        self.Nveh = len(trafficVehicles)
        # self.Nveh = trafficVehicles.getDim()
        self.Q = Q
        self.R = R
        self.P0 = P0
        self.process_noise = process_noise
        self.Possibility = Possibility
        self.N = N
        self.laneWidth = 3.5
        self.Param = Param()
        # ref val for the egoVehicle matrix
        self.v_ref = 15
        self.phi_ref = 0
        self.delta_ref = 0
        
        # Create Opti Stack
        self.opti = Opti()
        # Initialize opti stack
        self.x = self.opti.variable(self.nx, self.N + 1)
        self.u = self.opti.variable(self.nu, self.N)
        self.refx = self.opti.parameter(self.nrefx, self.N + 1)
        self.refu = self.opti.parameter(self.nrefu, self.N)
        self.x0 = self.opti.parameter(self.nx, 1)
        # slack variable for the IDM constraint
        self.lambda_s= self.opti.variable(1,self.N + 1)
        # slack variable for y 
        self.slack_y = self.opti.variable(1,self.N + 1)
        # IDM leading egoVehicle position parameter
        self.lead = self.opti.parameter(self.Nveh,self.N+1)
        self.p_leading = self.opti.parameter(1)
        # set car velocity 
        self.leading_velocity = self.opti.parameter(1)
        self.vel_diff = self.opti.parameter(1)    
        # set the controller options
        self.version = version
        self.scenario = scenario
        if version['version'] == 'leftChange' or version['version'] == 'rightChange':
            self.lead = self.opti.parameter(self.Nveh,self.N+1)
            self.traffic_x = self.opti.parameter(self.Nveh,self.N+1)
            self.traffic_y = self.opti.parameter(self.Nveh,self.N+1)
            self.traffic_sign = self.opti.parameter(self.Nveh,self.N+1)
            self.traffic_shift = self.opti.parameter(self.Nveh,self.N+1)
            self.traffic_flip = self.opti.parameter(self.Nveh,self.N+1)
            
    
    def compute_Dlqr(self):
        return self.MPC_tighten_bound.calculate_Dlqr()
    
    def IDM_constraint(self, p_leading, v_eg, d_s=1, L1=6, T_s=1.0, lambda_s=0):
        """
        IDM constraint for tracking the egoVehicle in front.
        """
        return p_leading - L1 - d_s - T_s * v_eg + lambda_s

    
    def calc_linear_discrete_model(self, v, phi=0, delta=0):
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
    
    def setStateEqconstraints(self, v=15, phi=0, delta=0):
        """
        Set state equation constraints.
        """
        for i in range(self.N):
            A, B, C = self.calc_linear_discrete_model(v, phi, delta)  
            self.A, self.B, self.C = A, B, C
            self.opti.subject_to(self.x[:, i+1] == A @ self.x[:, i] + B @ self.u[:, i] + C)
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
        
        # set slack variable for the y
        # print("the upb is: ",self.upb[1,0])
        # print("the slack_y is: ",self.slack_y)
        # self.upb[1,0] = 143.318146+1.75 
        # self.lwb[1,0] = -143.318146+1.75 

        # self.upb[1,0] += self.slack_y
        # self.lwb[1,0] += self.slack_y
    
    def setInEqConstraints(self):
        """
        Set inequality constraints.
        """
        v, phi, delta = self.v_ref, self.phi_ref, self.delta_ref  # Default values; adjust as necessary
        A, B, _ = self.calc_linear_discrete_model(v, phi, delta)
        D = np.eye(self.nx)  # Noise matrix
        
        self.setInEqConstraints_val()  # Set tightened bounds

        self.MPC_tighten_bound = MPC_tighten_bound(A, B, D, self.Q, self.R, self.P0, self.process_noise, self.Possibility)
        # Set the IDM constraint
        self.IDM_constraint_list = []
        for i in range(self.N+1):
            self.IDM_constraint_list.append(self.IDM_constraint(self.p_leading + self.leading_velocity*self.Param.dt*i, 
                                                                self.x[2, i],self.lambda_s[0,i]))
            
        
        # Set the y constraint 13
        self.y_upper = [143.318146+0.75] * (self.N+1)
        self.y_lower = [-143.318146+0.75] * (self.N+1)
        for i in range(self.N+1):
            self.y_upper[i] += self.slack_y[0,i]
            self.y_lower[i] += self.slack_y[0,i]
        
        # Set the vel_diff constraint 
        vel_diff_constrain_list = [self.vel_diff] * (self.N+1)

        # Example tightened bound application (adjust according to actual implementation)
        self.tightened_bound_N_list_up = self.MPC_tighten_bound.tighten_bound_N(self.P0, self.H_up, self.upb, self.N, 1)
        self.tightened_bound_N_list_lw = self.MPC_tighten_bound.tighten_bound_N(self.P0, self.H_low, self.lwb, self.N, 0)
        self.tightened_bound_N_IDM_list = self.MPC_tighten_bound.tighten_bound_N_IDM(self.IDM_constraint_list, self.N)
        self.tightened_bound_N_vel_diff_list = self.MPC_tighten_bound.tighten_bound_N_vel_diff(vel_diff_constrain_list, self.N)
        self.tightened_bound_N_y_upper_list = self.MPC_tighten_bound.tighten_bound_N_y_upper(self.y_upper, self.N)
        self.tightened_bound_N_y_lower_list = self.MPC_tighten_bound.tighten_bound_N_y_lower(self.y_lower, self.N)

        # the tightened bound (up/lw) is N+1 X NUM_OF_STATES  [x, y, v, psi] 
        # according to the new bounded constraints set the constraints
        for i in range(self.N+1):
            self.opti.subject_to(self.x[:, i] <= DM(self.tightened_bound_N_list_up[i].reshape(-1, 1)))
            self.opti.subject_to(self.x[:, i] >= DM(self.tightened_bound_N_list_lw[i].reshape(-1, 1)))
            # Set the IDM constraint
            self.opti.subject_to(self.x[0, i] <= self.tightened_bound_N_IDM_list[i].item())
            
            # Set the vel_diff constraint
            self.opti.subject_to(self.x[2, i] - self.leading_velocity <= self.tightened_bound_N_vel_diff_list[i].item())
            self.opti.subject_to(self.x[2, i] - self.leading_velocity >= -self.tightened_bound_N_vel_diff_list[i].item())
            # set the y constraint
            self.opti.subject_to(self.x[1, i] <= self.tightened_bound_N_y_upper_list[i].item())
            self.opti.subject_to(self.x[1, i] >= self.tightened_bound_N_y_lower_list[i].item())
            
        # set the constraints for the input  [-3.14/180,-0.7*9.81],[3.14/180,0.05*9.81]
        self.opti.subject_to(self.u[0, :] >= -3.14 / 180)
        self.opti.subject_to(self.u[0, :] <= 3.14 / 180)
        self.opti.subject_to(self.u[1, :] >= -0.5 * 9.81)
        self.opti.subject_to(self.u[1, :] <= 0.5 * 9.81)
        # tighten the change of the input
        # for i in range(self.N-1):
        #     self.opti.subject_to(self.u[1,i+1]-self.u[1,i] <= 0.5)
        #     self.opti.subject_to(self.u[1,i+1]-self.u[1,i] >= -0.5)
          
    
    def setTrafficConstraints(self):
        """
        Set inequality constraints for lane change scenario.
        """
        self.S = self.scenario.constraint(self.trafficVehicles)
        if self.scenario.name == 'simpleOvertake':
            for i in range(self.Nveh):
                self.opti.subject_to(self.traffic_flip[i,:] * self.x[1,:] 
                                    >= self.traffic_flip[i,:] * self.S[i](self.x[0,:], self.traffic_x[i,:], self.traffic_y[i,:],
                                        self.traffic_sign[i,:], self.traffic_shift[i,:]))


    def setCost(self):
        """
        Set cost function for the optimization problem.
        """
        L, Lf = self.egoVehicle.getCost()
        cost=getTotalCost(L, Lf, self.x, self.u, self.refx, self.refu, self.N)
        # Add slack variable cost
        cost += 3e4*self.lambda_s@ self.lambda_s.T
        # Add slack variable cost for y
        cost += 3e4*self.slack_y@ self.slack_y.T
        self.opti.minimize(cost)
    


    def setController(self):
            """
            Set constraints and cost function for the MPC controller.
            """
            self.setStateEqconstraints()
            self.setInEqConstraints()
            # if self.version == 'leftChange' or self.version == 'rightChange':
            if self.version['version'] == 'leftChange' or self.version['version'] == 'rightChange':
                self.setTrafficConstraints()
            self.setCost()

    def set_lane_change_parameters(self, traffic_state): # still working on this
        """
        Sets traffic parameters, to be used in the MPC controllers
        """

        sign = np.zeros((self.Nveh,self.N+1))
        shift = np.zeros((self.Nveh,self.N+1))
        flip = np.ones((self.Nveh,self.N+1))

        if self.version['version'] == 'leftChange':
            for ii in range(self.Nveh):
                for jj in range(self.N+1):
                    if traffic_state[1,jj,ii] > self.laneWidth + 143.318146: # doubt here
                        sign[ii,jj] = -1
                        shift[ii,jj] = 2 * self.laneWidth
                        flip[ii,jj] = -1
                    elif traffic_state[1,jj,ii] < 143.318146: 

                        sign[ii,jj] = 1
                        shift[ii,jj] = -self.laneWidth
                    else:                         
                        sign[ii,jj] = 1
                        shift[ii,jj] = 0

        elif self.version['version'] == 'rightChange':
            for ii in range(self.Nveh):
                for jj in range(self.N+1):
                    if traffic_state[1,jj,ii] > self.laneWidth:
                        sign[ii,jj] = -1
                        shift[ii,jj] = 2 * self.laneWidth
                        flip[ii,jj] = -1
                    elif traffic_state[1,jj,ii] < 0:
                        sign[ii,jj] = 1
                        shift[ii,jj] = -self.laneWidth
                    else:
                        sign[ii,jj] = -1
                        shift[ii,jj] = self.laneWidth
                        flip[ii,jj] = -1

        traffic_state[2,:,:] = sign.T
        traffic_state[3,:,:] = shift.T
        traffic_state[4,:,:] = flip.T
        return sign, shift, flip, traffic_state

    
    
    def solve(self, x0, ref_trajectory, ref_control, p_leading, traffic_state, leading_velocity=10, vel_diff=5):
        """
        Solve the MPC problem.
        """
        # Set the initial condition and reference trajectories
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.refx, ref_trajectory)
        self.opti.set_value(self.refu, ref_control)
        self.opti.set_value(self.p_leading, p_leading)
        self.opti.set_value(self.leading_velocity, leading_velocity)
        self.opti.set_value(self.vel_diff, vel_diff)

        # set the initial trajectories for the lane and right change scenarios
        if self.version['version'] == 'leftChange' or self.version['version'] == 'rightChange':
            self.opti.set_value(self.x0, x0)
            self.opti.set_value(self.refx, ref_trajectory)
            self.opti.set_value(self.refu, ref_control)
            traffic_sign, traffic_shift, traffic_flip, traffic_state = self.set_lane_change_parameters(traffic_state)
            self.opti.set_value(self.traffic_x,traffic_state[0,:,:].T)
            self.opti.set_value(self.traffic_y,traffic_state[1,:,:].T)
            self.opti.set_value(self.traffic_sign,traffic_sign)
            self.opti.set_value(self.traffic_shift,traffic_shift)
            self.opti.set_value(self.traffic_flip,traffic_flip)

        # Solver options
        opts = {"ipopt": {"print_level": 0, "tol": 1e-8}, "print_time": 0}
        self.opti.solver("ipopt", opts)
        
        try:
            sol = self.opti.solve()
            u_opt = sol.value(self.u)
            x_opt = sol.value(self.x)
            lambda_s = sol.value(self.lambda_s)
            # print(f"slack_y: {sol.value(self.slack_y)}")
            # also return tightened IDM constraint with solved op
            tightened_IDM_constraints = [sol.value(constraint) for constraint in self.IDM_constraint_list]
            return u_opt, x_opt, lambda_s, tightened_IDM_constraints
        except Exception as e:
            print(f"An error occurred: {e}")
            self.opti.debug.value(self.x)
            return None, None, None, None
        
    def getFunction(self):
        # if self.opts["version"] == "trailing":
        #     return self.opti.to_function('MPC',[self.x0,self.refx,self.refu,self.lead],[self.x,self.u],
        #             ['x0','refx','refu','lead'],['x_opt','u_opt'])
        # else:
            return self.opti.to_function('MPC',[self.x0,self.refx,self.refu,
                    self.traffic_x,self.traffic_y,self.traffic_sign,self.traffic_shift,self.traffic_flip],[self.x,self.u],
                    ['x0','refx','refu','t_x','t_y','t_sign','t_shift','t_flip'],['x_opt','u_opt'])
        
        
    def get_dynammic_model(self):
        """
        Return the dynamic model of the egoVehicle.
        """
        return self.A, self.B, self.C








