from casadi import *
import carla
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
sys.path.append(r'C:\Users\A490243\CARLA\CARLA_Latest\WindowsNoEditor\PythonAPI\carla')
sys.path.append(r'C:\Users\A490243\Desktop\Master_Thesis')
from Controller.LTI_MPC import MPC
from vehicleModel.vehicle_model import car_VehicleModel
from agents.navigation.controller import VehiclePIDController
from util.utils import *

# # ------------------------change map to Town06------------------------
# import subprocess
# # Command to run your script
# command = (
#     r'cd C:\Users\A490243\CARLA\CARLA_Latest\WindowsNoEditor\PythonAPI\util && '
#     r'python config.py --map Town06')
# subprocess.run(command, shell=True)
# # Run the command
car,truck = setup_carla_environment(Sameline_ACC=True)
time.sleep(1)
velocity1 = carla.Vector3D(10, 0, 0)
velocity2 = carla.Vector3D(10, 0, 0)
car.set_target_velocity(velocity1)
truck.set_target_velocity(velocity2)


# To start a basic agent
client = carla.Client('localhost', 2000)
world = client.get_world()
carla_map = world.get_map()
# initial the carla built in pid controller
car_contoller = VehiclePIDController(car, args_lateral = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': 0.01}, args_longitudinal = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': 0.01})
local_controller = VehiclePIDController(truck, args_lateral = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.0, 'dt': 0.01}, args_longitudinal = {'K_P': 1.95, 'K_I': 0.5, 'K_D': 0.2, 'dt': 0.01})
# To start a behavior agent with an aggressive car for truck to track
spawn_points = carla_map.get_spawn_points()
destination = carla.Location(x=1000, y=143.318146, z=0.3)
print(f"destination: {destination}")


#------------------initialize the robust MPC---------------------
dt = 0.1
N=12
vehicleADV = car_VehicleModel(dt,N, width = 2, length = 4)
nx,nu,nrefx,nrefu = vehicleADV.getSystemDim()
int_opt = 'rk'
vehicleADV.integrator(int_opt,dt)
F_x_ADV  = vehicleADV.getIntegrator()
vx_init_ego = 10   
vehicleADV.setInit([30,143.318146],vx_init_ego)
Q_ADV = [0,40,3e2,5]                            # State cost, Entries in diagonal matrix
R_ADV = [5,5]                                    # Input cost, Entries in diagonal matrix
vehicleADV.cost(Q_ADV,R_ADV)
vehicleADV.costf(Q_ADV)
L_ADV,Lf_ADV = vehicleADV.getCost()
# process_noise=DM(process_noise)
P0 = np.array([[0.002071893043635329, -9.081755508401041e-07, 6.625835814018275e-06, 3.5644803460420476e-06],
            [-9.081755508401084e-07, 0.0020727698104820867, -1.059020050149629e-05, 2.218506297137566e-06], 
            [6.625835814018292e-06, -1.0590200501496282e-05, 0.0020706909538251504, 5.4487618678242517e-08], 
            [3.5644803460420577e-06, 2.2185062971375356e-06, 5.448761867824289e-08, 0.002071025561957967]])
process_noise=np.eye(4)  # process noise
process_noise[0,0]=0.5  # x bound is [0, 3]
process_noise[1,1]=0.01/6  # y bound is [0, 0.1]
process_noise[2,2]=1.8/6  # v bound is [0, 1.8]
process_noise[3,3]=0.05/6  # psi bound is [0, 0.05]
mpc_controller = MPC(vehicleADV, np.diag([0,40,3e2,5]), np.diag([5,5]), P0, process_noise, 0.95, N)

# get initial state of truck 
x_iter = DM(int(nx),1)
x_iter[:],u_iter = vehicleADV.getInit()
print(f"initial state of the truck is: {x_iter}")
print(f"initial input of the truck is: {u_iter}")

ref_trajectory = np.zeros((nx, N + 1)) # Reference trajectory (states)
ref_trajectory[0,:] = 0
ref_trajectory[1,:] = 0
ref_trajectory[2,:] = 15
ref_control = np.zeros((nu, N))  # Reference control inputs

# Set the controller (this step initializes the optimization problem with cost and constraints)
mpc_controller.setController()

while True:
    time.sleep(0.01)
    # keep leading vehicle velocity keeps change
    # import  random  noise
    noise = np.random.normal(0, 1)
    velocity_leading = 7
    control_car = car_contoller.run_step(velocity_leading*3.6, 143.318146, False)
    car.apply_control(control_car)
    # print(f"velocity_leading: {velocity_leading}")
    
    car_state = get_state(car)
    truck_state = get_state(truck)
    car_x, car_y, car_v = car_state[C_k.X_km].item(), car_state[C_k.Y_km].item(), car_state[C_k.V_km].item()
    truck_x, truck_y, truck_v, truck_psi = truck_state[C_k.X_km].item(), truck_state[C_k.Y_km].item(), truck_state[C_k.V_km].item(), truck_state[C_k.Psi].item()
    

    if int(time.time())%1==0:
        # get the CARLA state
        p_leading = car_x
        x_iter = [truck_x, truck_y, truck_v, truck_psi]
        # print(f"x_iter is: {x_iter}")
        u_opt, x_opt = mpc_controller.solve(x_iter, ref_trajectory, ref_control, p_leading)
        x_iter=x_opt[:,1]
        # print(f"u_opt is: {u_opt}")
        # print(f"x_opt is: {x_iter}")
    print("leading velocity",car_v, "velocity of the truck: ", x_iter[2])
    
    #PID controller according to the x_iter of the MPC
    control_truck = local_controller.run_step(x_iter[2]*3.6, x_iter[1], False)
    truck.apply_control(control_truck)
