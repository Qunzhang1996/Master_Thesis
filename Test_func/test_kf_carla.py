## change map to Town06
import subprocess
# Command to run your script
command = (
    r'cd C:\Users\A490243\CARLA\CARLA_Latest\WindowsNoEditor\PythonAPI\util && '
    r'python config.py --map Town06')
# Run the command
subprocess.run(command, shell=True)
# ----------------- Importing Libraries ------------------------
import time
import numpy as np
import sys
# Configurations and Setup
CARLA_PATH = 'C:\\Users\\A490243\\CARLA\\CARLA_Latest\\WindowsNoEditor\\PythonAPI\\carla'
THESIS_PATH = 'C:\\Users\\A490243\\Desktop\\Master_Thesis'
sys.path.extend([CARLA_PATH, THESIS_PATH])
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from casadi import DM
from vehicleModel.vehicle_model import car_VehicleModel
from kalman_filter.kalman_filter import kalman_filter
from util.utils import get_state, setup_carla_environment,plot_paths,plot_diff
#save matrix to json file
import os
import json
# ----------------- Carla Settings ------------------------
#! set the initial position of the car and truck
import carla
car,truck = setup_carla_environment(Sameline_ACC=False)
velocity1 = carla.Vector3D(10, 0, 0)
velocity2 = carla.Vector3D(15, 0, 0)

#get initial state of truck and car,x,y,psi,v, shown seoerately
#! uncomment the following lines to get the initial state of the car and truck
#------------------get the initial state of the car and truck---------------------
time.sleep(1)
car.set_target_velocity(velocity1)
# truck_state = get_state(truck)
# car_state = get_state(car)
# print("truck_state is:","x:",truck_state[0],"y:",truck_state[1],"psi:",truck_state[2],"v:",truck_state[3])
# print("car_state is:","x:",car_state[0],"y:",car_state[1],"psi:",car_state[2],"v:",car_state[3])
# exit()
#---------------------------------------------------------------------------------

car.apply_control(carla.VehicleControl(throttle=0.0, steer=0.5, brake=0))
# System initialization 
dt = 0.2
N=10
end_time = 15
t_axis = np.arange(0, end_time, dt)
# ----------------- Ego Vehicle Dynamics and Controller Settings ------------------------
car_model = car_VehicleModel(dt,N, width = 2, length = 4)
nx,nu,nrefx,nrefu = car_model.getSystemDim()
int_opt = 'rk'
car_model.integrator(int_opt,dt)
F_x_ADV  = car_model.getIntegrator()
vx_init_ego = 10   
car_model.setInit([124,143.318146],vx_init_ego)
x_iter = DM(int(nx),1)
# get initial state and input
x_iter[:],u_iter = car_model.getInit()

# ----------------- Ego Vehicle obsver(kalman filter) Settings ------------------------
init_flag = True
F,B,_=car_model.calculate_AB(dt,init_flag=1)
G=np.eye(nx)*dt**2   #process noise matrix, x_iter=F@x_iter+B@u_iter+G@w
H=np.eye(nx)   #measurement matrix, y=H@x_iter

# initial state and control input
print(x_iter)
x0=x_iter
print(x0)
u_iter+=np.array([0,0])
u0=u_iter
sigma_process=0.1
sigma_measurement=0.01
P0=np.eye(nx)*(100*sigma_process)**4
Q0=np.eye(nx)*sigma_process**2
R0=np.eye(nx)*sigma_measurement**2
# initial the Kalman Filter
ekf=kalman_filter(F,B,H,x0,P0,Q0,R0)



# list to store the true and estimated state
true_x = []
true_x.append(float(x_iter[0]))
true_y = []
true_y.append(float(x_iter[1]))
nominal_x = [float(x0[0])]
nominal_y = [float(x0[1])]
estimated_x = [float(x0[0])]
estimated_y = [float(x0[1])]
x_difference = [abs(true_x[-1]-estimated_x[-1])]
y_difference = [abs(true_y[-1]-estimated_y[-1])]
v_difference = []
psi_difference = []


# update animation
def update(frame):
    global x_iter, u_iter,F,B
    t = t_axis[frame]
    q = np.random.normal(0.0, sigma_process, size=(nx, 1))
    r = np.random.normal(0.0, sigma_measurement, size=(nx, 1))
    #kalman filter
    x_iter = get_state(car)
    measurement = H @ x_iter + r
    ekf.predict(u0, F, B)
    #! get nominal state 
    nominal = ekf.get_predict
    ekf.update(measurement)
    x_estimate = ekf.get_estimate
    P_estimate = ekf.get_covariance
    # print("P_estimate is:",P_estimate)
    car_model.update(x_estimate, u_iter)
    F, B,_ = car_model.calculate_AB()
    
    #save the true and estimated state  
    true_x.append(float(x_iter[0]))
    true_y.append(float(x_iter[1]))
    nominal_x.append(float(nominal[0]))
    nominal_y.append(float(nominal[1]))
    estimated_x.append(float(x_estimate[0]))
    estimated_y.append(float(x_estimate[1]))
    
    
    
    x_diff = (abs(estimated_x[-1] - true_x[-1]))
    y_diff = (abs(estimated_y[-1] - true_y[-1]))
    # print("x_diff is:",x_diff,"y_diff is:",y_diff)
    x_difference.append(x_diff)
    y_difference.append(y_diff)
    plot_paths(true_x, true_y,estimated_x, estimated_y,t)
    
    
    # Convert F,B,P_estimate to a list of lists
    F_np=np.array(F)
    F_list=F_np.tolist()
    B_np=np.array(B)
    B_list=B_np.tolist()
    P_estimate_np = np.array(P_estimate)
    P_estimate_list = P_estimate_np.tolist()

    # Define the file path
    file_path = os.path.join("C:", os.sep, "Users", "A490243", "Desktop", "Master_Thesis", "Data", "Converged_P.json")
    # Write P_estimate to a JSON file
    with open(file_path, 'w') as file:
        json.dump(P_estimate_list, file)
      
    # Define the file path
    file_path = os.path.join("C:", os.sep, "Users", "A490243", "Desktop", "Master_Thesis", "Data", "F_matrix.json")
    # Write P_estimate to a JSON file
    with open(file_path, 'w') as file:
        json.dump(F_list, file)
    
    # Define the file path
    file_path = os.path.join("C:", os.sep, "Users", "A490243", "Desktop", "Master_Thesis", "Data", "B_matrix.json")
    # Write P_estimate to a JSON file
    with open(file_path, 'w') as file:
        json.dump(B_list, file)

# -------start animation----------------
fig = plt.figure(figsize=(5, 5))
ani = animation.FuncAnimation(fig, update, frames=len(t_axis), repeat=False)
# ani.save('C:\\Users\\A490243\\Desktop\\Master_Thesis\\Figure\\animation.gif', writer='imagemagick', fps=30)
plt.show()
# ----------------- check the x, y difference ------------------------
plot_diff(t_axis,x_difference,y_difference)

