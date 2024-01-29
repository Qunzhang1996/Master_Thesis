##change map to Town06
#python config.py --map Town06
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
from util.utils import get_state, setup_carla_environment

# ----------------- Carla Settings ------------------------
#! set the initial position of the car and truck
import carla
car,truck = setup_carla_environment()
velocity1 = carla.Vector3D(10, 0, 0)
velocity2 = carla.Vector3D(15, 0, 0)
car.set_target_velocity(velocity2)
car.apply_control(carla.VehicleControl(throttle=0.2, steer=1, brake=0))



# System initialization 
dt = 0.2
N=10
end_time = 40
t_axis = np.arange(0, end_time, dt)
# ----------------- Ego Vehicle Dynamics and Controller Settings ------------------------
car_model = car_VehicleModel(dt,N, width = 2, length = 4)
nx,nu,nrefx,nrefu = car_model.getSystemDim()
int_opt = 'rk'
car_model.integrator(int_opt,dt)
F_x_ADV  = car_model.getIntegrator()
vx_init_ego = 10   
car_model.setInit([124,143.318],vx_init_ego)
x_iter = DM(int(nx),1)
#get initial state and input
x_iter[:],u_iter = car_model.getInit()

# ----------------- Ego Vehicle obsver(kalman filter) Settings ------------------------
init_flag = True
F,B=car_model.calculate_AB(dt,init_flag=1)
G=np.eye(nx)*dt**2   #process noise matrix, x_iter=F@x_iter+B@u_iter+G@w
H=np.eye(nx)   #measurement matrix, y=H@x_iter

#initial state and control input
print(x_iter)
x0=x_iter+5
print(x0)
u_iter+=np.array([3.14/180,0.7*9.81])
u0=u_iter
sigma_process=0.1
sigma_measurement=0.05
P0=np.eye(nx)*(sigma_process)**4
Q0=np.eye(nx)*sigma_process**2
R0=np.eye(nx)*sigma_measurement**2
# initial the Kalman Filter
ekf=kalman_filter(F,B,H,x0,P0,Q0,R0)



#list to store the true and estimated state
true_x = []
true_y = []
estimated_x = []
estimated_y = []
x_difference = []
y_difference = []


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
    ekf.update(measurement)
    x_estimate = ekf.get_estimate
    car_model.update(x_estimate, u_iter)
    F, B = car_model.calculate_AB()
    #save the true and estimated state  
    true_x.append(float(x_iter[0]))
    true_y.append(float(x_iter[1]))
    estimated_x.append(float(x_estimate[0]))
    estimated_y.append(float(x_estimate[1]))
    x_diff = (true_x[-1] - estimated_x[-1])
    y_diff = (true_y[-1] - estimated_y[-1])
    x_difference.append(x_diff)
    y_difference.append(y_diff)
    # Plot for true and estimated paths
    plt.figure(1)
    plt.cla()
    plt.plot(true_x, true_y, label='True Path', color='blue')
    plt.plot(estimated_x, estimated_y, label='Estimated Path', color='red')
    plt.scatter(true_x[-1], true_y[-1], color='blue', s=50)
    plt.scatter(estimated_x[-1], estimated_y[-1], color='red', s=50)
    plt.title(f"Time: {t:.2f}s - Paths")
    plt.legend()
    # Flip the plot horizontally
    plt.gca().invert_xaxis()


# -------start animation----------------
fig = plt.figure(figsize=(5, 5))
ani = animation.FuncAnimation(fig, update, frames=len(t_axis), repeat=False)
ani.save('C:\\Users\\A490243\\Desktop\\Master_Thesis\\Test_func\\animation.gif', writer='imagemagick', fps=30)
# plt.show()
# ----------------- check the x, y difference ------------------------
# For x difference
plt.figure(2, figsize=(5, 5))
plt.plot(t_axis.flatten(), x_difference[:-1], label='x Difference', color='green')
plt.xlabel('Time (s)')
plt.ylabel('x Difference')
plt.title('Difference in x over Time')
plt.legend()
plt.savefig('C:\\Users\\A490243\\Desktop\\Master_Thesis\\Test_func\\x_difference.jpg')

# For y difference
plt.figure(3, figsize=(5, 5))
plt.plot(t_axis.flatten(), y_difference[:-1], label='y Difference', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('y Difference')
plt.title('Difference in y over Time')
plt.legend()
plt.savefig('C:\\Users\\A490243\\Desktop\\Master_Thesis\\Test_func\\y_difference.jpg')
plt.show()
