"""
This func is used to test the kalman filter for a kinematic bicycle model
"""
import matplotlib.pyplot as plt
import sys
from casadi import *
path_to_add='C:\\Users\\A490243\\Desktop\\Master_Thesis'
sys.path.append(path_to_add)
from Autonomous_Truck_Sim.vehicleModelGarage import vehBicycleKinematic
from vehicleModel.vehicle_model import car_VehicleModel
from kalman_filter.kalman_filter import kalman_filter
import matplotlib.animation as animation
# System initialization 
dt = 0.2
N=10
end_time = 40
t_axis = np.arange(0, end_time, dt)
# ----------------- Ego Vehicle Dynamics and Controller Settings ------------------------
car_model = car_VehicleModel(dt,N, width = 2, length = 4)
nx,nu,nrefx,nrefu = car_model.getSystemDim()
# Integrator
int_opt = 'rk'
car_model.integrator(int_opt,dt)
F_x_ADV  = car_model.getIntegrator()
vx_init_ego = 55/3.6    
car_model.setInit([0,0],vx_init_ego)
x_iter = DM(int(nx),1)
#get initial state and input
x_iter[:],u_iter = car_model.getInit()

# ----------------- Ego Vehicle obsver(kalman filter) Settings ------------------------
init_flag = True
F,B=car_model.calculate_AB(dt,init_flag=1)
G=np.eye(nx)*dt**2   #process noise matrix, x_iter=F@x_iter+B@u_iter+G@w
H=np.eye(nx)   #measurement matrix, y=H@x_iter
#initial state and control input
x0=x_iter
u_iter+=np.array([3.14/180,0.7*9.81])
u0=u_iter
sigma_process=1
sigma_measurement=0.1
P0=np.eye(nx)*sigma_process**4
Q0=np.eye(nx)*sigma_process**2
R0=np.eye(nx)*sigma_measurement**2
# initial the Kalman Filter
ekf=kalman_filter(F,B,H,x0,P0,Q0,R0)

# -------start animation----------------
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
    
    #add noise to the state
    x_iter = F_x_ADV(x_iter,u_iter)
    #measurements
    measurement = H @ x_iter + r

    ekf.predict(u0, F, B)
    ekf.update(measurement)
    x_estimate = ekf.get_estimate


    car_model.update(x_estimate, u_iter)
    F, B = car_model.calculate_AB()
    
    
    true_x.append(float(x_iter[0]))
    true_y.append(float(x_iter[1]))
    estimated_x.append(float(x_estimate[0]))
    estimated_y.append(float(x_estimate[1]))
    
    # print("this is true x",true_x)
    # print("this is estimated x",x_iter)
    # exit()
    
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

    

# create a figure
fig = plt.figure(figsize=(5, 2))
ani = animation.FuncAnimation(fig, update, frames=len(t_axis), repeat=False)
plt.show()


plt.figure(1)  # Create a new figure for the x difference6
plt.plot(t_axis.flatten(), x_difference[:-1], label='x Difference', color='green')
plt.xlabel('Time (s)')
plt.ylabel('x Difference')
plt.title('Difference in x over Time')
plt.legend()
# Plot the difference in y
plt.figure(2)  # Create a new figure for the y difference
plt.plot(t_axis.flatten(), y_difference[:-1], label='y Difference', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('y Difference')
plt.title('Difference in y over Time')
plt.legend()

plt.show()