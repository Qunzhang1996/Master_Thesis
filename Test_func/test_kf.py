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
end_time = 30
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
u_iter+=np.array([10*3.14/180,0])
u0=u_iter
sigma_process=0.0001
sigma_measurement=0.01
P0=np.eye(nx)*sigma_process**4
Q0=np.eye(nx)*sigma_process**2
R0=np.eye(nx)*sigma_measurement**2

# initial the Kalman Filter
ekf=kalman_filter(F,B,H,x0,P0,Q0,R0)
# list to store the true and estimated state
true_x = []
true_y = []
estimated_x = []
estimated_y = []
true_x.append(float(x0[0]))
true_y.append(float(x0[1]))
# update animation
def update(frame):
    global x_iter, u_iter,F,B
    t = t_axis[frame]
    q = np.random.normal(0.0, sigma_process, size=(nx, 1))
    r = np.random.normal(0.0, sigma_measurement, size=(nx, 1))
    x = F_x_ADV(x_iter,u_iter) +G@q
    y = H @ x_iter + r

    ekf.predict(u0, F, B)
    print(u0)
    ekf.update(y)
    x_iter = ekf.get_estimate

    car_model.update(x_iter, u_iter)
    F, B = car_model.calculate_AB()

    true_x.append(float(x[0]))
    true_y.append(float(x[1]))
    estimated_x.append(float(ekf.get_estimate[0]))
    estimated_y.append(float(ekf.get_estimate[1]))

    plt.cla()
    plt.plot(true_x, true_y, label='True Path', color='blue')
    plt.plot(estimated_x, estimated_y, label='Estimated Path', color='red')
    plt.scatter(true_x[-1], true_y[-1], color='blue', s=50)
    plt.scatter(estimated_x[-1], estimated_y[-1], color='red', s=50)
    plt.title(f"Time: {t:.2f}s")
    plt.legend()
    # plt.xlim(0, 1000)
    # plt.ylim(-100, 100)

# create a figure
fig = plt.figure(figsize=(5, 2))
ani = animation.FuncAnimation(fig, update, frames=len(t_axis), repeat=False)

plt.show()
