import sys
import time
sys.path.append(r'C:\Users\A490243\CARLA\CARLA_Latest\WindowsNoEditor\PythonAPI\carla')
sys.path.append(r'C:\Users\A490243\Desktop\Master_Thesis')
from enum import IntEnum
from util.utils import get_state, setup_carla_environment
from IDM import IDM
import carla
#!      ___      .______     ______   .______      .___________. __  
#!     /   \     |   _  \   /  __  \  |   _  \     |           ||  | 
#!    /  ^  \    |  |_)  | |  |  |  | |  |_)  |    `---|  |----`|  | 
#!   /  /_\  \   |   _  <  |  |  |  | |      /         |  |     |  | 
#!  /  _____  \  |  |_)  | |  `--'  | |  |\  \----.    |  |     |__| 
#! /__/     \__\ |______/   \______/  | _| `._____|    |__|     (__) 



#!  █████╗ ██████╗  ██████╗ ██████╗ ████████╗██╗        
#! ██╔══██╗██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██║        
#! ███████║██████╔╝██║   ██║██████╔╝   ██║   ██║        
#! ██╔══██║██╔══██╗██║   ██║██╔══██╗   ██║   ╚═╝        
#! ██║  ██║██████╔╝╚██████╔╝██║  ██║   ██║   ██╗        
#! ╚═╝  ╚═╝╚═════╝  ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝       
# # ------------------------change map to Town06------------------------
# import subprocess
# # Command to run your script
# command = (
#     r'cd C:\Users\A490243\CARLA\CARLA_Latest\WindowsNoEditor\PythonAPI\util && '
#     r'python config.py --map Town06')
# subprocess.run(command, shell=True)
# # Run the command

from agents.navigation.controller import VehiclePIDController
class C_k(IntEnum):
    X_km, Y_km, Psi, V_km =range(4)

car,truck = setup_carla_environment(Sameline_ACC=True)
time.sleep(1)

velocity1 = carla.Vector3D(10, 0, 0)
velocity2 = carla.Vector3D(10, 0, 0)
car.set_target_velocity(velocity1)
truck.set_target_velocity(velocity2)

# begin the IDM controller
# Initialize the IDM instance with parameters
idm_model = IDM(
    v0=15,  # Desired speed: 10 m/s 
    T=1.5,  # Time headway: 1.5 seconds
    a=3,  # Maximum acceleration:  m/s²
    b=1.67,  # Comfortable deceleration:  m/s²
    s0=2,   # Minimum distance: 2 meters
    dt=0.05 # Time step for the simulation: 0.05 seconds
)


# To start a basic agent
client = carla.Client('localhost', 2000)
world = client.get_world()
carla_map = world.get_map()
# initial the carla built in pid controller
car_contoller = VehiclePIDController(car, args_lateral = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': 0.05}, args_longitudinal = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': 0.05})
local_controller = VehiclePIDController(truck, args_lateral = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.0, 'dt': 0.05}, args_longitudinal = {'K_P': 1.95, 'K_I': 0.5, 'K_D': 0.2, 'dt': 0.05})
# To start a behavior agent with an aggressive car for truck to track
spawn_points = carla_map.get_spawn_points()
destination = carla.Location(x=500, y=143.318146, z=0.3)
print(f"destination: {destination}")
target=143.318146
while True:
    time.sleep(0.05)
    # start the running 
    control_car = car_contoller.run_step(10*3.6, target, False)
    car.apply_control(control_car)
    car_state = get_state(car)
    truck_state = get_state(truck)
    car_x, car_y, car_v = car_state[C_k.X_km].item(), car_state[C_k.Y_km].item(), car_state[C_k.V_km].item()
    truck_x, truck_y, truck_v = truck_state[C_k.X_km].item(), truck_state[C_k.Y_km].item(), truck_state[C_k.V_km].item()
    
    
    

    
    # Current state of the vehicle
    current_speed = truck_v # current speed in m/s
    print(f"current_speed: {current_speed}")
    current_position = truck_x  # current position in meters
    delta_v = car_v-truck_v  # speed difference with the vehicle in front (approaching)
    print(f"delta_v: {delta_v}")
    gap = car_x-truck_x-4  # gap to the vehicle in front in meters,length of the car is 4m

    # Update the vehicle's speed and position
    target_speed, _ = idm_model.update_speed_position(current_speed, current_position, delta_v, gap)
    print(f"target_speed: {target_speed}")

    
    
    control = local_controller.run_step(idm_model.ms_to_kmh(target_speed), car_y, False)
    # print(f"target_speed: {target_speed}, control: {control}")
    truck.apply_control(control)
    if car_x % 50 < 25:  
        target = 143.318146 + 7  
    else:
        target = 143.318146 - 7 
    time.sleep(0.05)


    
    
    
        
        