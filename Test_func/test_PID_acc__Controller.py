import sys
import time
sys.path.append(r'C:\Users\A490243\CARLA\CARLA_Latest\WindowsNoEditor\PythonAPI\carla')
sys.path.append(r'C:\Users\A490243\Desktop\Master_Thesis')
from enum import IntEnum
from util.utils import get_state, setup_carla_environment,plot_paths,plot_diff
import random
import carla
# # change map to Town06
# import subprocess
# # Command to run your script
# command = (
#     r'cd C:\Users\A490243\CARLA\CARLA_Latest\WindowsNoEditor\PythonAPI\util && '
#     r'python config.py --map Town06')
# subprocess.run(command, shell=True)
# # Run the command

from agents.navigation.basic_agent import BasicAgent
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.controller import VehiclePIDController
class C_k(IntEnum):
    X_km, Y_km, Psi, V_km =range(4)

car,truck = setup_carla_environment(Sameline_ACC=True)
time.sleep(1)

velocity1 = carla.Vector3D(10, 0, 0)
velocity2 = carla.Vector3D(10, 0, 0)
car.set_target_velocity(velocity1)
truck.set_target_velocity(velocity2)
# To start a basic agent
agent = BasicAgent(car)
client = carla.Client('localhost', 2000)
world = client.get_world()
carla_map = world.get_map()
# initial the carla built in pid controller
car_contoller = VehiclePIDController(car, args_lateral = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': 0.05}, args_longitudinal = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': 0.05})
local_controller = VehiclePIDController(truck, args_lateral = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.0, 'dt': 0.05}, args_longitudinal = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': 0.05})
# To start a behavior agent with an aggressive car for truck to track
# agent = BehaviorAgent(car, behavior='aggressive')
spawn_points = carla_map.get_spawn_points()
# destination = random.choice(spawn_points).location
destination = carla.Location(x=500, y=143.318146, z=0.3)
print(f"destination: {destination}")
# agent.set_destination(destination)
target=143.318146
while True:
    time.sleep(0.05)
    car_state = get_state(car)
    truck_state = get_state(truck)
    car_x, car_y, car_v = car_state[C_k.X_km].item(), car_state[C_k.Y_km].item(), car_state[C_k.V_km].item()
    truck_x, truck_y, truck_v = truck_state[C_k.X_km].item(), truck_state[C_k.Y_km].item(), truck_state[C_k.V_km].item()
    control_car = car_contoller.run_step(10*3.6, target, False)
    car.apply_control(control_car)
    
    # Calculate distance error (aim for 5m gap)
    distance_error = (car_x - truck_x) - 5  # Desired distance - Actual distance

    # Proportional control for speed adjustment
    Kp = 3  # Proportional gain; adjust as necessary for smoother control
    speed_adjustment = Kp * distance_error
    
    # Ensure the truck tries to maintain a safe distance with a reasonable speed adjustment
    target_speed = truck_v + speed_adjustment
    control = local_controller.run_step(target_speed, car_y, False)
    truck.apply_control(control)
    print("control:",control,"target_speed:",target_speed)
    # print("car_v:",car_v,"truck_v:",truck_v)
    # Change the target of the car timely  per 100m
    if car_x % 200 < 100:  
        target = 143.318146 + 7  
    else:
        target = 143.318146 - 7 
    time.sleep(0.05)


    
    
    
        
        