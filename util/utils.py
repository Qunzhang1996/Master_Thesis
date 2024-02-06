import math
import carla
import numpy as np
import matplotlib.pyplot as plt


class Param:
    """
    here is the parameters for the vehicle
    """
    
    dt = 0.1  # [s] time step
     # vehicle config
    RF = 16.71554/3 # [m] distance from rear to vehicle front end of vehicle
    RB = 0.8  # [m] distance from rear to vehicle back end of vehicle
    W = 2.4  # [m] width of vehicle
    WD = 0.7 * W  # [m] distance between left-right wheels
    WB =  4  # [m] Wheel base
    TR = 0.44  # [m] Tyre radius
    TW = 0.7  # [m] Tyre width





def get_state(vehicle):
    """Here is the func that help to get the state of the vehicle

    Args:
        vehicle (_type_): _description_

    Returns:
        _type_: _4X1_ vector that contains the state of the vehicle
        x, y, psi, v
    """
    vehicle_pos = vehicle.get_transform()
    vehicle_loc = vehicle_pos.location
    vehicle_rot = vehicle_pos.rotation
    vehicle_vel = vehicle.get_velocity()

    # Extract relevant states
    x = vehicle_loc.x 
    y = vehicle_loc.y 
    psi = math.radians(vehicle_rot.yaw)  # Convert yaw to radians
    v = math.sqrt(vehicle_vel.x**2 + vehicle_vel.y**2)
    # v = vehicle_vel.length()  #converting it to km/hr

    return np.array([[x, y, psi, v]]).T

def setup_carla_environment(Sameline_ACC=True):
    """
    Sets up the CARLA environment by connecting to the server, 
    spawning vehicles, and setting their initial states.
    Returns the car and truck actors.
    """
    '''
    Sameline_ACC=TRUE: the car and truck are in the same lane, and the truck trakcing the car
    Sameline_ACC=FALSE: the car and truck are in the different lane
    '''
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    # Destroy existing vehicles
    for actor in world.get_actors().filter('vehicle.*'):
        actor.destroy()

    if Sameline_ACC :
        # Spawn Tesla Model 3
        car_bp = bp_lib.find('vehicle.ford.ambulance')
        car_spawn_point = carla.Transform(carla.Location(x=50, y=143.318146, z=0.3))
        car = spawn_vehicle(world, car_bp, car_spawn_point)

        # Spawn Firetruck
        truck_bp = bp_lib.find('vehicle.carlamotors.european_hgv')
        truck_spawn_point = carla.Transform(carla.Location(x=30, y=143.318146, z=0.3))
        truck = spawn_vehicle(world, truck_bp, truck_spawn_point)
        return car, truck
    else:
        # Spawn Tesla Model 3
        car_bp = bp_lib.find('vehicle.tesla.model3')
        car_spawn_point = carla.Transform(carla.Location(x=124, y=143.318146, z=0.3))
        car = spawn_vehicle(world, car_bp, car_spawn_point)

        # Spawn Firetruck
        truck_bp = bp_lib.find('vehicle.carlamotors.firetruck')
        truck_spawn_point = carla.Transform(car_spawn_point.location + carla.Location(y=3.5, x=-100))
        truck = spawn_vehicle(world, truck_bp, truck_spawn_point)

        return car, truck

def spawn_vehicle(world, blueprint, spawn_point):
    """
    Attempts to spawn a vehicle at a given spawn point.
    Returns the vehicle actor if successful, None otherwise.
    """
    try:
        return world.spawn_actor(blueprint, spawn_point)
    except Exception as e:
        print(f"Error spawning vehicle: {e}")
        return None
    
    
def plot_paths(true_x, true_y,estimated_x, estimated_y,t):
    plt.figure(1)
    plt.cla()
    plt.plot(true_x, true_y, label='True Path', color='blue')
    plt.plot(estimated_x, estimated_y, label='Estimated Path', color='red')
    plt.scatter(true_x[-1], true_y[-1], color='blue', s=50)
    plt.scatter(estimated_x[-1], estimated_y[-1], color='red', s=50)
    plt.title(f"Time: {t:.2f}s - Paths")
    plt.legend()
    # plt.gca().invert_xaxis()


def plot_diff(t_axis,x_difference,y_difference):
    # For x difference
    plt.figure(2, figsize=(5, 5))
    plt.plot(t_axis.flatten(), x_difference[:-2], label='x Difference', color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('x Difference')
    plt.title('Difference in x over Time')
    plt.legend()
    plt.savefig('C:\\Users\\A490243\\Desktop\\Master_Thesis\\Figure\\x_difference.jpg')

    # For y difference
    plt.figure(3, figsize=(5, 5))
    plt.plot(t_axis.flatten(), y_difference[:-2], label='y Difference', color='orange')
    plt.xlabel('Time (s)')
    plt.ylabel('y Difference')
    plt.title('Difference in y over Time')
    plt.legend()
    plt.savefig('C:\\Users\\A490243\\Desktop\\Master_Thesis\\Figure\\y_difference.jpg')
    plt.show()
    
#make sure angle in -pi pi
def angle_trans(angle):
    if angle > np.pi:
        angle = angle - 2*np.pi
    elif angle < -np.pi:
        angle = angle + 2*np.pi
    return angle

def check_dimensions_car(nx, nu):
    """
    Check if the dimensions nx and nu are as expected.

    Parameters:
    - nx (int): The number of state variables.
    - nu (int): The number of control inputs.

    Raises:
    - ValueError: If nx is not equal to 4 or nu is not equal to 2.
    """
    if nx != 4 or nu != 2:
        raise ValueError("Error: nx must be equal to 4 and nu must be equal to 2.")
    


