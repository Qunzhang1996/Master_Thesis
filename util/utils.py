import math
import carla
import numpy as np


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

def setup_carla_environment():
    """
    Sets up the CARLA environment by connecting to the server, 
    spawning vehicles, and setting their initial states.
    Returns the car and truck actors.
    """
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    # Destroy existing vehicles
    for actor in world.get_actors().filter('vehicle.*'):
        actor.destroy()

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