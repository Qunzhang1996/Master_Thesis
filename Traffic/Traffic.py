
#! here is the class that receive the carla and return the trajectory
import sys
import numpy as np
sys.path.append(r'C:\Users\A490243\Desktop\Master_Thesis')
from util.utils import *
class Traffic:
    def __init__(self, N=12, dt=0.2):
        self.trajectory = []
        self.N = N
        self.dt = dt
        
        
        
    def spawn_vehicle(self,world, blueprint, spawn_point):
        """
        Attempts to spawn a vehicle at a given spawn point.
        Returns the vehicle actor if successful, None otherwise.
        """
        try:
            return world.spawn_actor(blueprint, spawn_point)
        except Exception as e:
            print(f"Error spawning vehicle: {e}")
            return None
        
    def setup_complex_carla_environment(self):
        """
        Sets up a CARLA environment by connecting to the server, destroying existing vehicles,
        and spawning a selection of vehicles with initial states.
        """
        client = carla.Client('localhost', 2000)
        world = client.get_world()
        bp_lib = world.get_blueprint_library()

        # Destroy existing vehicles
        for actor in world.get_actors().filter('vehicle.*'):
            actor.destroy()

        vehicles = [
            ('vehicle.tesla.model3', 80),
            ('vehicle.carlamotors.firetruck', 20), # ! this is ego vehicle
            ('vehicle.ford.mustang', 20, -7),
            ('vehicle.carlamotors.carlacola', 20, -3.5),
            ('vehicle.lincoln.mkz_2017', 80, -3.5),
            ('vehicle.ford.ambulance', 200),
            ('vehicle.nissan.patrol_2021', 20, 3.5),
            ('vehicle.mercedes.sprinter', 60, 3.5)
        ]
        center_line = 143.318146
        spawned_vehicles = []

        for vehicle_info in vehicles:
            bp = bp_lib.find(vehicle_info[0])
            location = carla.Location(x=vehicle_info[1], y=center_line + vehicle_info[2] if len(vehicle_info) > 2 else center_line, z=0.3)
            spawn_point = carla.Transform(location)
            vehicle = self.spawn_vehicle(world, bp, spawn_point)
            spawned_vehicles.append(vehicle)

        return spawned_vehicles, center_line
    
    # ! set the velocity of the vehicles
    def set_velocity(self,vehicle_list,velocities):
        """
        Set the velocity of the vehicle
        """
        car, truck, mustang, carlacola, lincoln, ford_ambulance, patrol, mercerdes=vehicle_list
        mustang.set_target_velocity(velocities['normal'])
        #vehicle on the second lane
        carlacola.set_target_velocity(velocities['normal'])
        lincoln.set_target_velocity(velocities['aggressive'])
        #vehicle on the third lane
        car.set_target_velocity(velocities['normal'])
        truck.set_target_velocity(velocities['reference']) # ! This is ego vehicle
        ford_ambulance.set_target_velocity(velocities['aggressive'])
        #vehicle on the fourth lane
        patrol.set_target_velocity(velocities['aggressive'])
        mercerdes.set_target_velocity(velocities['aggressive'])
    
    # ! iterate the vehicle list, get and predict the N step trajectory
    def predict_trajectory(self,vehicle_list):
        """Here, assume the surrounding vehicles go straight line
        """
        for i in range(len(vehicle_list)):
            vehicle = vehicle_list[i]
            vehicle_state = get_state(vehicle) # x, y, v, psi
            #! predict the N step trajectory, store x and y in np array
            pred_traj = np.zeros((2,self.N+1))
            for j in range(self.N+1):
                # avoid the ego vehicle: truck
                # if i == 1:
                #     continue
                pred_traj[0,j] = vehicle_state[0] + vehicle_state[2] * j * self.dt
                # print("this is the velocity",vehicle_state[2])
                pred_traj[1,j] = vehicle_state[1]
            self.trajectory.append(pred_traj)
        return self.trajectory
            