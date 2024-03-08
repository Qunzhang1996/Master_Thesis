
#! here is the class that receive the carla and return the trajectory
import sys
import numpy as np
sys.path.append(r'C:\Users\A490243\Desktop\Master_Thesis')
from util.utils import *
class Traffic:
    def __init__(self, N=12, dt=0.2, laneWidth=3.5):
        self.trajectory = []
        self.N = N
        self.dt = dt
        self.nx = 4
        self.laneWidth = laneWidth
        
          
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
            ('vehicle.ford.mustang', 20, -2*self.laneWidth),
            # ('vehicle.carlamotors.carlacola', 20, -self.laneWidth),
            # ('vehicle.lincoln.mkz_2017', 80, -self.laneWidth),
            # ('vehicle.ford.ambulance', 200),
            # ('vehicle.nissan.patrol_2021', 20, self.laneWidth),
            # ('vehicle.mercedes.sprinter', 60, self.laneWidth)
        ]
        center_line = 143.318146
        self.vehicle_list = []

        for vehicle_info in vehicles:
            bp = bp_lib.find(vehicle_info[0])
            location = carla.Location(x=vehicle_info[1], y=center_line + vehicle_info[2] if len(vehicle_info) > 2 else center_line, z=0.3)
            spawn_point = carla.Transform(location)
            vehicle = self.spawn_vehicle(world, bp, spawn_point)
            self.vehicle_list.append(vehicle)
        return self.vehicle_list, center_line
    
    # ! set the velocity of the vehicles
    def set_velocity(self,velocities):
        """
        Set the velocity of the vehicle
        """
        # car, truck, mustang, carlacola, lincoln, ford_ambulance, patrol, mercerdes=self.vehicle_list
        car, truck, mustang=self.vehicle_list
        # car, truck, mustang =self.vehicle_list
        mustang.set_target_velocity(velocities['normal'])
        #vehicle on the second lane
        # carlacola.set_target_velocity(velocities['normal'])
        # lincoln.set_target_velocity(velocities['aggressive'])
        #vehicle on the third lane
        car.set_target_velocity(velocities['passive'])
        truck.set_target_velocity(velocities['reference']) # ! This is ego vehicle
        # ford_ambulance.set_target_velocity(velocities['aggressive'])
        #vehicle on the fourth lane
        # patrol.set_target_velocity(velocities['aggressive'])
        # mercerdes.set_target_velocity(velocities['aggressive'])
    
    # ! iterate the vehicle list, get and predict the N step trajectory
    def predict_trajectory(self):
        """Here, assume the surrounding vehicles go straight line
        """
        self.states = np.zeros((self.nx,self.N+1,len(self.vehicle_list)))
        for i in range(len(self.vehicle_list)):
            vehicle = self.vehicle_list[i]
            vehicle_state = get_state(vehicle) # x, y, v, psi
            #! predict the N step trajectory, store x and y in np array
            pred_traj = np.zeros((self.nx, self.N+1))
            for j in range(self.N+1):
                pred_traj[0,j] = vehicle_state[0] + vehicle_state[2] * j * self.dt
                pred_traj[1,j] = vehicle_state[1]
                pred_traj[2,j] = vehicle_state[2]
                pred_traj[3,j] = vehicle_state[3]
            self.states[:,:,i] = pred_traj
        return self.states
    
    def getDim(self):
        self.Nveh = len(self.vehicle_list)
        return self.Nveh
    
    # get velocity of the vehicles use get state function
    def get_velocity(self):
        self.getDim()
        self.velocities = np.zeros((self.Nveh))
        for i in range(self.Nveh):
            vehicle = self.vehicle_list[i]
            self.velocities[i]=get_state(vehicle)[2]
        return self.velocities
    
    def get_size(self):
        """return the common size of the vehicles"""
        leadWidth = 1.7
        leadLength = 6
        
        return leadWidth, leadLength
    
    def get_laneWidth(self):
        return self.laneWidth
