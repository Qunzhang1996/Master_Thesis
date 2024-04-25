
#! here is the class that receive the carla and return the trajectory
import sys
import random
import numpy as np
sys.path.append(r'C:\Users\A490243\Desktop\Master_Thesis')
from util.utils import *

class surroundVehicle:
    """this is the struct used to store the information of the surrounding vehicles
    """
    def __init__(self, name, vehicle_id, x, y, v, psi, N, dt , center_line = 143.318146, laneWidth = 3.5):
        self.name = name
        self.vehicle_id = vehicle_id
        self.x = x
        self.y = y
        self.v = v  # velocity
        self.v0=v
        self.psi = psi  # orientation or heading
        self.state = np.array([x, y, v, psi])
        self.N = N
        self.dt = dt
        self.laneWidth = laneWidth
        self.init_bound = center_line-self.laneWidth/2
        self.leadWidth = 2.032
        self.leadLength = 4.78536
        # define the l_front and l_rear
        self.l_front = self.leadLength/2
        self.l_rear = self.leadLength/2
        self.type = "normal"
          
    def getLength(self):
        return self.l_front, self.l_rear 
        
    def getSize(self):
        return self.leadWidth, self.leadLength
    
    def getPosition(self):
        return np.array([self.x, self.y])
        
    def predict_trajectory(self):
        """
        based current state, predict the trajectory of the vehicle,including N=0
        """
        self.predict_state = np.zeros((4,self.N+1))
        for j in range(self.N+1):
            self.predict_state[0,j] = self.state[0] + self.state[2] * j * self.dt
            self.predict_state[1,j] = self.state[1]
            self.predict_state[2,j] = self.state[2]
            self.predict_state[3,j] = self.state[3]
        return  self.predict_state
        
    def getLane(self):
        if self.y > self.init_bound+self.laneWidth:
            return 1
        elif self.y < self.init_bound:
            return -1
        else:
            return 0
        
class Traffic:
    def __init__(self, N, dt, laneWidth=3.5):
        self.trajectory = []
        self.N = N
        self.dt = dt
        self.nx = 4
        self.laneWidth = laneWidth
        self.vehicle_list = []
        self.video_writer = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (800, 600))
        
        
          
    def spawn_vehicle(self, world, blueprint, spawn_point):
        try:
            return world.spawn_actor(blueprint, spawn_point)
        except Exception as e:
            print(f"Error spawning vehicle: {e}")
            return None

    def random_lane_offset(self):
        return random.choice([None, self.laneWidth, -self.laneWidth])

    def generate_vehicle_positions(self, vehicles, min_distance):
        generated_positions = []
        lanes = {}

        for vehicle_type, base_x_position, lane_offset in vehicles:
            if lane_offset not in lanes:
                lanes[lane_offset] = []
            
            # Generate initial random position
            x_position = base_x_position if base_x_position is not None else random.uniform(10, 100)
            valid_position = False

            # Ensure the vehicle maintains the minimum required distance from others in the same lane
            while not valid_position:
                print("INFO:  Generating random position.")
                valid_position = True
                for pos in lanes[lane_offset]:
                    if abs(x_position - pos) < min_distance:
                        valid_position = False
                        x_position =pos + random.uniform(min_distance, 65)
                        break
            
            lanes[lane_offset].append(x_position)
            generated_positions.append((vehicle_type, x_position, lane_offset))
        
        return generated_positions
           
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

        # Define the vehicle data with some needing random placement
        vehicles = [
            ('vehicle.tesla.model3', None, self.random_lane_offset()),
            ('vehicle.carlamotors.firetruck', None, None),  # This is the ego vehicle
            ('vehicle.tesla.model3', None, self.random_lane_offset()),
            ('vehicle.tesla.model3', None, self.random_lane_offset()),
            ('vehicle.tesla.model3', None, self.random_lane_offset()),
            ('vehicle.tesla.model3', None, self.random_lane_offset()),
            ('vehicle.tesla.model3', None, self.random_lane_offset()),
            ('vehicle.tesla.model3', None, self.random_lane_offset())
        ]
        center_line = 143.318146

        # Randomize positions while ensuring minimum distance
        positions = self.generate_vehicle_positions(vehicles, 55)

        # Spawn the vehicles
        for vehicle_info in positions:
            model, x_position, lane_offset = vehicle_info
            y_position = 143.318146 + (lane_offset if lane_offset is not None else 0)
            location = carla.Location(x=x_position, y=y_position, z=0.3)
            spawn_point = carla.Transform(location)
            bp = bp_lib.find(model)
            vehicle = self.spawn_vehicle(world, bp, spawn_point)
            self.vehicle_list.append(vehicle)
            
            
            if model == 'vehicle.carlamotors.firetruck':
                # Setup camera sensor
                self.camera_bp = bp_lib.find('sensor.camera.rgb')
                self.camera_bp.set_attribute('image_size_x', '800')
                self.camera_bp.set_attribute('image_size_y', '600')
                self.camera_bp.set_attribute('fov', '90')
                self.camera_bp.set_attribute('sensor_tick', '0.033')  # for 30 FPS


                # Adjust the camera location relative to the vehicle
                camera_transform = carla.Transform(carla.Location(x=3.0, z=2.7))
                self.camera = world.spawn_actor(self.camera_bp, camera_transform, attach_to=vehicle)

                # Listen to the camera data and display it
                

                self.camera.listen(self.process_image)

        print(f"INFO:  Spawned {len(self.vehicle_list)} vehicles in CARLA Environment.")
        return self.vehicle_list, center_line


    
    
    
    # def process_image(self,image):
    #                 array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    #                 array = np.reshape(array, (image.height, image.width, 4))
    #                 array = array[:, :, :3]  # Remove the alpha channel
    #                 array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    #                 cv2.imshow('Camera View', array)
    #                 cv2.waitKey(1)
    #                 # gray_image = cv2.cvtColor(array[:, :, :3], cv2.COLOR_RGB2GRAY)  # 转换为灰度图
    #                 # cv2.imshow('Camera View', gray_image)
    #                 # cv2.waitKey(1)
    
    def process_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # Remove the alpha channel
        bgr_image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        self.video_writer.write(bgr_image)

    def close_video_writer(self):
        self.video_writer.release()
                    
                    
                    
                    
                    
    # def generate_vehicle_positions(self, vehicles, min_distance):
    #     positions = []
    #     max_attempts = 1000  # Maximum attempts for random generation to prevent infinite loop

    #     for model, fixed_pos, lane_offset in vehicles:
    #         if fixed_pos is not None:
    #             positions.append((model, fixed_pos, lane_offset))
    #         else:
    #             attempts = 0
    #             while attempts < max_attempts:
    #                 start_position = random.randint(30, 230)  # Increased range for more flexibility
    #                 if all(abs(start_position - pos[1]) >= min_distance for pos in positions):
    #                     positions.append((model, start_position, lane_offset))
    #                     break
    #                 attempts += 1
    #             if attempts == max_attempts:
    #                 raise ValueError("Failed to place all vehicles with the required minimum distance.")

    #     return positions


    # def spawn_vehicle(self, world, blueprint, spawn_point):
    #     """
    #     Spawns a vehicle at a given spawn point.
    #     """
    #     return world.spawn_actor(blueprint, spawn_point)
    
    
    #! standard complex environment 
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

        # vehicles = [
        #     ('vehicle.tesla.model3', 30,-self.laneWidth),
        #     ('vehicle.carlamotors.firetruck', 20 ), # ! this is ego vehicle    'vehicle.carlamotors.firetruck'
        #     ('vehicle.tesla.model3', 130, ),
        #     ('vehicle.tesla.model3', 80, -self.laneWidth),
        #     ('vehicle.tesla.model3', 125,self.laneWidth ),
        #     ('vehicle.tesla.model3', 60),
        #     ('vehicle.tesla.model3', 170, -self.laneWidth),
        #     ('vehicle.tesla.model3', 100, self.laneWidth)
        # ]
        vehicles = [
            ('vehicle.tesla.model3', 30,-self.laneWidth),
            ('vehicle.carlamotors.firetruck', 10 ), # ! this is ego vehicle    'vehicle.carlamotors.firetruck'
            ('vehicle.tesla.model3', 110, -self.laneWidth),
            ('vehicle.tesla.model3', 80, -self.laneWidth),
            ('vehicle.tesla.model3', 190, ),
            ('vehicle.tesla.model3', 60),
            ('vehicle.tesla.model3', 150, -self.laneWidth),
            ('vehicle.tesla.model3', 110, self.laneWidth)
        ]
        center_line = 143.318146
        

        for vehicle_info in vehicles:
            bp = bp_lib.find(vehicle_info[0])
            location = carla.Location(x=vehicle_info[1], y=center_line + vehicle_info[2] if len(vehicle_info) > 2 else center_line, z=0.3)
            spawn_point = carla.Transform(location)
            vehicle = self.spawn_vehicle(world, bp, spawn_point)
            self.vehicle_list.append(vehicle)
            
            if vehicle_info[0] == 'vehicle.carlamotors.firetruck':
                # Setup camera sensor
                self.camera_bp = bp_lib.find('sensor.camera.rgb')
                self.camera_bp.set_attribute('image_size_x', '800')
                self.camera_bp.set_attribute('image_size_y', '600')
                self.camera_bp.set_attribute('fov', '90')
                self.camera_bp.set_attribute('sensor_tick', '0.033')  # for 30 FPS

                # Adjust the camera location relative to the vehicle
                camera_transform = carla.Transform(carla.Location(x=3.0, z=2.7))
                self.camera = world.spawn_actor(self.camera_bp, camera_transform, attach_to=vehicle)

                # Listen to the camera data and display it
                

                self.camera.listen(self.process_image)
            
            
            
            
            
        print(f"INFO:  Spawned {len(self.vehicle_list)} vehicles in CARLA Environment.")
        return self.vehicle_list, center_line
    
    # ! set the velocity of the vehicles
    def set_velocity(self,velocities):
        """
        Set the velocity of the vehicle
        """
        car, truck, mustang, carlacola, lincoln, ford_ambulance, patrol, mercerdes=self.vehicle_list
        mustang.enable_constant_velocity(velocities['normal'])
        #vehicle on the second lane
        carlacola.enable_constant_velocity(velocities['normal'])
        lincoln.enable_constant_velocity(velocities['passive'])
        #vehicle on the third lane
        car.enable_constant_velocity(velocities['normal'])
        truck.set_target_velocity(velocities['reference']) # ! This is ego vehicle
        ford_ambulance.enable_constant_velocity(velocities['normal'])
        #vehicle on the fourth lane
        patrol.enable_constant_velocity(velocities['normal'])
        mercerdes.enable_constant_velocity(velocities['normal'])
        
    # ! here, return the state of the vehicle in the vehicle list      
    def getVehicles(self):
        '''
        this function returns the list of vehicles(class)
        '''
        self.Total_vehicle_list = []
        for i in range(len(self.vehicle_list)):
            # if i==1:
            #     continue  # avoid putting the ego vehicle in the list
            vehicle_name = self.vehicle_list[i]
            vehicle_state = get_state(vehicle_name)
            surroundVehicle_=surroundVehicle(vehicle_name, i, vehicle_state[0], \
                                                                vehicle_state[1], vehicle_state[2], vehicle_state[3],self.N, self.dt)
            if i==1:
                self.leadWidth = 2.0
                self.leadLength = 5
                surroundVehicle_.leadWidth = 2.59
                surroundVehicle_.leadLength = 8.46
            self.Total_vehicle_list.append(surroundVehicle_)
        return self.Total_vehicle_list     
    
    def getStates(self):
        """ return the state of all surrounding vehicles
        """ 
        states = np.zeros((self.nx,self.Nveh))
        for i in range(len(self.vehicle_list)):
            vehicle_name = self.vehicle_list[i]
            states[:,i]=get_state(vehicle_name).flatten()
        return states
    
    # ! iterate the vehicle list, get and predict the N step trajectory
    def prediction(self):
        """Here, assume the surrounding vehicles go straight line
        """
        self.states = np.zeros((self.nx,self.N+1,len(self.vehicle_list)))
        for i in range(len(self.vehicle_list)):
            vehicle = self.vehicle_list[i]
            vehicle_state = get_state(vehicle) # x, y, v, psi
            
            #! CHECK THE Velocity of the vehicle, if the velocity is smaller than 5, rasie an error
            if vehicle_state[2] < 5:
                raise ValueError("The velocity of the vehicle is smaller than 5 m/s")
            
            #! predict the N step trajectory, store x and y in np array
            pred_traj = np.zeros((self.nx, self.N+1))
            for j in range(self.N+1):
                pred_traj[0,j] = vehicle_state[0] + vehicle_state[2] * j * self.dt
                pred_traj[1,j] = vehicle_state[1]
                pred_traj[2,j] = vehicle_state[2]
                pred_traj[3,j] = vehicle_state[3]
            self.states[:,:,i] = pred_traj
        return self.states
    
    
    def prejct_trajectory_complex(self):
        """
        #!  Here, try to use kinematic model to predict the trajectory
        """
        pass
    
    
    
    
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
        leadWidth = 1.9
        leadLength = 4.694
        
        return leadWidth, leadLength
    
    def get_laneWidth(self):
        return self.laneWidth
    
    def getNveh(self):
        return len(self.vehicle_list)
    
    def getDim(self):
        self.Nveh = len(self.vehicle_list)
        return self.Nveh 

    def getEgo(self):
        return self.vehicle_list[1]
    
    def getInitBound(self):
        return 143.318146-self.laneWidth/2