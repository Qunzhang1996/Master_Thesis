# Behavior of surrounding vehicles, together and alone
from casadi import *
import numpy

class combinedTraffic:
    """
    Methods for all vehicles in the current scenario
    """
    def __init__(self,vehicles,egoVehicle,N,f_controller, respawnTol = 75, respawnPos = 120):
        self.vehicles = vehicles
        self.egoVehicle = egoVehicle
        self.Nveh = len(vehicles)
        self.nx,self.N = vehicles[0].getDim()
        self.N_pred = int(self.N*f_controller)
        self.f_c = f_controller

        self.respawnTol = respawnTol            # Position behind ego vehicle when vehicles become respawned
        self.respawnPos = respawnPos
        self.randPxRange = 30
        self.randVxRange = 3

    def setScenario(self,scenario):
        for vehicle in self.vehicles:
            vehicle.setScenario(scenario)

    def prediction(self):
        # self.states = np.zeros((self.nx,self.N+1,self.Nveh))
        # for i in range(self.Nveh):
        #     self.states[:,:,i] = self.vehicles[i].prediction(self.N)

        self.states = np.zeros((self.nx,self.N_pred+1,self.Nveh))
        for i in range(self.Nveh):
            self.states[:,:,i] = self.vehicles[i].prediction(self.N_pred)

        # Downsample to controller frequency
        self.states = self.states[:,0::self.f_c,:]

        return self.states

    def update(self):
        for i in range(self.Nveh):
            # print("Vehicle ", i)
            self.vehicles[i].setUpdate(self.vehicles,self.egoVehicle)

        for vehicle in self.vehicles:
            vehicle.pushUpdate()

    def getStates(self):
        states = np.zeros((self.nx,self.Nveh))
        for i in range(self.Nveh):
            states[:,i] = np.squeeze(self.vehicles[i].getState())
        return states

    def getFeatures(self):
        features = np.zeros((self.nx+1,self.Nveh))
        for i in range(self.Nveh):
            features[:,i] = np.squeeze(self.vehicles[i].getFeatures())
        return features

    def getReference(self):
        reference = np.zeros((4,self.Nveh))
        for i in range(self.Nveh):
            reference[:,i] = np.squeeze(self.vehicles[i].getReference())
        return reference

    def getDim(self):
        return self.Nveh

    def getVehicles(self):
        return self.vehicles

    def tryRespawn(self,egoPx):
        # Respawns vehicles that are "far enough" out of distance
        for vehicle in self.vehicles:
            if vehicle.getState()[0] < egoPx - self.respawnTol:
                # They also need to maintain atleast the same speed as the prior vehicle!
                respawnPx = egoPx.full().item() +self.respawnPos + np.random.uniform(-self.randPxRange,self.randPxRange)
                respawnVx = np.random.uniform(-self.randVxRange,self.randVxRange)

                # Avoid instant collision with other vehicle when respawning
                doRespawn = True
                dangerDist = 35
                dangerTimeHeadway = 1
                dangerVelocityDiff = 2
                vEgo = vehicle.getState()[2]
                for otherVehicle in self.vehicles:
                    vOther = otherVehicle.getState()[2]
                    pxOther = otherVehicle.getState()[0]
                    if type(pxOther) == "casadi.casadi.DM":
                        pxOther = pxOther.full().astype("float")
                    
                    dangerZone = dangerDist + dangerTimeHeadway*vOther + dangerVelocityDiff*(vOther-vEgo)
                    if (dangerZone > np.abs(respawnPx - pxOther)) and (vehicle.getLane() == otherVehicle.getLane()):
                        doRespawn = False
                
                if doRespawn == True:
                    print("Vehicle is respawning")
                    vehicle.respawn(respawnPx,respawnVx)
            

class vehicleSUMO:
    """
    A relistic traffic simulator
    IDM for acceleration, MOBIL for lane change and PI controller for steering angle
    """
    def __init__(self,dt,N,p0,v0,width = 2.032, length = 4.78536, type = "normal"):
        """
        type - setting for vehicle behavoir 
            normal : Close to speed limit, considers other vehicles
            passive : Under speed limit, high consideration of other vehicles
            aggressive : Above speed limit, low/no consideration of other vehicles
        """
        # Initialize parameters
        self.type = type
        self.dt = dt
        self.N  = N
        self.p0 = p0                        # Initial vehicle position (x,y)
        self.v0_x = v0[0]
        self.width = width
        self.length = length
        self.nx = 4

        self.p = self.p0
        self.v = self.v0_x
        self.theta  = 0.0
        self.state = np.zeros((4,self.N+1),dtype=float)             # [p_x, p_y, v_x, theta, steering angle, acceleration]
        self.state[:2,0] = self.p
        self.state[2,0] = self.v0_x
        self.state[3,0] = self.theta

        self.u = np.zeros((2,N), dtype= float)

        # Set model parameters
        if type == "passive":
            self.typeEncoding = 1
            self.IDM_s0 = 5
            self.IDM_T  = 2.5
            self.IDM_amax = 0.5
            self.IDM_b  = 1.5
            self.IDM_a_exp = 4

            self.MOBIL_p = 1.5
            self.MOBIL_tresh = 0.5
            self.MOBIL_bsafe = 3
            self.MOBIL_requiredSpace_front = 70
            self.MOBIL_requiredSpace_rear = -30

            self.v0 = self.v0_x - 15/3.6
            self.v = self.v0_x # + np.random.uniform(-10/3.6,10/3.6)

            # Bonus distance + velocity difference considered for truck
            self.truck_respect_s = 5
            self.truck_respect_v = 5

        elif type == "normal":
            self.typeEncoding = 2
            self.IDM_s0 = 3
            self.IDM_T  = 2
            self.IDM_amax = 0.7
            self.IDM_b  = 1.7
            self.IDM_a_exp = 4

            self.MOBIL_p = 1
            self.MOBIL_tresh = 0.5
            self.MOBIL_bsafe = 4
            self.MOBIL_requiredSpace_front = 60
            self.MOBIL_requiredSpace_rear = -30

            self.v0 = self.v0_x
            self.v = self.v0_x  # + np.random.uniform(-15/3.6,15/3.6)

            # Bonus distance + velocity difference considered for truck
            self.truck_respect_s = 5
            self.truck_respect_v = 3

        elif type == "aggressive":
            self.typeEncoding = 3
            self.IDM_s0 = 2
            self.IDM_T  = 1.5
            self.IDM_amax = 1
            self.IDM_b  = 2
            self.IDM_a_exp = 5

            self.MOBIL_p = 0.3
            self.MOBIL_tresh = 0.5
            self.MOBIL_bsafe = 4
            self.MOBIL_requiredSpace_front = 50
            self.MOBIL_requiredSpace_rear = -30

            self.v0 = self.v0_x + 30/3.6
            self.v = self.v0_x # +  np.random.uniform(-10/3.6,10/3.6)

            # Bonus distance + velocity difference considered for truck
            self.truck_respect_s = 5
            self.truck_respect_v = 1
        else:
            raise TypeError("Not a predefined vehicle behavoir")

        self.dummy_var = 2e4                    # dummy variable when vehicle not found

        # PI controller for steering angle
        self.d_nearP = 15  # 5
        self.d_farP = 100
        self.PI_kf = 20  # 20
        self.PI_kn = 9   # 9
        self.PI_kI = 10  #

        # Initialize near and far point based on current lane
        self.nearP = [self.p[0]+self.d_nearP, self.p[1]]
        self.farP = [self.p[0]+self.d_farP, self.p[1]]
        self.theta_n = 0
        self.theta_f = 0

    def setScenario(self,scenario):
        _,_, self.laneCenters = scenario.getRoad()
        self.laneWidth = 2*self.laneCenters[0]
        self.setLane()
        self.laneTarget = self.lane

    def setLane(self):
        # Sets Lane defined simply on geometry
        if self.p[1] > self.laneWidth:
            self.lane = 1
        elif self.p[1] < 0:
            self.lane = -1
        else:
            self.lane = 0

        # Switches lane when controller reaches target,
        # This avoids instantanius switching between targets
        tol = 0.2 
        if self.p[1] > self.laneCenters[1]:
            self.laneTarget = 1
        elif self.p[1] < self.laneCenters[-1]:
            self.laneTarget = -1
        elif np.abs(self.p[1]-self.laneCenters[0]) < tol:
            self.laneTarget = 0  

    def setLaneTarget(self):
        # Switches lane when controller reaches target,
        # This avoids instantanius switching between targets
        tol = 0.2 
        if self.p[1] > self.laneCenters[1]:
            self.laneTarget = 1
        elif self.p[1] < self.laneCenters[-1]:
            self.laneTarget = -1
        elif np.abs(self.p[1]-self.laneCenters[0]) < tol:
            self.laneTarget = 0

    def model(self,x_init):
        dp_x = x_init[2]
        dp_y = x_init[2] * np.tan(x_init[3])
        dv_x = self.u[1,0]
        dtheta = x_init[2] * np.tan(self.u[0,0]) / (np.cos(x_init[3]) * self.length)

        dX = np.array([dp_x,dp_y,dv_x,dtheta])
        return dX

    def getVeh(self,vehicles,egoVehicle,lane,d,type):
        # Calculates speed.position and acceleration of closest vehicle in lane
        # Either for leading or trailing vehicle 
        # type = {"lead", "trail"}

        # Initialize with same v as ego and "infinitly" far away from current vehicle
        x_veh = self.getState()
        if type == "lead":
            x_veh[0] = self.dummy_var
        elif type == "trail":
            x_veh[0] = -self.dummy_var
        else:
            raise TypeError('Not an existing vehicle relation')

        u_veh = np.zeros((2,1), dtype = float)

        # Loop over other cars
        for vehicle in vehicles:
            state_i = vehicle.getState()

            if (lane == vehicle.getLane()) and (not(float(self.p[0]) == float(state_i[0]))):
                if type == "lead":
                    if (self.p[0] < state_i[0]) and (state_i[0] < (self.p[0]+d)) and (state_i[0] < x_veh[0]):
                        x_veh = state_i
                        u_veh = vehicle.getControl()
                elif type == "trail":
                    if (self.p[0] > state_i[0]) and (state_i[0] > (self.p[0]+d)) and (state_i[0] > x_veh[0]):
                        x_veh = state_i
                        u_veh = vehicle.getControl()

        # Check ego vehicle
        state_ego = egoVehicle.getState()
        state_i = state_ego[:,0]
        if (lane == egoVehicle.getLane()):
                if type == "lead":
                    if (self.p[0] < state_i[0]) and (state_i[0] < (self.p[0]+d)) and (state_i[0] < x_veh[0]):
                        x_veh = state_i[:4]
                        x_veh[0] -= 2*self.truck_respect_s
                        x_veh[2] -= self.truck_respect_v
                        u_veh = egoVehicle.getControl()
                elif type == "trail":
                    if (self.p[0] > state_i[0]) and (state_i[0] > (self.p[0]+d)) and (state_i[0] > x_veh[0]):
                        x_veh = state_i[:4]
                        x_veh[0] += self.truck_respect_s
                        x_veh[2] += self.truck_respect_v
                        u_veh = egoVehicle.getControl()
        return x_veh, u_veh

    def IDM(self,x_init,x_lead):
        s_star = self.IDM_s0 + x_init[2] * self.IDM_T + \
                x_init[2] * np.abs((x_init[2] - x_lead[2])) / (2 * np.sqrt(self.IDM_amax * self.IDM_b))

        a  = self.IDM_amax * ( 1 - (x_init[2]/self.v0_x) ** self.IDM_a_exp -
            ( s_star/ (x_init[0] - x_lead[0]) ) ** 2 )
        return a

    def mobil(self,state,vehicles,egoVehicle):
        # Returns updated near and far point for vehicle to follow
        # The near/far point corresponds to the road center of the desired lane at a near/far distance
        # the near distance is set "in clear view" of the driver
        # the far point correponds to the goal target e.g center of lane or leading vehicle further into the future

        # Remove unwanted lanes
        if self.lane == -1:
            laneChoices = [-1,0]
        elif self.lane == -0:
            laneChoices = [-1,0,1]
        elif self.lane == 1:
            laneChoices = [0,1]

        cost = np.zeros((3,))               # High == good
        for i in laneChoices:
            # Check for potentially harmful vehicles
            x_lead_test,_ = self.getVeh(vehicles,egoVehicle,i,self.MOBIL_requiredSpace_front,"lead") 
            x_trail_test,_ = self.getVeh(vehicles,egoVehicle,i,self.MOBIL_requiredSpace_rear,"trail") 

            if (i == self.lane):
                # Dont atempt an overtake, promote keeping own lane
                # Cost is just shy of threshold to allow for overtakes if ideal
                if not(x_lead_test[0] < self.dummy_var):
                    # If there is a no lead vehicle in the current lane, keep lane
                    cost[i] = self.dummy_var
                else:
                    cost[i] = self.MOBIL_tresh-0.01
            elif (x_lead_test[0] < self.dummy_var) or (self.dummy_var < x_trail_test[0]):
                # There is a vehicle in the target lane, dont switch!
                cost[i] = 0
            else:
                # It is ok to do an overtake
                x_lead_i,u_lead_i = self.getVeh(vehicles,egoVehicle,i,self.d_farP,"lead") 
                x_trail_i,u_trail_i = self.getVeh(vehicles,egoVehicle,i,-self.d_farP,"trail")
                x_lead_current,u_lead_current = self.getVeh(vehicles,egoVehicle,self.lane,self.d_farP,"lead")
                x_trail_current,u_trail_current = self.getVeh(vehicles,egoVehicle,self.lane,-self.d_farP,"trail")

                x_init_new = state.tolist()
                x_init_new[1] = self.laneCenters[i]

                x_assumed_ahead_of_ego = x_lead_test
                x_assumed_ahead_of_ego[0] = x_lead_test[0] + self.d_farP

                a_ego_currentLane = self.u[1,0]
                a_ego_newLane = self.IDM(x_init_new,x_lead_i)
                # print(a_ego_currentLane,a_ego_newLane)

                a_trail_currentLane = u_trail_current[1]
                a_trail_currentLaneNew = self.IDM(x_trail_current,x_lead_current)
                # print(a_trail_currentLane,a_trail_currentLaneNew)

                a_trail_newLane = u_trail_i[1]
                a_trail_newLaneNew = self.IDM(x_trail_i,x_init_new)
                # print(a_trail_newLane,a_trail_newLaneNew)

                thresh_value = (a_ego_newLane - a_ego_currentLane) + \
                            + self.MOBIL_p * ((a_trail_currentLaneNew + a_trail_currentLane) +
                             (a_trail_newLaneNew + a_trail_newLane))
                # print("MOBIL result",thresh_value)
                if self.MOBIL_tresh < thresh_value:
                    cost[i] = thresh_value

        # Calcualted near and far point for opimal lane
        optLane = np.argmax(cost)
        x_lead_opt,_ = self.getVeh(vehicles,egoVehicle,optLane,self.d_farP,"lead")
        x_veh = state.tolist()

        # nearP = [x_veh[0] + (self.length + self.d_nearP)*np.cos(x_veh[3]) ,
                # x_veh[1] + (self.length + self.d_nearP)*np.sin(x_veh[3]) ]

        nearP = [x_veh[0] + (self.length +self.d_nearP)*cos(x_veh[3]),self.laneCenters[optLane]]

        if x_lead_opt[0] < x_veh[0] + self.d_farP:
            x_far = x_lead_opt[0]
        else:
            x_far = x_veh[0] + self.length + self.d_farP

        farP = [x_far,self.laneCenters[optLane]]


        return nearP, farP

    def PI_steer(self,state,nearP,farP):
        x_A = state[0] + self.length * np.cos(state[3])
        y_A = state[1] + self.length * np.sin(state[3])

        theta_n = np.arctan((nearP[1] - y_A )/ (nearP[0] - x_A) ) - state[3]
        theta_f = np.arctan((farP[1] - y_A )/ (farP[0] - x_A) ) - state[3]

        theta_new = self.PI_kf*(theta_f-self.theta_f) + \
                    self.PI_kn*(theta_n-self.theta_n) + \
                    self.dt * self.PI_kI*self.theta_n + self.u[0,0]

        # print("Steer angle (degrees)",theta_new)
        # Update stored values
        self.theta_f = theta_f
        self.theta_n = theta_n
        
        #constrain theta? (Shouldnt be necessary!)

        return np.pi*theta_new/180

    def move(self,x_init):
        # Takes concatenated np array with initial states
        p =  [x_init[0] + self.dt * self.v,self.p[1]]

        return np.array(np.append(np.append(p, self.v),self.theta))

    def prediction(self,N_pred):
        # Makes prediction of traffic movement over horizon
        self.state = np.zeros((self.nx,N_pred+1))
        self.state[:,0] = np.array(np.append(np.append(self.p, self.v),self.theta))
        for i in range(1,N_pred+1):
            self.state[:,i] = self.move(self.state[:,i-1])

        return self.state

    def setUpdate(self,vehicles, egoVehicle):
        # Main update function for the current traffic state
        # print("State Update")
        state = self.getState()                     # (unneccesary?)
        x_lead,_ = self.getVeh(vehicles,egoVehicle,self.lane,self.d_farP,"lead")                     # (gets lead and trailingvehicle based on goal+safety)
        self.nearP, self.farP = self.mobil(state,vehicles, egoVehicle)        # (gets goals for controllers)
        # print(self.nearP,self.farP)
        self.u[0,0] = self.PI_steer(state,self.nearP,self.farP)                 # (gets steer angle)

        self.u[1,0] = self.IDM(state,x_lead)          # (gets appropriate acceleration) (can mby be based on current lane?)
        dX = self.model(state)

        self.state_next = state + self.dt*dX

    def pushUpdate(self):
        p_next = self.state_next[:2]
        v_next = self.state_next[2]
        theta_next  = self.state_next[3]
        
        self.p = p_next.tolist()
        self.v = v_next
        self.theta = theta_next

        self.setLane()

    def getState(self):
        # Returns current state of the vehicle
        return np.array(np.append(np.append(self.p,self.v),self.theta)).transpose()

    def getFeatures(self):
        # Returns current state of the vehicle
        return np.array(np.append(np.append(np.append(self.p,self.v),self.theta),self.typeEncoding)).transpose()

    def getReference(self):
        return np.array(np.append(self.nearP,self.farP)).T

    def getControl(self):
        return self.u[:,0]

    def getSize(self):
        # Returns size of vehicle
        return self.width, self.length

    def getDim(self):
        return self.nx, self.N

    def getLane(self):
            return self.lane

    def respawn(self,respawnPx,respawnVx):
        # Respawns the vehicle at given position and velocity
        self.p[0] = respawnPx
        self.v = self.v + respawnVx
        self.theta = 0

        self.nearP = [self.p[0]+self.d_nearP, self.p[1]]
        self.farP = [self.p[0]+self.d_farP, self.p[1]]
        self.theta_n = 0
        self.theta_f = 0
        self.u = np.zeros((2,self.N),dtype= float)

        pass