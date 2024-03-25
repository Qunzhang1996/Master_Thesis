# Different traffic situations
import numpy as np
from casadi import *



class trailing:
    '''
    The ego vehicle keeps lane and adapts to leadning vehicle speed
    '''
    def __init__(self,vehicle,N,min_distx = 5, lanes = 3, laneWidth = 3.5,v_legal = 60/3.6):
        self.name = 'trailing'
        self.N = N
        self.vehicle = vehicle
        self.nx,self.nu,_,_ = vehicle.getSystemDim()
        self.egoWidth, self.egoLength,self.L_tract, self.L_trail = vehicle.getSize()
        # Road definitions
        self.lanes = lanes
        self.laneWidth = laneWidth

        self.vmax = v_legal+5/3.6

        self.Time_headway = 0.5
        self.init_bound = self.vehicle.getInitBound()

        self.min_distx = min_distx
        self.p = MX.sym('p',1,N+1)

        
        #! slack variable for the IDM constraint
        self.traffic_slack = MX.sym('slack',1,N+1)
        self.egoPx = []
        self.egoPy = []

    def getReference(self,refx,refu):
        # Returns state and input reference for all steps in horizon 
        refx_out = DM(self.nx,self.N+1)
        refu_out = DM(self.nu,self.N)
        
        for i in range(self.nx):
            refx_out[i,:] = refx[i]
        for i in range(self.nu):
            refu_out[i,:] = refu[i]

        return refx_out, refu_out

    # ! DEFINE THE CONSTRAINTS FOR THE TRAILING 
    # ! NEED TO CHANGE, ADD SLACK VARIABLE!!!!!!!!!
    def constraint(self,traffic,opts):
        leadWidth, leadLength = traffic.getVehicles()[0].getSize()
        idx = self.getLeadVehicle(traffic)
        if len(idx) == 0:
            dist_t = 0
        else:
            v0_idx = traffic.getVehicles()[idx[0]].v0
            dist_t = v0_idx * self.Time_headway

        safeDist = self.min_distx + leadLength + self.L_tract
        return Function('S',[self.p],[self.p-safeDist],['p'],['D_min'])
    
    def constrain_tightened(self,traffic,opts,temp_x, tempt_y):
        leadWidth, leadLength = traffic.getVehicles()[0].getSize()
        idx = self.getLeadVehicle(traffic)
        if len(idx) == 0:
            dist_t = 0
        else:
            v0_idx = traffic.getVehicles()[idx[0]].v0
            dist_t = v0_idx * self.Time_headway

        safeDist = self.min_distx + leadLength + self.L_tract + temp_x
        return Function('S',[self.p],[self.p-safeDist],['p'],['D_min'])

    def getRoad(self):
        roadMax = 2*self.laneWidth + self.init_bound
        roadMin = -(self.lanes-2)*self.laneWidth  +self.init_bound
        laneCenters = [self.init_bound+self.laneWidth/2,self.init_bound+self.laneWidth*3/2,self.init_bound-self.laneWidth*1/2]

        laneWidth = self.laneWidth
        return roadMin, roadMax, laneCenters, laneWidth

    def getVmax(self):
        return self.vmax
    
    def getEgoLane(self):
        return self.egoLane
    
    def setEgoLane(self,traffic):
        x = traffic.getStates()[:2,1]
        # print(f"this is the ego position {x[0]} and {x[1]}")
        self.egoPx = x[0]
        self.egoPy = x[1]
        # print(f"this is the ego position {self.egoPx} and {self.egoPy}")
        if self.egoPy < self.init_bound:
            self.egoLane = -1
        elif self.egoPy > self.init_bound+self.laneWidth:
            self.egoLane = 1
        else:
            self.egoLane = 0
        print("INFO: Current Lane is:", [self.egoLane])
            

    def getLeadVehicle(self, traffic):
        """
        Find the lead vehicle in the same lane as the ego vehicle, the closest one in front of the ego vehicle.
        """
        self.vehicle_list = traffic.getVehicles()
        self.setEgoLane(traffic)
        closestDistance = 10000 # Use infinity as the initial comparison value
        leadVehicleIdx = []  # Initialize with None to indicate no vehicle is found initially

        for idx, vehicle in enumerate(self.vehicle_list):
            # since the get_state have some error, so the self.egoPx here should be ascope variable
            if self.egoLane == vehicle.getLane() and vehicle.getPosition()[0] > self.egoPx:
                # print(f"vehicle {idx} is in lane {vehicle.getLane()}")
                distance = vehicle.getPosition()[0] - self.egoPx
                if distance < closestDistance:
                    closestDistance = distance
                    leadVehicleIdx = [idx]  # Store the index of the closest vehicle
        
        print("INFO: Lead vehicle index is: ", leadVehicleIdx)
        return leadVehicleIdx

    def slackCost(self,q):
        slack_cost = q*dot(self.traffic_slack,self.traffic_slack)
        
        self.Ls = Function('Ls',[self.traffic_slack], [slack_cost],['slack'],['slackCost'])
        pass
        
    def getSlackCost(self):
        return self.Ls
    
class simpleOvertake:
    '''
    The ego vehicle overtakes the lead vehicle
    '''
    def __init__(self,vehicle,N, min_distx = 5, lanes = 3, laneWidth = 3.5, v_legal = 60/3.6):
        self.name = 'simpleOvertake'
        self.N = N
        self.vehicle = vehicle
        self.nx,self.nu,_,_ = self.vehicle.getSystemDim()
        self.init_bound = self.vehicle.getInitBound()

        # Road definitions
        self.lanes = lanes
        self.laneWidth = laneWidth

        self.vmax = v_legal+5/3.6

        self.Time_headway = 0.5
        self.init_bound = 143.318146-laneWidth/2

        # Ego vehicle dimensions
        self.egoWidth, self.egoLength,self.L_tract, self.L_trail = vehicle.getSize()
        self.egoTheta_max = vehicle.xConstraints()[1][3]  #[1][3] is the index of the heading angle
        self.egoTheta_max= 0  #! In this situation, we do not have egoTheta_max. no trailor
        # Safety constraint definitions
        self.min_distx = min_distx
        self.pxL = MX.sym('pxL',1,N+1)
        self.px = MX.sym('x',1,N+1)

        self.traffic_x = MX.sym('x',1,N+1)
        self.traffic_y = MX.sym('y',1,N+1)
        self.traffic_sign = MX.sym('sign',1,N+1)
        self.traffic_shift = MX.sym('shift',1,N+1)
        self.traffic_slack = MX.sym("slack",1,N+1)
        
        #! turn off/on the stochastic mpc
        self.stochastic_mpc = 0
        


    def getReference(self,refx,refu):
        # Returns state and input reference for all steps in horizon 
        refx_out = DM(self.nx,self.N+1)
        refu_out = DM(self.nu,self.N)
        
        for i in range(self.nx):
            refx_out[i,:] = refx[i]
        for i in range(self.nu):
            refu_out[i,:] = refu[i]

        return refx_out,refu_out

    # def getRoad(self):
    #     roadMax = 2*self.laneWidth
    #     roadMin = -(self.lanes-2)*self.laneWidth
    #     laneCenters = [self.laneWidth/2,self.laneWidth*3/2,-self.laneWidth*1/2]
        
    #     return roadMin, roadMax, laneCenters, self.laneWidth
    def getRoad(self):
        roadMax = 2*self.laneWidth + self.init_bound
        roadMin = -(self.lanes-2)*self.laneWidth  +self.init_bound
        laneCenters = [self.init_bound+self.laneWidth/2,self.init_bound+self.laneWidth*3/2,self.init_bound-self.laneWidth*1/2]
        # print(roadMin, roadMax, laneCenters)
        laneWidth = self.laneWidth
        return roadMin, roadMax, laneCenters, laneWidth

    def constraint(self,traffic,opts):
        constraints = []
        
        #! In this situation, we do not have egoTheta_max. no trailor
        d_lat_spread =  self.L_trail* np.tan(self.egoTheta_max)
        for i in range(traffic.getDim()):
            
            # #! avoid the ego vehicle itself
            # if i == 1 : continue
            # Get Vehicle Properties
            v0_i = traffic.getVehicles()[i].v0
            # traffic.getVehicles
            l_front,l_rear = traffic.getVehicles()[i].getLength()
            leadWidth, _ = traffic.getVehicles()[i].getSize()
            # Define vehicle specific constants
            alpha_0 = self.traffic_sign * (self.traffic_sign*(self.traffic_y-self.traffic_shift)+leadWidth/2)
            alpha_1 = l_rear + self.L_tract/2 + v0_i * self.Time_headway + self.min_distx 
            alpha_2 = l_front + self.L_tract/2+ v0_i * self.Time_headway+ self.min_distx 
            alpha_3 = self.traffic_shift
            d_w_e = (self.egoWidth/2+d_lat_spread)*self.traffic_sign
            # Construct function
            func1 = alpha_0 / 2 * tanh(self.px - self.traffic_x + alpha_1)+alpha_3/2
            func2 = alpha_0 / 2 * tanh(self.traffic_x - self.px + alpha_2)+alpha_3/2
            S = func1+func2 + d_w_e
            # !SHIFT ACCORDING TO THE INIT_BOUND
            S = S + self.init_bound 
            constraints.append(Function('S',[self.px,self.traffic_x,self.traffic_y,
                                    self.traffic_sign,self.traffic_shift,],
                                    [S],['px','t_x','t_y','t_sign','t_shift'],['y_cons']))
        return constraints
    
    def constrain_tightened(self,traffic,opts,temp_x, tempt_y):
        constraints = []
        
        #! In this situation, we do not have egoTheta_max. no trailor
        d_lat_spread =  self.L_trail* np.tan(self.egoTheta_max)
        for i in range(traffic.getDim()):
            
            # #! avoid the ego vehicle itself
            # if i == 1 : continue
            # Get Vehicle Properties
            v0_i = traffic.getVehicles()[i].v0
            # traffic.getVehicles
            l_front,l_rear = traffic.getVehicles()[i].getLength()
            leadWidth, _ = traffic.getVehicles()[i].getSize()
            # Define vehicle specific constants
            alpha_0 = self.traffic_sign * (self.traffic_sign*(self.traffic_y-self.traffic_shift+tempt_y)+leadWidth/2)
            alpha_1 = l_rear + self.L_tract + v0_i * self.Time_headway + self.min_distx  # this is for the end
            alpha_2 = l_front + 0 + v0_i * self.Time_headway+ self.min_distx  # this is for the front
            alpha_3 = self.traffic_shift
            d_w_e = (self.egoWidth/2+d_lat_spread)*self.traffic_sign
            # Construct function
            func1 = alpha_0 / 2 * tanh(self.px - self.traffic_x + alpha_1+temp_x)+alpha_3/2
            func2 = alpha_0 / 2 * tanh(self.traffic_x - self.px + alpha_2+temp_x)+alpha_3/2
            S = func1+func2 + d_w_e
            # !SHIFT ACCORDING TO THE INIT_BOUND
            S = S + self.init_bound 
            constraints.append(Function('S',[self.px,self.traffic_x,self.traffic_y,
                                    self.traffic_sign,self.traffic_shift,],
                                    [S],['px','t_x','t_y','t_sign','t_shift'],['y_cons']))
        return constraints
    
    def getVmax(self):
        return self.vmax
    
    
    def slackCost(self,q):
        slack_cost = q*dot(self.traffic_slack,self.traffic_slack)
        
        self.Ls = Function('Ls',[self.traffic_slack], [slack_cost],['slack'],['slackCost'])

        
    def getSlackCost(self):
        return self.Ls