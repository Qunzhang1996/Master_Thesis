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
        self.init_bound = 143.318146-laneWidth/2

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

        safeDist = self.min_distx + leadLength/2 + self.L_tract
        return Function('S',[self.p],[self.p-safeDist],['p'],['D_min'])


    def getRoad(self):
        roadMax = 2*self.laneWidth + self.init_bound
        roadMin = -(self.lanes-2)*self.laneWidth  +self.init_bound
        laneCenters = [self.init_bound+self.laneWidth/2,self.init_bound+self.laneWidth*3/2,self.init_bound-self.laneWidth*1/2]

        return roadMin, roadMax, laneCenters

    def getVmax(self):
        return self.vmax
    
    def getEgoLane(self):
        return self.egoLane
    
    def setEgoLane(self):
        x = self.vehicle.getPosition()
        self.egoPx = x[0]
        self.egoPy = x[1]
        if self.egoPy < self.init_bound:
            self.egoLane = -1
        elif self.egoPy > self.init_bound+self.laneWidth:
            self.egoLane = 1
        else:
            self.egoLane = 0
            
    def getLeadVehicle(self,traffic):
        """
        Find the lead vehicle in the same lane with the ego vehicle, the cloest one in front of the ego vehicle
        """
        self.vehicle_list = traffic.getVehicles()
        self.setEgoLane()
        i = 0
        reldistance = 10000             # A large number
        leadInLane = []
        for vehicle in self.vehicle_list:
            if self.egoLane == vehicle.getLane():
                if vehicle.getPosition()[0] > self.egoPx:
                    distance = vehicle.getPosition()[0] - self.egoPx
                    if distance < reldistance:
                        leadInLane = [i]
                        reldistance = distance
            i += 1
        return leadInLane
    
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
    def __init__(self,vehicle,N, min_distx = 5, lanes = 3, laneWidth = 6.5,v_legal = 60/3.6):
        self.name = 'simpleOvertake'
        self.N = N
        self.nx,self.nu,_,_ = vehicle.getSystemDim()

        # Road definitions
        self.lanes = lanes
        self.laneWidth = laneWidth

        self.vmax = v_legal+5/3.6
        self.init_bound = 143.318146-laneWidth/2

        self.Time_headway = 0.5

        # Ego vehicle dimensions
        self.egoWidth, self.egoLength,self.L_tract, self.L_trail = vehicle.getSize()
        
        # Safety constraint definitions
        self.min_distx = min_distx
        self.pxL = MX.sym('pxL',1,N+1)
        self.px = MX.sym('x',1,N+1)

        self.traffic_x = MX.sym('x',1,N+1)
        self.traffic_y = MX.sym('y',1,N+1)
        self.traffic_sign = MX.sym('sign',1,N+1)
        self.traffic_shift = MX.sym('shift',1,N+1)


    def getReference(self,refx,refu):
        # Returns state and input reference for all steps in horizon 
        refx_out = DM(self.nx,self.N+1)
        refu_out = DM(self.nu,self.N)
        
        for i in range(self.nx):
            refx_out[i,:] = refx[i]
        for i in range(self.nu):
            refu_out[i,:] = refu[i]

        return refx_out,refu_out

    def getRoad(self):
        roadMax = 2*self.laneWidth + self.init_bound
        roadMin = -(self.lanes-2)*self.laneWidth  +self.init_bound
        laneCenters = [self.init_bound+self.laneWidth/2,self.init_bound+self.laneWidth*3/2,self.init_bound-self.laneWidth*1/2]

        return roadMin, roadMax, laneCenters

    def getVmax(self):
        return self.vmax
    
    def constraint(self,traffic,opts):
        constraints = []
        leadWidth, leadLength = traffic.getVehicles()[0].getSize()
        for i in range(traffic.getDim()):
            v0_i = traffic.vehicles[i].v0
            func1 = self.traffic_sign * (self.traffic_sign*(self.traffic_y-self.traffic_shift)/2 + self.egoWidth + leadWidth) / 2 * \
                    tanh(self.px - self.traffic_x + leadLength/2 + self.L_tract + v0_i * self.Time_headway + self.min_distx )  + self.traffic_shift/2
            func2 = self.traffic_sign * (self.traffic_sign*(self.traffic_y-self.traffic_shift)/2 + self.egoWidth + leadWidth) / 2 * \
                    tanh( - (self.px - self.traffic_x)  + leadLength/2 + self.L_trail + v0_i * self.Time_headway+ self.min_distx )  + self.traffic_shift/2

            constraints.append(Function('S',[self.px,self.traffic_x,self.traffic_y,
                                    self.traffic_sign,self.traffic_shift,],
                                    [func1+func2],['px','t_x','t_y','t_sign','t_shift'],['y_cons']))
        return constraints