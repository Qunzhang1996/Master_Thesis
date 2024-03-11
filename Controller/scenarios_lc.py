"""Class for lane change scenario
"""
import numpy as np
from casadi import *

class simpleLaneChange: 
    '''
    The ego vehicle overtakes the lead vehicle
    '''
    def __init__(self,N, Ntraffic, min_distx = 5, lanes = 3, laneWidth = 3.5):
        self.N = N
        self.name = 'simpleOvertake'
        self.Ntraffic = Ntraffic

        # Road definitions
        self.lanes = lanes
        self.laneWidth = laneWidth

        self.vmax = (60+5)/3.6

        self.Time_headway = 0.5
        self.L_tract=6
        self.L_trail=0
        self.egoWidth = 2.54
        
        # Safety constraint definitions
        self.min_distx = min_distx
        self.pxL = MX.sym('pxL',1,N+1)
        self.px = MX.sym('x',1,N+1)

        self.traffic_x = MX.sym('x',1,N+1)
        self.traffic_y = MX.sym('y',1,N+1)
        self.traffic_sign = MX.sym('sign',1,N+1)
        self.traffic_shift = MX.sym('shift',1,N+1)



        
    def constraint(self):
        leadLength = 6
        v0_i = 15
        leadWidth = 1.7
        constraints = []
        for i in range(self.Ntraffic):
            func1 = self.traffic_sign * (self.traffic_sign*(self.traffic_y-self.traffic_shift)/2 + self.egoWidth + leadWidth) / 2 * \
                        tanh(self.px - self.traffic_x + leadLength/2 + self.L_tract + v0_i * self.Time_headway + self.min_distx )  + self.traffic_shift/2 
            func2 = self.traffic_sign * (self.traffic_sign*(self.traffic_y-self.traffic_shift)/2 + self.egoWidth + leadWidth) / 2 * \
                        tanh( - (self.px - self.traffic_x)  + leadLength/2 + self.L_trail + v0_i * self.Time_headway+ self.min_distx )  + self.traffic_shift/2
            
            constraints.append(Function('S',[self.px,self.traffic_x,self.traffic_y,
                                    self.traffic_sign,self.traffic_shift,],
                                    [func1+func2 + 143.318146 -3.5/2],['px','t_x','t_y','t_sign','t_shift'],['y_cons']))
            # constraints.append(func1+func2 + 143.318146 -3.5/2)
        return constraints