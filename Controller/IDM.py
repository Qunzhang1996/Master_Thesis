# here is the idm model to calculate the speed and position of the vehicle
import numpy as np
class IDM:
    """Here is the Intelligent Driver Model (IDM) for the autonomous vehicle to track the vehicle in front.
    
    """
    def __init__(self, v0, T, a, b, s0, dt):
        '''
        v0: the velocity the vehicle would drive at in free traffic (m/s)
        T: time headway (s)
        a: maximum acceleration (m/s^2)
        b: comfortable deceleration (m/s^2)
        s0: minimum distance (m)
        dt: time step for the simulation (s)
        '''
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.s0 = s0
        self.dt = dt

    def update_speed_position(self, current_speed, current_position, delta_v, gap):
        """
        Update the speed and position of the vehicle using the IDM model.
        """
        # Calculate the acceleration
        
        acceleration = self.a * (1 - (current_speed / self.v0) ** 4 - (self.s0 + current_speed * self.T + 0.5 * current_speed * delta_v / np.sqrt(self.a * self.b)) / gap ** 2)

        # Update the speed and position
        target_speed = current_speed + acceleration * self.dt
        target_position = current_position + current_speed * self.dt + 0.5 * acceleration * self.dt ** 2

        return target_speed, target_position
    
    def kmh_to_ms(self, speed):
        return speed / 3.6
    
    def ms_to_kmh(self, speed): 
        return speed * 3.6