from casadi import *
import numpy as np
import matplotlib.pyplot as plt

leadLength = 6
N = 12
v0_i = 15
leadWidth = 1.7
laneWidth = 3.5


class test:
    def __init__(self, traffic_x, traffic_y, traffic_shift,traffic_sign, min_distx=5) -> None:
        self.traffic_sign = traffic_sign
        self.traffic_x = traffic_x
        self.traffic_y = traffic_y
        self.traffic_shift = traffic_shift
        self.Time_headway = 0.5
        self.min_distx = min_distx
        self.L_tract=16.1544/3
        self.L_trail=16.1544-self.L_tract
        self.egoWidth = 2.54

    def constraint(self, px):
        func1 = self.traffic_sign * (self.traffic_sign*(self.traffic_y-self.traffic_shift)/2 + self.egoWidth + leadWidth) / 2 * \
                    tanh(px - self.traffic_x + leadLength/2 + self.L_tract + v0_i * self.Time_headway + self.min_distx )  + self.traffic_shift/2
        func2 = self.traffic_sign * (self.traffic_sign*(self.traffic_y-self.traffic_shift)/2 + self.egoWidth + leadWidth) / 2 * \
                    tanh( - (px - self.traffic_x)  + leadLength/2 + self.L_trail + v0_i * self.Time_headway+ self.min_distx )  + self.traffic_shift/2
        # print(func1 + func2)
        # exit()
        return func1 + func2

# Create instances for traffic_x, traffic_y, and traffic_shift
traffic_x = np.array([10*i for i in range(30)]).reshape(-1,1)
traffic_x_2 = np.array([10*i+60 for i in range(30)]).reshape(-1,1)
traffic_shift =  -laneWidth 
# traffic_shift = 0*traffic_shift
# Create an instance of the test class

my_test = test(traffic_x, traffic_y=-laneWidth/2, traffic_shift=-laneWidth, traffic_sign=1)
my_test_2 = test(traffic_x_2, traffic_y=laneWidth/2, traffic_shift=laneWidth,traffic_sign=-1)

# Time points at which to evaluate the constraint
time_points = np.linspace(-30, 100, 100)  # Example time points

# Evaluate the constraint function at each time point
constraint_values = [float(my_test.constraint(px)[0]) for px in time_points]
constraint_values_2 = [float(my_test_2.constraint(px)[0]) for px in time_points]
# Plot the constraint curve
plt.figure(figsize=(12,4))
plt.plot(time_points, constraint_values,'r', linewidth=2)
plt.plot(time_points, constraint_values_2,'r', linewidth=2)
plt.plot([-30,100],[0,0],'k--')
plt.plot([-30,100],[laneWidth,laneWidth],'k--')
plt.plot([-30,100],[-laneWidth,-laneWidth],'k--')
plt.plot([-30,100],[2*laneWidth,2*laneWidth],'k--')
plt.plot([-30,100],[laneWidth/2,laneWidth/2],'b--')
plt.plot([-30,100],[-laneWidth/2,-laneWidth/2],'b--')
plt.xlabel('Time')
plt.ylabel('Constraint Value')
plt.title('Constraint Curve')
plt.grid(True)
plt.show()
