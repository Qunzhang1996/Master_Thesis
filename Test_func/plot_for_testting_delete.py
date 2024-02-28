import matplotlib.pyplot as plt
from casadi import *


car_x=124
car_y=143.318146
car2_x = 124 - 30
car2_y = 143.318146 + 3.5
truck_x = 124 - 50
truck_y = 143.318146

# trying to plot the box constraints for the car on the right road - car2
traffic_x = car2_x
traffic_y = car2_y
traffic_sign = 1
traffic_shift = 0
egoWidth = 2
leadWidth = 2
leadLength = 5
v0_i = 10
Time_headway = 0.5
min_distx = 5
L_tract = 2
L_trail = 2
px = 124

def tanh(x):
    return (exp(x)-exp(-x)) / (exp(x)+exp(-x))

func1 = traffic_sign * (traffic_sign*(traffic_y-traffic_shift) + egoWidth + leadWidth) / 2 * \
                        tanh(px - traffic_x + leadLength/2 + L_tract + v0_i * Time_headway + min_distx )  + traffic_shift/2
                
func2 = traffic_sign * (traffic_sign*(traffic_y-traffic_shift) + egoWidth + leadWidth) / 2 * \
                        tanh( - (px - traffic_x)  + leadLength/2 + L_trail + v0_i * Time_headway+ min_distx )  + traffic_shift/2

func = func1 + func2


# Define road parameters
center_lane_x = 124 # center points of the center lane: x
center_lane_y = 143.318146 # center points of the center lane: y
lane_width = 3.5
num_lanes = 3

# Plotting the lanes
fig, ax = plt.subplots()

# Plot left lane
left_lane_y = center_lane_y + lane_width
plt.plot([0, 248], [left_lane_y + lane_width/2, left_lane_y + lane_width/2], color='black', linestyle='-')

# Plot center lane
plt.plot([0, 248], [center_lane_y+ lane_width/2, center_lane_y + lane_width/2], color='black', linestyle='dashed')
plt.plot([0, 248], [center_lane_y - lane_width/2, center_lane_y - lane_width/2], color='black', linestyle='dashed')

# Plot right lane
right_lane_y = center_lane_y - lane_width
plt.plot([0, 248], [right_lane_y - lane_width/2, right_lane_y - lane_width/2], color='black', linestyle='-')

# Set y limits
ax.set_ylim(top=right_lane_y - 2*lane_width, bottom=left_lane_y + 2*lane_width)

# Hide x-axis
ax.xaxis.set_visible(False)

plt.scatter(car_x, car_y, color='red', marker='o')
plt.scatter(car2_x, car2_y, color='blue', marker='o')
plt.scatter(truck_x, truck_y, color='green', marker='o')


# Plot func
x_values = np.linspace(0, 248, 100)  # Adjust the number of points as needed
y_values = np.array(func)
plt.plot(x_values, y_values, color='orange', linestyle='-')

plt.show()
