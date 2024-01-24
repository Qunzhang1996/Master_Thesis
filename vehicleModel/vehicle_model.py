# Vehicle model setup for casadi interface
#! include bicycle model for car and truck
from casadi import*
import sys
path_to_add='C:\\Users\\A490243\\Desktop\\Master_Thesis'
sys.path.append(path_to_add)
from Autonomous_Truck_Sim.vehicleModelGarage import vehBicycleKinematic

class car_VehicleModel(vehBicycleKinematic):
    """Here, inheriting from vehBicycleKinematic class

    Kinematic bicycle model with trailer
    x = [p_x p_y v_x theta]
    u = [steer_ang, acc_v,x]
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "car_bicycle"
        self.nx = 4                     # State dimensions
        
    #! rewrite the model from vehBicycleKinematic to car_VehicleModel
    def model(self):
        dp_x

car_VehicleModel = car_VehicleModel(0.1, 10)
print(car_VehicleModel.name)