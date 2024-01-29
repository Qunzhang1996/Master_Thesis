# this file is used to test the acc controller
import sys
CARLA_PATH = 'C:\\Users\\A490243\\CARLA\\CARLA_Latest\\WindowsNoEditor\\PythonAPI\\carla'
THESIS_PATH = 'C:\\Users\\A490243\\Desktop\\Master_Thesis'
sys.path.extend([CARLA_PATH, THESIS_PATH])
from casadi import *
from  Autonomous_Truck_Sim.controllers import makeController, makeDecisionMaster
from  Autonomous_Truck_Sim.scenarios import trailing
from vehicleModel.vehicle_model import car_VehicleModel
from util.utils import check_dimensions_car
class car_trailing(trailing):
    """
    inherit from trailing class and used for car
    The ego vehicle keeps lane and adapts to leadning vehicle speed
    """
    def __init__(self, vehicle, N, min_distx=5, lanes=3, laneWidth=6.5):
        super().__init__(vehicle, N, min_distx, lanes, laneWidth)
        self.name = 'car_trailing'
        
#test the output of the sub class
test_car = car_VehicleModel(dt=0.2,N=10, width = 2, length = 4)
vehWidth,vehLength,_,_ = test_car.getSize()
nx,nu,nrefx,nrefu = test_car.getSystemDim()
check_dimensions_car(nx,nu)
check_dimensions_car(nrefx,nrefu)
# System initialization 
dt = 0.2                    # Simulation time step (Impacts traffic model accuracy)
f_controller = 5            # Controller update frequency, i.e updates at each t = dt*f_controller
N =  12                     # MPC Horizon length
ref_vx = 60/3.6             # Higway speed limit in (m/s)
int_opt = 'rk'
test_car.integrator(int_opt,dt)
F_x_ADV  = test_car.getIntegrator()
# Set Cost parameters
Q_ADV = [0,40,3e2,5,5]                            # State cost, Entries in diagonal matrix
R_ADV = [5,5]                                    # Input cost, Entries in diagonal matrix
q_ADV_decision = 50
test_car.cost(Q_ADV,R_ADV)
test_car.costf(Q_ADV)
L_ADV,Lf_ADV = test_car.getCost()
# print(L_ADV,Lf_ADV)

#! this one is for the trailing 
# opts3 = {"version" : "trailing", "solver": "ipopt", "integrator":"rk"}
# MPC3 = makeController(vehicleADV,traffic,scenarioTrailADV,N,opts3,dt_MPC)
# MPC3.setController()
# trailLead = MPC3.getFunction()