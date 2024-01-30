# here is the simple MPC controller for the trailing vehicle
# this file is used to test the acc controller
import sys
CARLA_PATH = 'C:\\Users\\A490243\\CARLA\\CARLA_Latest\\WindowsNoEditor\\PythonAPI\\carla'
THESIS_PATH = 'C:\\Users\\A490243\\Desktop\\Master_Thesis'
sys.path.extend([CARLA_PATH, THESIS_PATH])
from casadi import *
from  Autonomous_Truck_Sim.controllers import makeController, makeDecisionMaster
from  Autonomous_Truck_Sim.scenarios import trailing
from vehicleModel.vehicle_model import car_VehicleModel
class trailing_controller:
    """here is the simple MPC controller for the trailing vehicle
    vehicle: the vehicle model class
    N: the prediction horizon
    opts: the options for the controller
        {"version" : "trailing", "solver": "ipopt", "integrator":"rk"}
    dt: the time step for the controller
    """
    def __init__(self,vehicle,N,opts,dt) -> None:
        self.vehicle = vehicle
        self.opts = opts
        # Get constants
        # self.Nveh = self.traffic.getDim()
        self.N = N
        self.vehWidth,_,_,_ = self.vehicle.getSize()
        self.nx,self.nu,self.nrefx,self.nrefu = self.vehicle.getSystemDim()
        print(self.nx,self.nu,self.nrefx,self.nrefu)
        # self.roadMin, self.roadMax, self.laneCenters = self.scenario.getRoad()
        self.opts = opts
        #create Opti Stack
        self.opti = Opti()
        pass
# System initialization 
dt = 0.2                    # Simulation time step (Impacts traffic model accuracy)
f_controller = 5            # Controller update frequency, i.e updates at each t = dt*f_controller
N =  12                     # MPC Horizon length
dt_MPC = dt*f_controller    # Controller time step
# ----------------- Ego Vehicle Dynamics and Controller Settings ------------------------
vehicleADV = car_VehicleModel(dt,N)

vehWidth,vehLength,L_tract,L_trail = vehicleADV.getSize()
nx,nu,nrefx,nrefu = vehicleADV.getSystemDim()

# Integrator
int_opt = 'rk'
vehicleADV.integrator(int_opt,dt)
F_x_ADV  = vehicleADV.getIntegrator()
opts3 = {"version" : "trailing", "solver": "ipopt", "integrator":"rk"}
MPC_trailing = trailing_controller(vehicleADV,N,opts3,dt_MPC)