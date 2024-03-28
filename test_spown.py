import sys
path_to_add='C:\\Users\\A490243\\Desktop\\Master_Thesis'
sys.path.append(path_to_add)
from Autonomous_Truck_Sim.helpers import *
from Controller.MPC_tighten_bound import MPC_tighten_bound
from Controller.Controllers import makeController
from vehicleModel.vehicle_model import car_VehicleModel
from Traffic.Traffic import Traffic
from Traffic.Scenarios import trailing, simpleOvertake
from util.utils import *

print(1+1)

# ------------------------change map to Town06------------------------
import subprocess
# Command to run your script
command = (
    r'cd C:\Users\A490243\CARLA\CARLA_Latest\WindowsNoEditor\PythonAPI\util && '
    r'python config.py -m Town06')
subprocess.run(command, shell=True)
exit()
# Run the command