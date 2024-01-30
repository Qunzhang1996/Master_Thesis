## change map to Town06
import subprocess
# Command to run your script
command = (
    r'cd C:\Users\A490243\CARLA\CARLA_Latest\WindowsNoEditor\PythonAPI\util && '
    r'python config.py --map Town06')
# Run the command
subprocess.run(command, shell=True)
#! This file is useed to test the acc controller
from casadi import *