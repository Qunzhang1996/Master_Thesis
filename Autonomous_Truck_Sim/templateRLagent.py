"""
Placeholder class for communication with code base
Allows the current decision maker to be overriden with your RL input

 - Decision encoding: [0,1,2,NaN] = [left change, right change, no change, let MPC decide]
"""

import numpy as np

# Notes:
# - Feel free to edit this file as appropriate, changing template names requires changes troughout code base

class RLAgent:
    """
    RL agent class:
    Decides the appropriate choice of MPC pathplanner

    Methods:
    - featchVehicleFeatures: Fetches the current vehicle states and classes
    - getDecision: Returns the appropriate trajectory option to decision master

    Variables:
    - vehicleFeatures: Vehicle states at the current time prior to executing the optimal control action
    - decision: Current decision made by the RL agent
    """
    def __init__(self):
        self.vehicleFeatures = []
        self.decision = float('nan')

    def fetchVehicleFeatures(self,features):
        # Fetches the most recent vehicle features, automatically refreshes each simulation step
        self.vehicleFeatures = features[:,0:]

    def getDecision(self):
        # Returns final decision for RL agent
        return self.decision

