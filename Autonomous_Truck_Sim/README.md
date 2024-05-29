# Autonomous Truck Simulator
Author: Erik Börve, borerik@chalmers.se  

 ## Purpose
 This project provides an implementation of an autonomous truck in a multi-lane highway scenario. The controller utilizes non linear optimal control to compute multiple feasible trajectories, of which the most cost-efficent is choosen.
 
 ![](https://github.com/BorveErik/Autonomous-Truck-Sim/blob/main/simRes.gif)

 ## Getting Started

 ### Prerequisites

 Clone the project in your local machine.

Locate the repo and run:
* pip
  ```sh
  pip install -r requirements.txt
  ```


## Usage
Simulations are run via the "main" file. This is also where simulations are configured, including e.g., designing traffic scenarios and setting up the optimal controllers.

 ## Project Structure
The projects contains the following files.
```bash
.
├── controllers.py
├── Data_example
    └── ex1.csv
    └── metaData_ex1.txt
├── gitignore
├── helpers.py
├── main.py
├── README.md
├── requirements.txt
├── scenarios.py
├── simRes.gif
├── templateRLagent.py
├── traffic.py
└── vehicleModelGarage.py

```

 ### controllers.py
 makeController:
 Generates optimal controller based on specified scenario.
 
 decisionMaster:
 Optimizes the trajectory choice, returns optimal policy.
 ### helpers.py
 Contains assisting functions, e.g., for data extraction and plotting.
 ### main.py
 Running and setting up simulations.
 
 ### scenarios.py
 Formulates constraints for the different scenarios considered in the optimal controllers.
 
 ### templateRLagent.py
 Template class for communicating with the lower level controllers.
 
 ### traffic.py
 Combined traffic: Used to communicate with all vehicles in traffic scenario.
 
 vehicleSUMO: Creates a vehicle with specified starting position, velocity and class.
 
 ### vehicleModelGarage.py
 Contains truck models that can be utilized in the simulation.
