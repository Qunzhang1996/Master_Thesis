# MSc Thesis: Robust MPC for Autonomous Vehicles in Uncertain Situations

**Authors:** Erik Börve, Qun Zhang, Saeed Salih   
**Emails:** [borerik@chalmers.se](mailto:borerik@chalmers.se), [qunz@chalmers.se](mailto:qunz@chalmers.se), [saeedsal@chalmers.se](mailto:saeedsal@chalmers.se)   
**Affiliation:** Department of Electrical Engineering, Chalmers University of Technology, Göteborg, Sweden

## Purpose

The objectives of this thesis are to:

- Construct safety-critical scenarios for a heavy vehicle in the CARLA simulator, emphasizing realistic challenges in autonomous driving.
- Design and implement a Robust Model Predictive Controller (RMPC) that accounts for uncertainties in the ego-vehicle's state and dynamics, ensuring safety and reliability.
- Extend the RMPC to effectively handle uncertainties related to the surrounding vehicles, improving situational awareness and decision-making.

## Workflow

Our workflow integrates an Extended Kalman Filter (EKF) with a Model Predictive Controller (MPC) for enhanced accuracy and robustness, depicted in the figures below. Notably, we simulate sensor inputs rather than using actual CARLA sensors to streamline our process.

**Workflow Overview:**  
![Work Flow Diagram](Figure/Work_flow.png)

**Trajectory Propagation:**  
The difference between the simulated vehicle in CARLA and our nominal model is treated as noise. The figure illustrates how this discrepancy propagates over time.  
![Propagation of Trajectory](Figure/propagation_of_trajectory.png)

**SMPC Constraint Tightening:**  
To address these issues, we employ Stochastic MPC (SMPC) techniques to tighten state constraints, especially for trailing and lane changing maneuvers.  
![MPC Constraint Tightening](Figure/MPC_tighten_bound.jpg)

For detailed constraint definitions, please refer to our supervisor's paper:  
*E. Börve, N. Murgovski, and L. Laine, "Interaction-Aware Trajectory Prediction and Planning in Dense Highway Traffic using Distributed Model Predictive Control."*

## Simulation in the CARLA Environment

We showcase our RMPC's performance in various driving scenarios within the CARLA simulator. Below are GIFs depicting different aspects of the driving scenario, including Adaptive Cruise Control (ACC) in heavy traffic, lane changing with MPC_PID control, decision-making processes.

**Adaptive Cruise Control in Heavy Traffic:**  
![ACC in Heavy Traffic](Figure/ACC_in_heavy_traffic.gif)

**MPC vs. PID Lane Changing:**  
![MPC vs. PID Lane Changing](Figure/MPC_PID_LC%20(2).gif)

**Decision Making Process:**  
![Decision Making Process](Figure/decision_master.gif)

**Driving in Heavy Traffic Conditions:**  
![Heavy Traffic Conditions](Figure/crazy_traffic_mix3.gif)

**Controller Testing in CARLA:**  
![Controller Testing](Figure/CARLA_simulationn_Make_Controller_TEST_ref.png)

**Vehicle Trajectory:**  
The trajectory followed by the vehicle during the tests, highlighting the precision and robustness of our controller.  
![Vehicle Trajectory](Figure/CARLA_simulation_Make_Controller_TEST.png)
