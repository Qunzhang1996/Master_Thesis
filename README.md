# MSc Thesis: Robust MPC for Autonomous Vehicles in Uncertain Situations

**Authors:** Qun Zhang, Saeed Salih, Erik Börve   
**Emails:** [qunz@chalmers.se](mailto:qunz@chalmers.se), [saeedsal@chalmers.se](mailto:saeedsal@chalmers.se), [borerik@chalmers.se](mailto:borerik@chalmers.se)  
**Affiliation:** Department of Electrical Engineering, Chalmers University of Technology, Göteborg, Sweden

## Purpose

The objectives of this thesis are to:

- Construct safety-critical scenarios for a heavy vehicle in the CARLA simulator, emphasizing realistic challenges in autonomous driving.
- Design and implement a Robust Model Predictive Controller (RMPC) that accounts for uncertainties in the ego-vehicle's state and dynamics, ensuring safety and reliability.
- Extend the RMPC to effectively handle uncertainties related to the surrounding vehicles, improving situational awareness and decision-making.
<p align="center">
  <img src="Figure/scenerio.png" alt="Scenerio Diagram" width="50%">
</p>

## Workflow

Our workflow integrates an Extended Kalman Filter (EKF) with a Model Predictive Controller (MPC) for enhanced accuracy and robustness, depicted in the figures below. Notably, we simulate sensor inputs rather than using actual CARLA sensors to streamline our process.

**Workflow Overview:**  
![Work Flow Diagram](Figure/Work_flow.png)

**Trajectory Propagation:**  
The difference between the simulated vehicle in CARLA and our nominal model is treated as noise. The figure illustrates how this discrepancy propagates over time.  
![Propagation of Trajectory](Figure/propagation_of_trajectory.png)


**Constraint Definitions:**  
For detailed constraint definitions, please refer to our supervisor's paper:  
<p align="center">
  <img src="Figure\Constraints.png" alt="Constraints Defination" width="100%">
</p>

*E. Börve, N. Murgovski, and L. Laine, "Interaction-Aware Trajectory Prediction and Planning in Dense Highway Traffic using Distributed Model Predictive Control."*

If you find the details on constraint definitions helpful or if they've sparked some ideas for your own work, we'd really appreciate it if you could cite our supervisor's paper.

**SMPC Constraint Tightening:**  
To address these issues, we employ Stochastic MPC (SMPC) techniques to tighten state constraints, especially for trailing and lane changing maneuvers.  
![MPC Constraint Tightening](Figure/MPC_tighten_bound.jpg)
![MPC Constraint Tightening_ACC](Figure/Tightened_IDM.png)
![MPC Constraint Tightening_OverTaking](Figure/Tightened_tanh.png)


## Simulation in the CARLA Environment

We showcase our RMPC's performance in various driving scenarios within the CARLA simulator. Below are GIFs depicting different aspects of the driving scenario, including Adaptive Cruise Control (ACC) in heavy traffic, lane changing with MPC_PID control, decision-making processes.

**Adaptive Cruise Control in Heavy Traffic:**  
<p align="center">
  <img src="Figure/ACC_in_heavy_traffic.gif" alt="ACC in Heavy Traffic" width="100%">
</p>

**MPC_PID Lane Changing:**  
<p align="center">
  <img src="Figure/MPC_PID_LC%20(2).gif" alt="MPC_PID Lane Changing" width="100%">
</p>

**Decision Making Process:**  
<p align="center">
  <img src="Figure/decision_master.gif" alt="Decision Making Process" width="100%">
</p>

**Driving in Heavy Traffic Conditions:**  
<p align="center">
  <img src="Figure/crazy_traffic_mix3.gif" alt="Heavy Traffic Conditions" width="100%">
</p>

**Driving in Heavy Traffic Conditions using EKF:**  
<p align="center">
  <img src="Figure/crazy_traffic_mix3_EKF.gif" alt="Controller Testing in CARLA" width="100%">
</p>




**Controller Testing in CARLA:**  
<p align="center">
  <img src="Figure/CARLA_simulationn_Make_Controller_TEST_ref.png" alt="Controller Testing in CARLA" width="100%">
</p>



**Vehicle Trajectory:**  
The trajectory followed by the vehicle during the tests.
<p align="center">
  <img src="Figure/CARLA_simulation_Make_Controller_TEST.png" alt="Vehicle Trajectory" width="100%">
</p>
