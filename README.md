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

**Vehicle Model:**  
The vehicle model is shown below:  
<!-- ![Vehicle Model](Figure/kinematic_model_cog.png) -->
<p align="center">
  <img src="Figure\kinematic_model_cog.png" alt="kinematic_model_cog" width="50%">
</p>

**Trajectory Propagation:**  
The difference between the simulated vehicle in CARLA and our nominal model is treated as noise. The figure illustrates how this discrepancy propagates over time.  
![Propagation of Trajectory](Figure/propagation_of_trajectory.png)


**Constraint Definitions:**  
<p align="center">
  <img src="Figure\Constraint.gif" alt="Constraints Defination" width="100%">
</p>

For detailed constraint definitions, please refer to our supervisor's paper:  

```
E. Börve, N. Murgovski, and L. Laine, "Interaction-Aware Trajectory Prediction and Planning in Dense Highway Traffic using Distributed Model Predictive Control."
```

If you find the details on constraint definitions helpful or if they've sparked some ideas for your own work, we'd really appreciate it if you could cite our supervisor's paper.



**Illustration of the coordinate system** 
![ Illustration of the coordinate system](Figure/Illistration_of_coordinate_system.png)

**SMPC Theory:**  
![MPC Constraint Theory](Figure/smpc_theory.png)
**SMPC Constraint Tightening:**  
To address these issues, we employ Stochastic MPC (SMPC) techniques to tighten state constraints, especially for trailing and lane changing maneuvers.  
<!-- ![MPC Constraint Tightening](Figure/MPC_tighten_bound.jpg)
![MPC Constraint Tightening_ACC](Figure/Tightened_trailing.png)
![MPC Constraint Tightening_OverTaking](Figure/Tightened_overtake.png) -->  
**(a)**
![MPC Constraint Comparison](Figure/IDM_compare.png)

**(b)**
![MPC Constraint Comparison](Figure/RC_compare.png)

**(c)**
![MPC Constraint Comparison](Figure/LC_compare.png)

**Figure(a),(b),(c) are the initial constraints  and the constraints after the SMPC**   



![MPC Constraint Tightening](Figure/Tightened_constraints.png)



## Simulation in the CARLA Environment

<!-- **Decision Making Process:**  
<p align="center">
  <img src="Figure/decision_master.gif" alt="Decision Making Process" width="100%">
</p> -->

**Simulation in CARLA:**  

```
Collision Avoidance Success Rate:  99/100 
(Simulate in CARLA with 100 experiments of randomly generated environment)
```

**Complex Scenerio Simulation in CARLA:**  

<p align="center">
  <img src="Figure/complex_scenerio.png" alt="Complex Scenerio" width="100%">
</p>

<p align="center">
  <img src="Figure/CARLA_SIM.gif" alt="Decision Making Process" width="100%">
</p>



**Step 10:**

- **Controller Cost**: 
  - Base Cost: 931.918
  - Intermediate Cost: 3585.441
  - Final Cost: 4.245
- **Slack**: 
  - Primary Slack: 3.746e-17
  - Secondary Slack: 1.309
  - Tertiary Slack: 1.433e-18
- **Decision**: Trailing

<p align="center">
  <img src="Figure/Combined_Trajectories_10.png" alt="Combined_Trajectories_10.png" width="100%">
</p>

**Step 20:**

- **Controller Cost**: 
  - Base Cost: 929.150
  - Intermediate Cost: 6136.020
  - Final Cost: 33261.869
- **Slack**: 
  - Primary Slack: 3.751e-17
  - Secondary Slack: 12.063
  - Tertiary Slack: 59.889
- **Decision**: Left Change

<p align="center">
  <img src="Figure/Combined_Trajectories_20.png" alt="Combined_Trajectories_20.png" width="100%">
</p>

**Step 30:**

- **Controller Cost**: 
  - Base Cost: 926.382
  - Intermediate Cost: 8686.598
  - Final Cost: 66419.494
- **Slack**: 
  - Primary Slack: 3.756e-17
  - Secondary Slack: 22.817
  - Tertiary Slack: 118.349
- **Decision**: Left Change
<p align="center">
  <img src="Figure/Combined_Trajectories_30.png" alt="Combined_Trajectories_30.png" width="100%">
</p>

**...........**



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
  <img src="Figure/CARLA_simulation_compare_ref.gif" alt="Controller Testing in CARLA" width="100%">
</p>

**Vehicle Trajectory:**  
The comparison of trajectory with mpc and smpc.
<p align="center">
  <img src="Figure/velocity_tracking_layout_nosmpc.png" alt="Vehicle Trajectory" width="100%">
</p>
<p align="center">
  <img src="Figure/velocity_tracking_layout.png" alt="Vehicle Trajectory" width="100%">
</p>

<p align="center">
<img src="Figure/acceleration_profile_all_nosmpc.png" alt="Vehicle Trajectory" width="49%"> 
<img src="Figure/acceleration_profile_all.png" alt="acceleration" width="49%">
  
  
</p>

