# Master_Thesis
This the repo for master_thesis
## 2024/01/24 QunZhang
Create the repo and add some related folder   

## 2024/01/26 QunZhang
Add kalman filter and test it in CARLA simulation, and tighten the constraint as shown below: 
<p float="left">
  <img src="Figure\CARLA_KF_TEST.gif" width="40%" style="transform: scaleX(-1);" />
  <img src="Figure/animation.gif" width="21.2%" />
  <img src="Figure/MPC_tighten_bound.jpg" width="32%" /> 
</p>    



## 2024/02/09 QunZhang
This is the result of the MPC_PID_ACC:
<p float="left">
  <img src="Figure/MPC_PID_ACC.gif" width="85%" /> 
</p>   
This is the result of the MPC_PID_ACC without kalman filter (red bounding box is tighten IDM_constraint):  
<p float="left">
  <img src="Figure/IDM_constraint_simulation_plots.gif" width="85%" /> 
</p>   
<p float="left">
  <img src="Figure/simulation_plots.png" width="85%" /> 
</p>   
## 2024/02/14 QunZhang
For the kf_MPC, we use the workflow shown below (we did not use the sensor in carla actually):
<p float="left">
  <img src="Figure/Work_flow.png" width="85%" /> 
</p>   
<p float="left">
  <img src="Figure/ACC_in_heavy_traffic.gif" width="85%" /> 
</p>   
This is the result of the MPC_PID_ACC with kalman filter (red bounding box is tighten IDM_constraint):  
<p float="left">
  <img src="Figure/CARLA_IDM_constraint_simulation_plots_with_filter.gif" width="85%" /> 
</p>   
<p float="left">
  <img src="Figure/CARLA_simulation_plots_kf_state_compare.png" width="85%" /> 
</p>   
<p float="left">
  <img src="Figure/CARLA_simulation_plots_with_filter.png" width="85%" /> 
</p>   
<p float="left">
  <img src="Figure/CARLA_simulation_compare_ref.png" width="85%" /> 
</p>   
This is the result of the MPC_PID_ACC with kalman filter (PID works every 0.2s, MPC works every 1s):
(Computional time of the MPC is:  0.067s with N=12, 0.11s with N=30)
<p float="left">
  <img src="Figure/CARLA_IDM_constraint_simulation_plots_with_filter.gif" width="85%" /> 
</p>   
</p>   
<p float="left">
  <img src="Figure/CARLA_simulation_plots_kf_state_compare_diffF.png" width="85%" /> 
</p>   
<p float="left">
  <img src="Figure/CARLA_simulation_plots_with_filter_diffF.png" width="85%" /> 
</p>   
<p float="left">
  <img src="Figure/CARLA_simulation_compare_ref_diffF.png" width="85%" /> 

## 2024/02/19 QunZhang
TODO List: 
```
1. build heavy traffic scenerio
2. lane changing
3. decision master
```