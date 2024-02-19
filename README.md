# Master_Thesis
This the repo for master_thesis
## 2024/01/24 QunZhang
Create the repo and add some related folder   
TODO List: 
```
1. Finish the vehicle kinematic model
2. Linerize the model and try apply kf for it
3. Try ACC for the car model
```
## 2024/01/26 QunZhang
Add kalman filter and test it in CARLA simulation, and tighten the constraint as shown below: 
<p float="left">
  <img src="Figure\CARLA_KF_TEST.gif" width="40%" style="transform: scaleX(-1);" />
  <img src="Figure/animation.gif" width="21.2%" />
  <img src="Figure/MPC_tighten_bound.jpg" width="32%" /> 
</p>    

TODO List: 
```
1. Finish the vehicle kinematic mode, Done
2. Linerize the model and try apply kf for it, Done
3. Try ACC for the car model   
   ACC_PID Done
   MPC_PID Done
```
This is the result of IDM_PID:
<p float="left">
  <img src="Figure/ACC_PID.gif" width="80%"  style="transform: scaleX(-1); /> 
</p>   


## 2024/02/09 QunZhang
TODO List: 
```
1. Use chance MPC to tighten the IDM, Done, 2024/2/14
2. Use chance MPC to tighen the velocity diff, Done, 2024/2/14
3. Add the kf into the simulation, Done, 
4. Try lane changing
```
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