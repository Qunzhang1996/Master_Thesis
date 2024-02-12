# Master_Thesis
This the repo for master_thesis
## 2024/01/24 QunZhang
Create the repo and add some related folder   
TODO List: 
```
1. Finish the one degree vehicle kinematic model
2. Linerize the model and try apply kf for it
3. Try ACC for the car model
```
## 2024/01/26 QunZhang
Add kalman filter and test it in CARLA simulation, and tighten the constraint as shown below: 
<p float="left">
  <img src="Figure\CARLA_KF_TEST.gif" width="40%" />
  <img src="Figure/animation.gif" width="21%" />
  <img src="Figure/MPC_tighten_bound.jpg" width="26.5%" /> 
</p>    

TODO List: 
```
1. Finish the one degree vehicle kinematic model, done
2. Linerize the model and try apply kf for it, done
3. Try ACC for the car model   
   ACC_PID DONE
   MPC_PID Done
```
This is the result of IDM_PID:
<p float="left">
  <img src="Figure/ACC_PID.gif" width="80%" /> 
</p>   
This is the result of the MPC_PID_ACC:
<p float="left">
  <img src="Figure/simulation_plots.png" width="80%" /> 
</p>   

## 2024/02/09 QunZhang
TODO List: 
```
1. Use chance MPC to tighten the IDM 
2. Use chance MPC to tighen the velocity diff
3. Add the kf into the simulation
4. Try lane changing
```