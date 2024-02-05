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
<div style="display: flex;">
  <div style="flex: 1;">
    <div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
      <img src="Figure/CARLA_KF_TEST.gif" style="width: 50%; height: 200px;" />
      <img src="Figure/animation.gif" style="width: 50%; height: 200px;" />
      <img src="Figure/x_difference.jpg" style="width: 50%; height: 200px;" /> 
      <img src="Figure/y_difference.jpg" style="width: 50%; height: 200px;" />
    </div>
  </div>

  <div style="flex: 1; display: flex; align-items: center; justify-content: center;">
    <img src="Figure/MPC_tighten_bound.jpg" style="width: 100%; height: 400px;" />
  </div>
</div>




TODO List: 
```
1. Finish the one degree vehicle kinematic model, done
2. Linerize the model and try apply kf for it, done
3. Try ACC for the car model   
   ACC_PID DONE
   MPC_PID TBD
```
<p float="left">
  <img src="Figure/ACC_PID.gif" width="100%" /> 
</p>   