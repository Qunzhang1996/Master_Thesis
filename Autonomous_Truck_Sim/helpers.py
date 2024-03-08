from casadi import *
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

"""
    Additional helper functions for:
     - Plotting 
     - Data extraction
     - Controllers
"""

def createFeatureMatrix(truck,traffic):
    Nveh = traffic.getDim()
    features = np.zeros((5,1,Nveh+1))
    trafficFeatures = traffic.getFeatures()
    truckState = truck.getState()[:-1]
    truckFeatures = np.array([np.append(truckState,np.array([[0]]))]).T
    featureMap = np.append(truckFeatures,trafficFeatures,axis = 1)
    features[:,0,:] = featureMap
    return features

def features2CSV(X,Nveh,Nsim):
    # Creates data file with features from simulation
    Nvehp1 = Nveh + 1
    X_2D = np.zeros((5,Nvehp1*Nsim))
    for i in range(Nsim):
        X_2D[:,i*Nvehp1:(i+1)*Nvehp1] = np.round(X[:,i,:],5)
    
    DF = pd.DataFrame(X_2D)
    DF.to_csv('simData.csv',index=False,header=False)

def tanh(x):
    return (exp(x)-exp(-x)) / (exp(x)+exp(-x))

def getTotalCost(L,Lf,x,u,refx,refu,N):
            cost = 0
            for i in range(0,N):
                cost += L(x[:,i],u[:,i],refx[:,i],refu[:,i])
            cost += Lf(x[:,N],refx[:,N])
            return cost

def rotmatrix(L,xy,ang):
    x_new = np.cos(ang)*L[0] -np.sin(ang)*L[1] + xy[0]
    y_new = np.sin(ang)*L[0] + np.cos(ang)*L[1] + xy[1]
    return x_new,y_new

def borvePictures(X,X_traffic,X_traffic_ref,paramLog,decisionLog,vehList,X_pred,vehicle,scenarioTrailADV,scenario,traffic,i_crit,f_c,directory):
    print("Generating gif ...")
    Nveh = traffic.getDim()
    vehWidth, vehLength,_,_ = vehicle.getSize()
    roadMin, roadMax, laneCenters = scenario.getRoad()
    laneWidth = 2*laneCenters[0]
    leadWidth,leadLength = traffic.vehicles[0].getSize()
    decision_string = ["Change Left","Change Right","Keep Lane     "]
    figanime = plt.figure(2)
    axanime = figanime.add_subplot(111)

    _,_,L_tract,L_trail = vehicle.getSize()

    def animate(i):
        plt.cla()

        # Plot Road
        frameSize = 70
        X_road  = np.append(X[0,0:i_crit,0],X[0,i_crit,0]+frameSize)
        X_road  = np.append(X[0,0,0]-frameSize,X_road)
        plt.plot(X_road,np.ones((1,i_crit+2))[0]*roadMax/2,'--',color = 'r')
        plt.plot(X_road,np.zeros((1,i_crit+2))[0],'--',color = 'r')
        plt.plot(X_road,np.ones((1,i_crit+2))[0]*roadMin,'-',color = 'k')
        plt.plot(X_road,np.ones((1,i_crit+2))[0]*roadMax,'-',color = 'k')

        # Plot ego vehicle
        X_new = rotmatrix([0,-vehWidth/2],[X[0,i,0],X[1,i,0]],X[3,i,0])
        tractor = Rectangle((X_new[0],X_new[1]),width = L_tract, height = vehWidth, angle = 180*X[3,i,0]/np.pi,
                            linewidth=1, edgecolor = 'k',facecolor='c', fill=True)

        axanime.add_patch(tractor)

        plt.scatter(X[0,i,0],X[1,i,0])

        X_new = rotmatrix([-L_trail,-vehWidth/2],[X[0,i,0],X[1,i,0]],X[4,i,0])
        trailer = Rectangle((X_new[0],X_new[1]),width = L_trail, height = vehWidth, angle = 180*(X[4,i,0])/np.pi,
                        linewidth=1, edgecolor = 'k',facecolor='c', fill=True)

        axanime.add_patch(trailer)
        start = (i) % f_c
        j = 0
        for x in X_pred[0,start:,i]:
            if x < X[0,i]:
                j += 1
        X_pred_x = np.append(X[0,i],X_pred[0,start+j:,i])
        X_pred_y = np.append(X[1,i],X_pred[1,start+j:,i])
        plt.plot(X_pred_x,X_pred_y,'--',color='k')

        # Plot traffic
        colors= {"aggressive" : "r","normal": "b", "passive": "g"}
        for j in range(Nveh):
            color = colors[vehList[j].type]
            axanime.add_patch(Rectangle(
                            xy = (X_traffic[0,i,j]-leadLength/2,X_traffic[1,i,j]-leadWidth/2), width=leadLength, height=leadWidth,
                            angle= 180*X_traffic[3,i,j]/np.pi, linewidth=1, edgecolor = 'k',
                            facecolor=color, fill=True))
            plt.scatter(X_traffic_ref[0,i,j],X_traffic_ref[1,i,j],marker = '.',color = color)

        # Plot box with info
        props = dict(boxstyle='round', facecolor='lightgray', alpha=0.5)
        # textstr = "Velocity: " + str(np.round(X[2,i,0])*3.6) + " (km/h) \n" + str(i) 
        textstr = "Velocity: " + '{:.2f}'.format(round(X[2,i,0]*3.6, 2)) + " (km/h)" 
        # place a text box in upper left in axes coords
        axanime.text(0.05, 0.95, textstr, transform=axanime.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

        props = dict(boxstyle='round', facecolor='lightgray', alpha=0.5)
        textstr = "Decision: " + decision_string[decisionLog[i]] 
        # place a text box in upper left in axes coords
        axanime.text(0.575, 0.95, textstr, transform=axanime.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

        plt.axis('equal')
        plt.xlim(X[0,i,0]-frameSize, X[0,i,0]+frameSize)
        plt.ylim([roadMin-2, roadMax+2])

        # Plot Constraints
        constraint_laneChange = scenario.constraint(traffic,[])
        if decisionLog[i] == 0:
            # Left Change plot
            XY = np.zeros((2,2*frameSize,Nveh))
            for j in range(Nveh):
                p_ij = paramLog[:,i,j,0]
                x_ij = np.arange(-frameSize,frameSize,1)
                for k in range(len(x_ij)):
                    y_cons_ij = constraint_laneChange[j](x_ij[k],p_ij[0],p_ij[1],p_ij[2],p_ij[3])[0].full().item()
                    XY[0,k,j] = X[0,i,0]+x_ij[k]
                    XY[1,k,j] = y_cons_ij

            upperY = np.zeros((2*frameSize,))
            lowerY = np.zeros((2*frameSize,))
            for k in range(len(x_ij)):
                idx = np.where(paramLog[2,i,:,0] < 0)[0]
                idx_upper = np.argmin(XY[1,k,idx]-X[1,i,0])
                upperY[k] = XY[1,k,idx[idx_upper]]
                idx = np.where(paramLog[2,i,:,0] > 0)[0]
                idx_lower = np.argmin(X[1,i,0]-XY[1,k,idx])
                lowerY[k] = XY[1,k,idx[idx_lower]]
            
            idx_join = np.where(lowerY> upperY)[0]
            idx_join_forward = np.where(idx_join > frameSize)[0]
            idx_join_backward = np.where(idx_join < frameSize)[0]
            idx_start = idx_join[idx_join_backward[-1]]-1 if idx_join_backward.size and idx_join[idx_join_backward[-1]] > 0 else 0 
            idx_end = idx_join[idx_join_forward[0]]+1 if idx_join_forward.size and idx_join[idx_join_forward[0]] < 2*frameSize-1 else -1

            plt.plot(XY[0,idx_start:idx_end,0],upperY[idx_start:idx_end],'b', alpha = 1)
            plt.plot(XY[0,idx_start:idx_end,0],lowerY[idx_start:idx_end],'b', alpha = 1)

        elif decisionLog[i] == 1:
            # Right Change plot
            XY = np.zeros((2,2*frameSize,Nveh))
            for j in range(Nveh):
                p_ij = paramLog[:,i,j,1]
                x_ij = np.arange(-frameSize,frameSize,1)
                for k in range(len(x_ij)):
                    y_cons_ij = constraint_laneChange[j](x_ij[k],p_ij[0],p_ij[1],p_ij[2],p_ij[3])[0].full().item()
                    XY[0,k,j] = X[0,i,0]+x_ij[k]
                    XY[1,k,j] = y_cons_ij
                    
            upperY = np.zeros((2*frameSize,))
            lowerY = np.zeros((2*frameSize,))
            for k in range(len(x_ij)):
                idx = np.where(paramLog[2,i,:,1] < 0)[0]
                idx_upper = np.argmin(XY[1,k,idx]-X[1,i,0])
                upperY[k] = XY[1,k,idx[idx_upper]]
                idx = np.where(paramLog[2,i,:,1] > 0)[0]
                idx_lower = np.argmin(X[1,i,0]-XY[1,k,idx])
                lowerY[k] = XY[1,k,idx[idx_lower]]


            idx_join = np.where(lowerY> upperY)[0]
            idx_join_forward = np.where(idx_join > frameSize)[0]
            idx_join_backward = np.where(idx_join < frameSize)[0]
            idx_start = idx_join[idx_join_backward[-1]]-1 if idx_join_backward.size and idx_join[idx_join_backward[-1]] > 0 else 0 
            idx_end = idx_join[idx_join_forward[0]]+1 if idx_join_forward.size and idx_join[idx_join_forward[0]] < 2*frameSize-1 else -1

            plt.plot(XY[0,idx_start:idx_end,0],upperY[idx_start:idx_end],'b', alpha = 1)
            plt.plot(XY[0,idx_start:idx_end,0],lowerY[idx_start:idx_end],'b', alpha = 1)

        else:
            dX_lead = np.sum(paramLog[0,i,:,2])
            min_distx = scenarioTrailADV.min_distx
            D_safe = min_distx + L_tract + leadLength/2
            t_headway = scenarioTrailADV.Time_headway

            if X[1,i,0] > laneWidth:
                lane = 1
            elif X[1,i,0] < 0:
                lane = -1
            else:
                lane = 0
            laneBounds = laneCenters[lane] + np.array([-laneWidth/2,laneWidth/2])

            X_limit = X[0,i,0]+dX_lead.item()-D_safe - X[2,i,0] * t_headway
            plt.plot([X[0,i,0]-frameSize,X_limit],[laneBounds[0],laneBounds[0]],'b')
            plt.plot([X[0,i,0]-frameSize,X_limit],[laneBounds[1],laneBounds[1]],'b')
            plt.plot([X_limit,X_limit],laneBounds,'b')

    anime = FuncAnimation(figanime, animate, frames=i_crit, interval=100, repeat=False)

    writergif = animation.PillowWriter(fps=30) 
    anime.save(directory, writer=writergif)
    print("Finished.")
    plt.show()