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

def borvePictures(X,X_traffic,X_traffic_ref,vehList,X_pred,vehicle,scenario,traffic,i_crit,f_c,directory):
    print("Generating gif ...")
    Nveh = traffic.getDim()
    vehWidth, vehLength,_,_ = vehicle.getSize()
    roadMin, roadMax, laneCenters = scenario.getRoad()
    leadWidth,leadLength = traffic.vehicles[0].getSize()
    
    figanime = plt.figure(2)
    axanime = figanime.add_subplot(111)

    if vehicle.name == "truck_trailer_bicycle":
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

            plt.axis('equal')
            plt.xlim(X[0,i,0]-frameSize, X[0,i,0]+frameSize)
            plt.ylim([roadMin-2, roadMax+2])

    else:

        def animate(i):
            plt.cla()

            # Plit ego vehicle
            axanime.add_patch(Rectangle(
                            xy = (X[0,i,0]-vehLength/2,X[1,i,0]-vehWidth/2), width=vehLength, height=vehWidth,
                            linewidth=1, color='blue', fill=True))

            plt.plot(X_pred[0,:,i],X_pred[1,:,i],'--',color='k')

            # Plot traffic
            for j in range(Nveh):
                axanime.add_patch(Rectangle(
                                xy = (X_traffic[0,i,j]-leadLength/2,X_traffic[1,i,j]-leadWidth/2), width=leadLength, height=leadWidth,
                                linewidth=1, color='green', fill=True))
            
            # Plot Road
            plt.plot(X[0,0:i_crit,0],np.ones((1,i_crit))[0]*roadMax/2,'--',color = 'r')
            plt.plot(X[0,0:i_crit,0],np.zeros((1,i_crit))[0],'--',color = 'r')
            plt.plot(X[0,0:i_crit,0],np.ones((1,i_crit))[0]*roadMin,'-',color = 'k')
            plt.plot(X[0,0:i_crit,0],np.ones((1,i_crit))[0]*roadMax,'-',color = 'k')

            plt.axis('equal')
            plt.xlim(X[0,i,0]-50, X[0,i,0]+50)
            plt.ylim([roadMin-2, roadMax+2])

    anime = FuncAnimation(figanime, animate, frames=i_crit, interval=100, repeat=False)

    writergif = animation.PillowWriter(fps=30) 
    anime.save(directory, writer=writergif)
    print("Finished.")
    plt.show()