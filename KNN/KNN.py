import matplotlib 
import numpy as np 
import math
import random
import matplotlib.pyplot as plt
import tkinter
import sys
rad=10
thk=5
sep=5
def GnerateData(thk,rad,sep,n,x1=0,y1=0):
    r1=rad+thk
    x2=x1+r1
    y2=y1-sep
    Data1=[]
    Data2=[]
    XR1=[]
    XR2=[]
    for i in range(n):
        x= random.uniform(-r1,r1)
        if (abs(x)>rad):
            ymi=0
        else:
            ymi=math.sqrt(rad**2-x**2)
        ymax=math.sqrt(r1**2-x**2)
        y= random.uniform(ymi,ymax)
        d1x=x1+x
        d1y=y1+y
        d2x=x2-x
        d2y=y2-y        
        Data1.append(np.array([d1x,d1y,1]))
        Data2.append(np.array([d2x,d2y,-1]))
        XR1.append(np.array([d1x,d1y]))
        XR2.append(np.array([d2x,d2y]))
        plt.scatter(d1x,d1y,c='r',alpha=0.5,s=1)
        plt.scatter(d2x,d2y,c='b',alpha=0.5,s=1)
    Data=np.array(Data1+Data2)
    #plt.show()
    return Data
def d(x,x1):
    return np.linalg.norm(x-x1)
def KNN(data,k,x):
    distances=[]
    for i in range(len(data)):
        X=np.array(data[i][:-1])
        Y=data[i][-1]
        distance=d(X,x)
        distances.append([distance,i])
    distances.sort()
    knpi=[]
    for i in range(k):
        knpi.append(distances[i][1])
    plus1=0
    minus1=0
    for i in knpi:
        if data[i][2]==1.0 or data[i][2]==1:
            plus1+=1
        else:
            minus1+=1
    if plus1>minus1:
        return 1
    else:
        return -1
def plotdata(data,k):
    X1l=np.linspace(-30,30,250)
    X2l=np.linspace(-30,30,250)
    X1,X2=np.meshgrid(X1l,X2l) 
    Z=[]
    newZ=[]
    for x,y in zip(X1,X2):
        Z.append(zip(x,y))
    for z in Z:
        zz=[KNN(data,k,tdata) for tdata in z]
        newZ.append(zz)
    print(newZ)
    cp=plt.contour(X1,X2,newZ)
    plt.title('3-NN Plot')
    plt.xlabel('X1')
    plt.ylabel('X2')    
    #plt.colorbar(cp)
    plt.show()  
data=GnerateData(thk,rad,sep,1000,0,0)
plotdata(data,3)