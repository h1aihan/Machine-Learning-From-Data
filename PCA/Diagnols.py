import matplotlib 
import numpy as np 
import math
import random
import matplotlib.pyplot as plt
import tkinter
import sys
def angle(A,B):
    '''
    angle=arcos(A dot B/ (norm(A)norm(B)))
    '''
    C=A.dot(B)
    return np.arccos(A.dot(B)/(np.linalg.norm(A)*np.linalg.norm(B)))    
def Generatedata(n,d):
    '''
    Gnerate n pairs of diagnols in d dimension,
    n pairs of [diagnol1, diagnol2]
    diagnol[l1,l2,l3,l4...ld]
    '''
    dsets=[]
    angles=[]
    for i in range(n):
        diagnol1=[]
        diagnol2=[]
        for j in range(d):
            diagnol1.append(random.choice([-1,1]))
            diagnol2.append(random.choice([-1,1]))
        a=angle(np.array(diagnol1),np.array(diagnol2))
        angles.append(a*180/np.pi)
        dsets.append((np.array(diagnol1),np.array(diagnol2)))
    return dsets,angles
def myround(x):
    return round(x+0.5)
def plotPMF(Data,d):
    zero=round(min(Data))
    right=round(max(Data))
    mean=np.mean(Data)
    variance=np.var(Data)
    Xaxis=np.arange(zero,right,0.1)
    Yaxis=[]
    for i in range(len(Xaxis)):
        Yaxis.append(0)
    for i in Data:
        for x in range(len(Xaxis)):
            if i<=Xaxis[x]+0.5 and Xaxis[x]-0.5<=i:
                Yaxis[x]+=1
    Yaxis=np.array(Yaxis)/len(Data)
    print("for %d dimension:"% d)
    print("The minimum value is %d"% zero)
    print("The maximum value is %d"% right)
    print("The mean is %d"%mean)
    print("The variance is %d"% variance)
    title="PMF of "+str(len(Data))+" data in " +str(d)+" dimension"
    plt.scatter(Xaxis,Yaxis,c='b',s=1)
    plt.title(title)
    plt.xlabel("degree between two vectors")
    plt.ylabel("probability")
    plt.show()
    return 0
if __name__ == "__main__":
    d,ags=Generatedata(100,10)
    d2,ags2=Generatedata(100,100)
    d3,ags3=Generatedata(100,1000)
    plotPMF(ags,10)
    plotPMF(ags2,100)
    plotPMF(ags3,1000)
            
        
    