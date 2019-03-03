import matplotlib 
import numpy as np 
import math
import random
import matplotlib.pyplot as plt
import tkinter
import sys
from copy import copy
def StepFunction(net):
    if net<0:
        return 0
    if net==0:
        return 0
    else:
        return 1
def sigmond(x):
    if x>500:
        return 1.0
    if x<-500:
        return 0.0
    return 1 / (1 + math.exp(-x))
def countclasses(Data):
    unique=set()
    for i in Data:
        unique.add(i[len(i)-1])
    return len(unique)
def MLPT(Data,m,neta,maxiter):
    p=countclasses(Data)
    d=len(Data[0])-1
    weights={}
    #Initialize bias vectors
    Thetah=np.array([random.randint(-10,10)/100.0 for i in range(m)])
    Thetao=np.array([random.randint(-10,10)/100.0  for i in range(p)])
    #Initialize weight matrices
    Wh=np.array([[random.randint(-10,10)/100.0  for i in range(m)] for j in range(d)])
    Wo=np.array([[random.randint(-10,10)/100.0  for i in range(p)] for j in range(m)])
    t=0
    while (t<maxiter):
        np.random.shuffle(Data)
        for data in Data:
            X= np.array(data[:len(data)-1])
            X=X.reshape((d,1))
            Y= np.array([0.0 for i in range(p)])
            Y[int(data[len(data)-1])-1]=1
            #feed forward
            netz=Wh.T.dot(X)
            netzh=np.array([i[0] for i in netz])
            netz=netzh+Thetah
            zi= np.array([sigmond(i) for i in netz])
            ziR=zi.reshape((m,1))
            neto=Wo.T.dot(ziR).reshape(p)
            neto=neto+Thetao
            o= np.array([sigmond(i) for i in neto])
            #backpropagation 
            onesp=np.array([1.0 for i in range(p)])
            onesm=np.array([1.0 for i in range(m)])
            theltao = np.multiply(np.multiply(o,onesp-o),(o-Y))
            temp2=np.multiply(zi,onesm-zi)
            temp3=Wo.dot(theltao)
            theltah = np.multiply(temp2,temp3.T)
            #Gradient descent for bias vector  
            Thetao=Thetao-neta*theltao
            Thetah=Thetah-neta*theltah
            #Gradient descent for weight matrix
            theltaor=theltao.reshape((p,1))
            Wo=Wo+(neta*(ziR.dot(theltaor.T)))
            Thetahr=Thetah.reshape((m,1))
            Wh=Wh+(neta*(X.dot(Thetahr.T)))
        t=t+1
        print("Executing ITERATION",t)
        sys.stdout.flush()
    return ((Thetah,Thetao),(Wh,Wo))
def prediction(Bias,Weight,m,Data):
    Thetah,Thetao= Bias
    Wh,Wo =Weight
    print("Thetah:",Thetah)
    print("Thetao:",Thetao)
    print("Wo:",Wo)
    print("Wh:",Wh)    
    predictions=[]
    p=countclasses(Data)
    d=len(Data[0])-1
    accuracy=0
    for data in Data:
        X= np.array(data[:len(data)-1])
        X= X.reshape((d,1))
        Y= int(data[len(data)-1])
        #feed forward
        netz=Wh.T.dot(X)
        netzh=np.array([i[0] for i in netz])
        netz=netzh+Thetah
        zi= np.array([sigmond(i) for i in netz])
        ziR=zi.reshape((m,1))
        neto=Wo.T.dot(ziR).reshape(p)
        neto=neto+Thetao
        o= np.array([sigmond(i) for i in neto])
        output=np.argmax(o)+1
        if output==Y:
            accuracy+=1
        predictions.append(output)
    #print(predictions)
    accuracy=accuracy/len(Data)
    print("Accuracy:%.3f"% (accuracy))
if __name__ == '__main__':
    dtrain=sys.argv[1]
    dtest=sys.argv[2]
    dprac="iris-numeric.txt"
    m= int(sys.argv[3])
    neta= float(sys.argv[4])
    epochs=int(sys.argv[5])
    tran = np.genfromtxt(dtrain, delimiter=',')
    test = np.genfromtxt(dtest , delimiter=',')
    Bias,Weight=MLPT(tran,m,neta,epochs)
    prediction(Bias,Weight,m,test)
    
        
    