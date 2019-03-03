import matplotlib 
import numpy as np 
import math
import random
import matplotlib.pyplot as plt
import tkinter
import sys
from copy import copy
def AverageIntensity(data):
    data=data[1:]
    s=0
    for i in data:
        s+=float(i)
    s=s/len(data)
    return s
def symmetry(data):
    data=data[1:]
    ig=[data[i:i+16] for i in range(0,len(data),16)]
    s1=0
    s2=0
    for i in range(1,16):
        for j in range(1,8): 
            s1+=float(ig[16-j][i-1])
    for i in range(1,16):
        for j in range(8,16): 
            s2+=float(ig[16-j][i-1]) 
    if (abs(s1)<abs(s2)):
        return s1/s2
    else:
        return s2/s1
def featureExtraction(data):
    train=[]
    for d in data:
        ai=AverageIntensity(d)
        si=symmetry(d)
        if(d[0]==1):
            y=1
        else:
            y=-1
        train.append([ai,si,y])
    return np.array(train)
def randomSet300(data):
    sampleIndex=random.sample(range(0,len(data)),300)
    Dtran=[]
    Dtest=[]
    for i in sampleIndex:
        Dtran.append(data[i])
    for i in range(len(data)):
        if i not in sampleIndex:
            Dtest.append(data[i])
    return Dtran,Dtest
def drawplane():
    plt.scatter(1,0)
    plt.scatter(-1,0)
    plt.plot([0,0,0],[-1.2,0,1.2])
    X=np.arange(-1.2,1.2,0.01)
    Y=[x**3 for x in X]
    plt.plot(X,Y)
    plt.ylim(-1.2,1.2)
    plt.xlim(-1.2,1.2)
    plt.show()
    import matplotlib 
    import numpy as np 
    import math
    import random
    import matplotlib.pyplot as plt
    import tkinter
    import sys
    from copy import copy
    '''
    Implement Algorithm 21.1 for the Dual SVM training based on SGA. Use hinge loss. 
    After computing the alpha values, print support-vectors i,alphai such that alphai>0 and 
    compute accuracy on test set.
    '''
def quadradic_K(x1,x2):
    return x1.dot(x2)**2
def linear_K(X1,X2):
    return np.dot(X1,X2)
def gaussian_K(X1,X2,sigma):
    X3=X1-X2
    s=0
    for i in X3:
        s+=i**2
    return math.exp(-1*s/(2*sigma))
def Kernal(ktype,sigma,X1,X2):
    if ktype=="linear":
        return linear_K(X1,X2)
    if ktype=="quadradic":
        return quadradic_K(X1,X2)
    else:
        return gaussian_K(X1,X2,sigma)
def KernalMatrix(Data,ktype,sigma):
    KM=[]
    for i in Data:
        row=[]
        for j in Data:
            row.append(Kernal(ktype,sigma,i,j))         
        KM.append(row)
    return np.array(KM)
def SVM_DUAL(oD,ktype,C,epsilon):
    sigma=0
    # map bias
    X=[]
    Y=[]
    for i in oD:
        x=i[:-1]
        x=np.append([1],x)
        y=i[-1]
        X.append(x)
        Y.append(y)
    X=np.array(X)
    Y=np.array(Y)
    # create Kernal 
    K=KernalMatrix(X,ktype,0.1)
    # growth rate neta
    neta=[]
    for k in range(len(X)):
        netak=1.0/K[k][k]
        neta.append(netak)
    neta=np.array(neta)
    #calculate alpha through gradient ascent
    t=0
    alpha=np.array([0.0 for i in X])
    while(True):
        alpha_prev=copy(alpha)
        for k in range(len(X)):
            gradient= 1-Y[k]*(sum([alpha[i]*Y[i]*K[i][k] for i in range(len(X))]))
            alpha[k]= alpha[k]+neta[k]*gradient
            if alpha[k]<0:
                alpha[k]=0.0
            if alpha[k]>C:
                alpha[k]=C
        t=t+1
        print(np.linalg.norm(alpha-alpha_prev))
        sys.stdout.flush()
        if (np.linalg.norm(alpha-alpha_prev)<=epsilon):
            break
    supportindex=[]
    for i in range(len(alpha)):
        if alpha[i]!=0:
            supportindex.append(i)
    print("support vector between 0 and 1",supportindex)
    '''
    for i in supportindex:
        print("relative alpha, its X, its Y",alpha[i],X[i], Y[i])
    '''
    return K,alpha,supportindex,X,Y
def sign(x):
    if x>0:
        return 1
    else:
        return -1
def predict(test,K_matrix,alpha,supportindex,X,Y,ktype='linear',sigma=0.1):
    accuracy=0
    predictions=[]
    for i in test:
        x=i[:-1]
        x=np.append([1],x)
        y=i[-1]
        sumc=(sum([alpha[j]*Y[j]*Kernal(ktype,sigma,x,X[j]) for j in supportindex]))
        yhat=sign(sumc)
        predictions.append(yhat)
        if yhat==y:
            accuracy+=1
    accuracy/=len(test)
    print(accuracy)
    print(Y)
    print(predictions)
def predict2(tdata,K_matrix,alpha,supportindex,X,Y,ktype='linear',sigma=0.1):
    print("flag",tdata,flush=True)
    tdata=np.append([1],tdata)
    sumc=(sum([alpha[j]*Y[j]*Kernal(ktype,sigma,tdata,X[j]) for j in supportindex]))
    yhat=sign(sumc)
    return yhat
def plotdata(weights):
    X1l=np.arange(-1.2,1.2,0.01)
    X2l=np.arange(-1.2,1.2,0.01)
    X1,X2=np.meshgrid(X1l,X2l)
    transformed=[Transformx1x2(x1, x2,8) for x1,x2 in zip(X1,X2)]
    Z=[]
    for x in transformed:
        weighted=sum([w*x for w,x in zip(weights,x)])
        Z.append(weighted)
    plt.title("Decision boundary with lambda=1.35 regularization on training")
    plt.contour(X1, X2, Z, [0])
    plt.axis([-1.1, 0, -1.1, 1.1])
    plt.show() 
def plottran(data):
    for i in data:
        if i[2]==1:
            plt.scatter(i[0],i[1],c='r',marker='x',s=5)
        else:
            plt.scatter(i[0],i[1],c='b',marker='x',s=5)
if __name__ == '__main__':
    
    data1 =np.genfromtxt("ZipDigits.train",delimiter=' ')
    data2= np.genfromtxt("ZipDigits.test",delimiter=' ') 
    data=[]
    ktype="guassian"
    for i in data1:
        data.append(i)    
    for i in data2:
        data.append(i)
    np.random.shuffle(np.array(data))
    tran=featureExtraction(data)
    Dtran,Dtest=randomSet300(tran)     
    K,alpha,supportindex,X,Y=SVM_DUAL(Dtran,ktype,1.5,0.01)
    predict(Dtest,K,alpha,supportindex,X,Y,ktype)    
    plottran(Dtran)
    ######
    X1l=np.linspace(-1.0,1.0,100)
    X2l=np.linspace(-1.0,1.0,100)
    X1,X2=np.meshgrid(X1l,X2l) 
    Z=[]
    Z=[]
    newZ=[]
    for x,y in zip(X1,X2):
        Z.append(zip(x,y))
    print (Z)
    for z in Z:
        zz=[predict2(tdata,K,alpha,supportindex,X,Y,ktype) for tdata in z]
        newZ.append(zz)
    #Z=[predict(X1[i],X2[i],K,alpha,supportindex,X,Y,ktype) for i in range(len(X1))]
    plt.title("SVM Decision boundary with C=20 regularization on training")
    plt.contour(X1, X2, newZ)
    plt.axis([-1.1, 0, -1.1, 1.1])
    plt.show()     