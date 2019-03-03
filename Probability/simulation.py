import random
import numpy as np
import matplotlib.pyplot as plt
import math
def findfraction(X):
    counter=0
    for i in X:
        if i==1:
            counter+=1
    return counter/float(10)
## helper function to return an array ready to be plot as histogram
def stat(X):
    S=[0,0,0,0,0,0,0,0,0,0,0]
    for i in X:
        S[i]+=1
    return S
def simulation(n):
    mu=0.5 
    C=[]
    p=[]
    for i in range(n):
        X=[]
        for j in range(10):
            X.append(random.randint(0,1))
        C.append(X)
        counter=0
        for j in range(10):
            if X[j]==1:
                counter+=1
        v=counter/float(10)
        p.append(v)
    randI=random.randint(0,n-1)
    minI=-1
    mini=1
    for i in range(len(p)):
        if(p[i]<mini):
            mini=p[i]
            minI=i
    return findfraction(C[0]),findfraction(C[randI]), findfraction(C[minI])
def estimateError(v,epsilon):
	counter=0
	for i in v:
		if (abs(i/10.0-0.5)>epsilon):
			counter+=1
	return counter/float(len(v))

v1=[]
vrand=[]
vmin=[]
for i in range(1000):
    a,b,c=simulation(1000)
    v1.append(int(a*10))
    vrand.append(int(b*10))
    vmin.append(int(c*10))
v1a=sum(v1)/len(v1)
vranda=sum(vrand)/len(vrand)
vmina=sum(vmin)/len(vmin)
plot1=stat(v1)
plot2=stat(vrand)
plot3=stat(vmin)
v1e=[]
vrande=[]
vmine=[]
epsilon=np.arange(0,1,0.01)
hoff=[]
for i in epsilon:
    hoff.append(2*math.exp(-2*i**2*10))
for i in range(len(epsilon)):
	v1e.append(estimateError(v1,epsilon[i]))
	vrande.append(estimateError(vrand,epsilon[i]))
	vmine.append(estimateError(vmin,epsilon[i]))
plt.bar(range(11),plot1)
plt.xlabel("number of heads")
plt.title("distribution of v1 over 100000 simulation")
plt.show()
plt.bar(range(11),plot2)
plt.xlabel("number of heads")
plt.title("distribution of vrand over 100000 simulation")
plt.show()
plt.bar(range(11),plot3)
plt.xlabel("number of heads")
plt.title("distribution of vmin over 100000 simulation")
plt.show()
plt.plot(epsilon,hoff,label="Hoeffding upper bound")
plt.plot(epsilon,v1e,label="P|v1-u|")
plt.plot(epsilon,vrande,label="P|vrand-u|")
plt.plot(epsilon,vmine,label="P|vmin-u|")
plt.legend()
plt.ylabel("Probability")
plt.xlabel("epsilon")
plt.show()