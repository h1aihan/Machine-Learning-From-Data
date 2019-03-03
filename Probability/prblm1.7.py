import random
import numpy as np
import matplotlib.pyplot as plt
import math
def sixthrow():
	X1=[]
	X2=[]
	x1v=0
	x2v=0
	for i in range(6):
		X1.append(random.randint(0,1))
		X2.append(random.randint(0,1))
	for i in range(6):
		if X1[i]==1:
			x1v+=1
		if X2[i]==1:
			x2v+=1
	return max(x1v,x2v)
def simulation(n,epsilon):
	maxv=[]
	for i in range(n):
		maxv.append(sixthrow())
	counter=0
	for i in maxv:
		if abs(i-3)/(6.0)>epsilon:
			counter+=1
	return float(counter)/n
epsilon= np.arange(0,1,0.01)
hoeffding=np.array([2*math.exp(-2*i**2*6) for i in epsilon])
p=[]
for i in epsilon:
	p.append(simulation(10000,i))
plt.plot(epsilon,p,label="P[maxv|vi-ui|>epsilon]")
plt.plot(epsilon,hoeffding,label="Hoeffding bound")
plt.title("P[maxi|vi-ui|>epsilon] after 10000 simulations")
plt.xlabel("epsilon")
plt.ylabel("probability")
plt.ylim(0,2)
plt.legend()
plt.show()