import matplotlib 
import numpy as np 
import math
import random
import matplotlib.pyplot as plt
import tkinter
import sys
from io import StringIO
from copy import copy
def Gaussian_Mixture(x,ui,Covi,d):
	if np.linalg.det(Covi)==0.0:
		a=1.0/((2*np.pi)**(d/2)*np.linalg.det(Covi)**0.5+0.001)
	else:
		a=1.0/((2*np.pi)**(d/2)*np.linalg.det(Covi)**0.5)
	if Isinvertible(Covi):
		invc = np.linalg.inv(Covi)
	else:
		invc=np.linalg.inv(Covi+0.001*np.identity(d))
	power= -np.dot((x-ui).dot(invc),(x-ui))/2.0
	if power>500:
		b=1
	elif power<-500:
		b=0.0
	else:
		b=math.exp(power)
	return a*b
def Isinvertible(matrix):
	return matrix.shape[0]==matrix.shape[1] and np.linalg.matrix_rank(matrix) == matrix.shape[0]
def clusterAndPurity(Data,PC,k,cov,mu,d):
	n=len(Data)
	X=Data[:,:-1]
	Y=Data[:,-1]
	w=np.array([[0.0 for j in range(n)] for i in range(k)])
	finalcluster=[[] for i in range(k)]
	for i in range(k):
		for j in range(n):
			w[i][j]= (Gaussian_Mixture(X[j],mu[i],cov[i],d)*PC[i])/(sum([Gaussian_Mixture(X[j],mu[a],cov[a],d)*PC[a] for a in range(k)]))
	for j in range(n):
		highprob=0.0
		highclust=0
		for i in range(k):
			if w[i][j]>highprob:
				highprob=w[i][j]
				highclust=i
		finalcluster[highclust].append(Data[j])
	fsize=[len(i) for i in finalcluster]
	truecluster={}
	for i in Data:
		if i[-1] in truecluster:
			truecluster[i[-1]]+=1
		else:
			truecluster[i[-1]]=0
	tc=[]
	for key in truecluster:
		tc.append(truecluster[key])
	fsize=sorted(fsize)
	tc=sorted(tc)
	print("final size of each cluster:",fsize)
	print(tc)
	purity=1-abs(sum(np.array(fsize)-np.array(tc)))/sum(fsize)
	print(purity)
def EM(Data,k,epsilon):
	#initialization
	X=Data[:,:-1]
	Y=Data[:,-1]
	t=0
	d=len(Data[0])-1
	n=len(Data)
	mu=np.array([[random.uniform(min(X[:,di]),max(X[:,di])) for di in range(d)] for ki in range(k)])
	cov=np.array([np.identity(d) for i in range(k)])
	PC=np.array([1.0/k for ki in range(k)])
	while(True):
		mu_prev=copy(mu)
		t+=1
		w=np.array([[0.0 for j in range(n)] for i in range(k)])
		#Expectation Step
		for i in range(k):
			for j in range(n):
				w[i][j]= (Gaussian_Mixture(X[j],mu[i],cov[i],d)*PC[i])/(sum([Gaussian_Mixture(X[j],mu[a],cov[a],d)*PC[a] for a in range(k)]))
		#Maximization Step
		for i in range(k):
			mu[i]=sum([w[i][j]*X[j] for j in range(n)])/sum([w[i][j] for j in range(n)])
			for a in range(d):
				for b in range(d):
					newcov=sum([w[i][j]*((X[j][a]-mu[i][a])*(X[j][b]-mu[i][b])) for j in range(n)]) / sum([w[i][j] for j in range(n)])
					cov[i][a][b]=newcov
			PC[i]=sum([w[i][j] for j in range(n)])/n
		print("iteration:",t,np.linalg.norm(mu-mu_prev),flush=True)
		if (np.linalg.norm(mu-mu_prev)**2<epsilon):
			break
	print("The final mean for each cluster:\n",mu)
	print("The final covariance matrix for each cluster:\n",cov)
	print("Number of iterations the EM algorithm took to converge:\n",t)
	print("P(Ci):",PC)
	clusterAndPurity(Data,PC,k,cov,mu,d)
if __name__ == '__main__':
	ds1="iris.txt"
	ds2="1R2RC_truth.txt"
	ds3="dancing_truth.txt"
	tran=np.genfromtxt(ds1, delimiter=',')
	for i in tran:
		if i[-1]==b'Iris-setosa':
			i[-1]=float(1.0)
		elif i[-1]==b'Iris-versicolor':
			i[-1]=float(2.0)
		elif i[-1]==b'Iris-virginica':
			i[-1]=float(3.0)
		else:
			continue
	print(tran)
	EM(tran,3,0.001)
	
		
				
		
    