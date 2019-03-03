import matplotlib 
import numpy as np 
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
import tkinter
import sys
def mean(D):
	return np.mean(np.array(D))
def variance(D):
	return np.var(np.array(D))
## covariance between two variables
def covariance(X,Y):
	t=0;
	for i in range(len(X)):
		t=t+(X[i]-mean(X))*(Y[i]-mean(Y))
	return t/len(X)
def centeredDataMatrix(Data,u):
	Z=[]
	for i in range(len(Data)):
		Z.append(np.array(Data[i])-u[i])
	return np.array(Z)
def innerCovarianceMatrix(Z):
	return np.array([ 
				[np.dot(Z[:,i], Z[:,j]) / len(Z) for i in range(len(Z.T))] 
				for j in range(len(Z.T))
			])
# = Sum from i=1 to n (zi dot zi^T) divide n
def outerCovarianceMatrix(Z):
	return sum([np.outer(Z[i],Z[i])for i in range(len(Z))])/len(Z)
		
##covariance matrix using covariance between two variables
def covarianceMatrix(Data):
	M=[]
	for i in Data:
		R=[]
		for j in Data:
			R.append(covariance(i,j))
		M.append(R)
	return M
## total variance is the sum of each variance
def totalvariance(Data):
	t=0;
	for i in Data:
		t+=variance(i)
	return t
## correlation between two variables
def correlation(X,Y):
	return covariance(X,Y)/ math.sqrt(variance(X)*variance(Y))
def corelationMatrix(Data):
	M=[]
	for i in Data:
		R=[]
		for j in Data:
			R.append(correlation(i,j))
		M.append(R)
	return M
def IdentityMatrix(size):
	I=[]
	for i in range(size):
		R=[]
		for j in range(size):
			if i==j:
				R.append(1)
			else:
				R.append(0)
def dominantValue(V):
	return max(V)
def eigenvalue(C, epsilon):
	xi=[]
	d1=1
	d2=0
	for i in C:
		xi.append(1)
	Xi=np.array(xi)
	while(abs(d2-d1)>epsilon):
		xtemp=C.dot(Xi)
		k=dominantValue(xtemp)
		d1=d2
		d2=k
		xtemp=np.array(xtemp)
		Xi=xtemp/k
	return d2,Xi
def project(X,D):
	for i in range(len(D)):
		D[i]= np.array(D[i])*X[i] 
	return D
fn = str(sys.argv[1])
epsilon = float(sys.argv[2])
X1=[]
X2=[]
X3=[]
X4=[]
X5=[]
X6=[]
with open(fn) as f:
	data=[line.split() for line in f]  
for i in data:
	X1.append(float(i[0]))
	X2.append(float(i[1]))
	X3.append(float(i[2]))
	X4.append(float(i[3]))
	X5.append(float(i[4]))
	X6.append(float(i[5]))
Data=[]
u=[]
Data.append(X1)
Data.append(X2)
Data.append(X3)
Data.append(X4)
Data.append(X5)
Data.append(X6)
for i in Data:
	u.append(mean(i))
print("Mean vector is : \n"+str(u))
v=totalvariance(Data)
print("Total variance var(D) is: \n" +str(v))
#covariance matrix as pair covariance
c=covarianceMatrix(Data)
#centerDataMatrix
Z=np.transpose(centeredDataMatrix(Data,u))
#covariance matrix as inner product
I=innerCovarianceMatrix(Z)
#covariance matrix as outer product
O=outerCovarianceMatrix(Z)
#correlation mtarix
R=corelationMatrix(Data)
for i in range(len(R)):
	for j in range(len(R[i])):
		R[i][j]=round(R[i][j],5)

print("Correlation Matrix:\n")
for i in R:
	print (i)
ctrue=np.cov(Data)
print("CovarianceMatrix using Inner product:\n")
print(I)
print("CovarianceMatrix using Outer product:\n")
print(O)
w,v =eigenvalue(ctrue,epsilon)
print("dominant eigen value found by power iteration:\n")
print(w)
print("eigen vector:\n")
print(v)
print("Projection:\n")
Dnew=project(w*v,Data)
for i in Dnew:
	print(i)

for i in range(len(X1)):
	plt.scatter(X2[i],X5[i],c='r',s=3)
plt.title("The most correlated variables X2,X5, with correlation 0.753")
plt.show()
for i in range(len(X1)):
	plt.scatter(X2[i],X3[i],c='b',s=3)
plt.title("The most anti-correlated variables X2,X3, with correlation -0.505")
plt.show()
for i in range(len(X1)):
	plt.scatter(X1[i],X3[i],c='b',s=3)
plt.title("The least correlated variables X1,X3, with correlation -0.00366")
plt.show()
