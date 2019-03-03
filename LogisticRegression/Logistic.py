import matplotlib 
import numpy as np 
import math
import random
import matplotlib.pyplot as plt
import tkinter
import sys
from copy import copy
#logistic function sigmond
def sigmond(z):
	return 1/(1+math.exp(-z))
# Logisitic Regression Algorithm: Stochastic Gradient Ascent
def LSGA(Data,epsilon,eta):
	w=np.array([0 for i in range(len(Data.T))])
	w_prev=np.array([1 for i in range(len(Data.T))])
	while(np.linalg.norm(w-w_prev)>=epsilon):
		w_prev=copy(w)
		np.random.shuffle(Data)
		X = Data[:,:len(Data.T) - 1]
		Y = Data[:,len(Data.T) - 1]
		Y = np.array([[y] for y in Y])
		ones=np.array([[1] for i in range(len(X))])
		X=np.concatenate((ones,X), axis=1)
		for i in range(len(Data)):
			x=X[i]
			y=Y[i]
			Gradient=((y-sigmond(np.dot(w,x)))*x)
			w= w+eta*Gradient
	return w
def prediction(Data,weights):
	predictions=[]
	X=Data[:,:len(Data.T)-1]
	ones=np.array([[1] for i in X])
	X=np.concatenate((ones,X),axis=1)
	for x in X:
		yj=sigmond(np.dot(weights,x))
		if yj>=0.5:
			predictions.append(1)
		else:
			predictions.append(0)
	counter=0.0
	Y=Data[:,len(Data.T)-1]
	
	#print(predictions)
	for i in range(len(Y)):
		if Y[i]==predictions[i]:
			counter+=1
	accuracy=counter/len(Y)
	return accuracy
if __name__ == '__main__':
	training_f = sys.argv[1]
	test_f = sys.argv[2]
	epsilon=float(sys.argv[3])
	eta = float(sys.argv[4])
	epsilonrange=np.arange(0.1,1,0.1)
	etarange=np.arange(0.01,0.1,0.01)
	training=np.genfromtxt(training_f,delimiter=',')
	testing=np.genfromtxt(test_f,  delimiter=',')
	weights=LSGA(training,epsilon,eta)
	accuracy=prediction(testing,weights)
	print ("With eta:%.3f, epsilon%.3f we have accuracy%.3f"%(eta,epsilon,accuracy))
	
	for i in epsilonrange:
		weights=LSGA(training,i,eta)
		accuracy=prediction(testing,weights)
		print("With eta:%.3f, epsilon%.3f we have accuracy%.3f"%(eta,i,accuracy))
		plt.scatter(i,accuracy,c='r')
	plt.xlabel("epsilon")
	plt.ylabel("accuracy")
	plt.title("With eta fixed at 0.01, epsilon vs accuracy")
	plt.show()	
	for i in etarange:
		weights=LSGA(training,i,eta)
		accuracy=prediction(testing,weights)
		print("With eta:%.3f, epsilon%.3f we have accuracy%.3f"%(i,epsilon,accuracy))
		plt.scatter(i,accuracy,c='b')
	plt.xlabel("eta")
	plt.ylabel("accuracy")	
	plt.title("With epsilon fixed at 0.3, eta vs accuracy")
	plt.show()

    