import random
import numpy as np
import matplotlib.pyplot as plt
def dataset(n):
        X1=[]
        Y1=[]
        X2=[]
        Y2=[]
        w0=0.0
        w1=0.0
        w2=0.0
        w=[w0,w1,w2]
        for i in range(n):
                a=random.uniform(0,10)
                X1.append(a)
                Y1.append(random.uniform(0,10-a))
                b=random.uniform(0,10)
                X2.append(b)
                Y2.append(random.uniform(10-b,10))   
        data1=[np.array([1,X1[i],Y1[i],1]) for i in range(n)]
        data2=[np.array([1,X2[i],Y2[i],-1]) for i in range(n)]
        data=data1+data2
        random.shuffle(data)
        ite=0
        while (misclassified(data,w)):
                i=data[random.randint(0,2*n-1)]
                if sign(i[:3].dot(w))*i[-1]<0:
                        w+=i[:3]*i[-1]
                        ite+=1
        print(w[0], w[1])
        for i in range(n):
                plt.scatter(X1[i],Y1[i],c='r',s=3)
                plt.scatter(X2[i],Y2[i],c='b',s=3)
        X4=np.arange(0,10,0.1)
        Y4=10-X4
        plt.title(str(2*n)+"data randomly generatedin range(0,10)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.plot(X4,Y4,label="target Y=10-X")
        plt.legend()
        plt.show()
        #y=10-x
        X3=np.arange(0,10,0.1)
        #y=ax+b where b =-w0/w2 a=-w1/w2
        Y3=np.array([(X3[i]*w[1]+w[0])/(-w[2]) for i in range(len(X3))])
        X4=np.arange(0,10,0.1)
        Y4=10-X4
        for i in range(n):
                plt.scatter(X1[i],Y1[i],c='r',s=3)
                plt.scatter(X2[i],Y2[i],c='b',s=3)        
        plt.title("After "+str(ite)+" iteration")
        plt.xlabel("X")
        plt.ylabel("Y") 
        plt.ylim(ymax=10)
        plt.ylim(ymin=0)
        plt.plot(X3,Y3,label="PLA learned g(x) with w0: "+str(round(w[0],2))+" w1: "+str(round(w[1],2))+" w2: "+str(round(w[2],2)))
        plt.plot(X4,Y4,label="target f(x) Y=10-X")
        plt.legend()
        plt.show()
def sign(x):
        if x>=0:
                return 1
        else:
                return -1
def misclassified(x,w):
        for i in x:
                if sign(i[:3].dot(w))*i[-1]<0:
                        return True
        return False
if __name__== "__main__":
        dataset(200)


