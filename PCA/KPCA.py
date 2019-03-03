import matplotlib 
import numpy as np 
import math
import random
import matplotlib.pyplot as plt
import tkinter
import sys
def linear_K(X1,X2):
    return np.dot(X1,X2)
def gaussian_K(X1,X2,sigma):
    X3=X1-X2
    s=0
    for i in X3:
        s+=i**2
    return math.exp(-1*s/(2*sigma))
def KernalMatrix(Data,kf,sigma):
    KM=[]
    for i in Data:
        row=[]
        for j in Data:
            if kf=='linear':
                row.append(linear_K(i,j))
            else:
                row.append(gaussian_K(i,j,sigma))
        KM.append(row)
    return np.array(KM)
def CenterKernal(K):
    # a= I- (1/n) 1_nxn
    # C=aKa
    n=len(K)
    I=np.identity(n)
    print(n)
    ones=np.ones((n,n))
    a=I-ones/(n*1.0)
    return a.dot(K).dot(a)
def eigen(K):
    return np.linalg.eigh(K)
def variance(ei):
    v=[]
    for i in ei:
        v.append(i/len(ei))
    return np.array(v)
def checkUTU1(val,vec):
    for i in range(len(vec)):
        ci=vec[i]
        if ci!=ci.dot():
            return False
    return True
def fr(r,v):
    
    '''
    the fraction of dimensionality fr= sum1toR(variance)/sum1toD(variance)
    '''
    d=len(v)
    nume=0
    ri=0
    totalvariance=sum(v)
    while (ri<r):
        nume+=v[ri]
        ri+=1
    return nume/totalvariance
def findD(v,alpha):
    d=len(v)
    while(fr(d,v)>alpha):
        d-=1
        if d<1:
            break
    if (d+1>len(v)):
        return d
    print("fraction coverage of this dimensionalty:\n%f \nwhich is higher than alpha:\n %f"%(fr(d+1,v),alpha))
    return d+1
def normalize(X):
    return math.sqrt(sum(map(lambda i: i**2, X)))
def KPCA(Data,alpha,kf,sigma):
    if kf=='linear':
        print ("#######################################\nNow begin linear Kernel principle component Analysis\n#######################################\n")
    else:
        print("#######################################\nNow begin gaussian Kernel principle component Analysis\n#######################################\n")
    K=KernalMatrix(Data,kf,sigma)
    print ('K before centered:\n')
    print (K)
    K=CenterKernal(K)
    print ('K after centered:\n')
    print (K)    
    eigen_val, eigen_vec=eigen(K)
    eigen_val=eigen_val[::-1]
    eigen_vec=np.fliplr(eigen_vec)
    for i in range(len(eigen_val)):
        if eigen_val[i]<0:
            eigen_val[i]=0
    eigen_vec = np.array([math.sqrt(1/eigen_val[i]) * eigen_vec.T[i] for i in range(len(eigen_val))]).T
    print("eigen values:")
    print (eigen_val)
    print("eigen vectors:")
    print(eigen_vec)
    v=variance(eigen_val)
    print("variance of each component:\n")
    print(v)
    dimension= findD(v,alpha)
    print("Dimensionality of reduction:")
    print(dimension)
    C_r = eigen_vec[:,:dimension]
    #for i in range(dimension):
     #   C_r[i]=C_r[i] * math.sqrt(1/eigen_val[i])
    print("CR\n")
    print(C_r)
    A = np.dot(K,C_r)
    print("PC:\n")
    print(A)
    plt.figure()
    plt.scatter(A[:,0], A[:,1], color='red', alpha=0.7)
    if kf=='linear':
        plt.title("Linear kpca first two PC of 0.95 alpha")
    else:
        plt.title("gausian kpca first two PC")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()
def CPCA(Data,alpha):
    '''
    normal pca using covriance matrix
    '''
    def getmean(Data):
        return np.mean(Data, axis=0)
    def centerdata(Data,mu):
        return Data-np.transpose(mu[:,None])
    def covariance(Z):
        return np.array([ 
					[np.dot(Z[:,i], Z[:,j]) / len(Z) for i in range(len(Z.T))] 
					for j in range(len(Z.T))
                       ])
    #CPCA main operations
    print("##################3Now begin covariance PCA########################:\n")
    mu =getmean(Data)
    Z= centerdata(Data,mu)
    cvar=covariance(Z)
    eigval,eigvec=np.linalg.eigh(cvar)
    eigvec = np.fliplr(eigvec)
    eigval = eigval[::-1]
    print("eigen values:")
    print(eigval)
    print ("eigen vectors:")
    print (eigvec)       
    dimension=findD(eigval,alpha)
    print("Dimensionality of reduction:")
    print(dimension)
    U_r=eigvec[:,:2]
    print (U_r)
    A= np.matmul(Z,U_r)
    plt.figure()
    plt.scatter(A[:,0], A[:,1], color='red', alpha=0.7)    
    plt.title("Covariance pca first two PC")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()    
if __name__ == "__main__":
    f_name = sys.argv[1]
    spread = float(sys.argv[2])    
    data = np.genfromtxt(f_name,delimiter=',') 
    print (np.std(data))
    KPCA(data,0.95,'linear',0)
    CPCA(data,0.95)
    KPCA(data,0.95,'gaussian',spread)
    
    