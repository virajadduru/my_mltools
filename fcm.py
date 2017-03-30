# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 09:20:51 2015

This module implements a simple Fuzzy C-means clustering algorithm

Paper: "FCM: The Fuzzy c-means Clustering Algorithm"
        J. C. Bezdek, R. Ehrlich, and W. Full,
        Comput. Geosci., vol. 10, no. 2–3, pp. 191–203, 1984

@author: Viraj Adduru
"""
# 

import numpy as np
import matplotlib.pyplot as plt


#%% Data

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
centers = [[4, 2],
           [1, 7],
           [5, 6]]

# Define three cluster sigmas in x and y, respectively
sigmas = [[0.8, 0.3],
          [0.9, 0.5],
          [1.1, 0.7]]
          
np.random.seed(42)
xpts = np.empty(1)
ypts = np.empty(1)
labels = np.empty(1)
for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centers, sigmas)):
    xpts = np.hstack((xpts, np.random.standard_normal(200) * xsigma + xmu))
    ypts = np.hstack((ypts, np.random.standard_normal(200) * ysigma + ymu))
    labels = np.hstack((labels, np.ones(200) * i))
    
fig0, ax0 = plt.subplots()
for label in range(3):
    ax0.plot(xpts[labels == label], ypts[labels == label], '.',
             color=colors[label])
ax0.set_title('Test data: 200 points x3 clusters.')

#%% FCM


c = 3  # number of clusters
m = 2 # fuzzyness
E = 0.00001 # minimum change in the norm for convergence
maxiter=1000 # maximum iterations
alldata=np.vstack((xpts,ypts)) # assigning data

dims = alldata.shape # data dimensions
n= dims[1] # number of data points



C= np.cov(alldata) # covariance of the data
C_inv=np.linalg.inv(C) # taking inverse of cov for mahalanobis distance

# initializing random points for centers
v_init=[]
v_init.append(np.random.randint(xpts.min(),xpts.max(),c))
v_init.append(np.random.randint(ypts.min(),ypts.max(),c))

V = np.asarray(v_init)
# initializing parameters for interations
end_iter=False 
num_iter=0
U=np.empty(1)
U_prev=np.empty(1)
norm = np.inf

# Looping for converging
while end_iter==False and num_iter<=maxiter:
    
    if num_iter!=0: # Not computing for the first iteration as that is for initialization
        v_temp=[]
        u_m = np.power(U,m)
        u_m_sum = np.sum(u_m,axis=1)
        for i in range(c):
            vi=np.sum(alldata*u_m[i,:],axis=1)/u_m_sum[i]
            v_temp.append(vi)
        
        V=np.array(v_temp).T
        
    # calculating mahalanobis Distance matrix
    D = []    
    for i in range(c):
        diff = alldata-V[:,i].reshape(dims[0],1)
        d_sqr = np.diag(np.dot(np.dot(diff.T,C_inv),diff))
        D.append(d_sqr)
    D=np.array(D)
    
    # calculating the membership matrix U
    temp=np.power(D,(2/(m-1)))
    temp1=np.sum(np.reciprocal(temp),axis=0)
    U=np.reciprocal(np.multiply(temp,temp1))  # membership matrix
    
    # condition to terminate iterations
    if num_iter!=0:
        udiff=U-U_prev
        norm=np.linalg.norm(udiff,1)   # calculating norm of the difference matrix
        if norm<E:      # termination criterion
            end_iter=True
        # calculate 
            
    if num_iter%1==0:
        print 'iter:',num_iter,'norm:',norm
    U_prev = U
    num_iter+=1


lab = np.argmax(U,axis=0)


fig0, ax0 = plt.subplots()
for label in range(c):
    ax0.plot(xpts[lab == label], ypts[lab == label], '.',
             color=colors[label])
ax0.set_title('Test data: 200 points x3 clusters.')

#%% Prediction 
num_points=1000
newdata= np.vstack([np.random.uniform(-2,8,num_points),np.random.uniform(0,9,num_points)])

D = []    
for i in range(c):
    diff = newdata-V[:,i].reshape(dims[0],1)
    d_sqr = np.diag(np.dot(np.dot(diff.T,C_inv),diff))
    D.append(d_sqr)
D=np.array(D)

# calculating the membership matrix U
temp=np.power(D,(2/(m-1)))
temp1=np.sum(np.reciprocal(temp),axis=0)
U=np.reciprocal(np.multiply(temp,temp1))  # membership matrix


lab2 = np.argmax(U,axis=0)


fig0, ax0 = plt.subplots()
for label in range(c):
    ax0.plot(newdata[0,lab2 == label], newdata[1,lab2 == label], '.',
             color=colors[label])
    ax0.plot(V[0,label],V[1,label],'o',color=colors[label])
ax0.set_title('Prediction : 500 points 3 clusters.')