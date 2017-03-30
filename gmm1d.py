# -*- coding: utf-8 -*-

# Fits a GMM using EM algorithm 
# First it uses Kmeans for initial estimates of cluster centroids and uses
# these centroids as initial centroids for GMM.

# Author: Viraj Adduru
# Date: 7.7.2016

import scipy.stats as ss
from matplotlib import pyplot as plt
import numpy as np


def softmax(x):
    x_exp = np.exp(x)
    return x_exp/np.sum(x_exp)
    
    
def fit_gmm(data,
            n_clusters,
            tol = 0.1,
            max_iter = 100,
            n_iter=1,
            verbose = False):
    """
    data : number of samples as rows and features as columns
    
    """
    
    # Performing K means
    if verbose == True:
        print 'Performing KMeans for initial cluster centroids'
        
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters = n_clusters,
                init = 'k-means++',
                n_init = n_iter,
                max_iter = 100,
                tol = tol,
                verbose = verbose)
                
    km.fit(data)
    
    # Performing EM with above cluster centroids.
    if verbose == True:
        print 'Performing EM for GMM'
    # Initializing
    gamma_init = np.random.uniform(high = 10,size = n_clusters)
    lbda = softmax(gamma_init)
    mean = km.cluster_centers_
    sdev = np.ones(shape = [n_clusters,1])*data.std() # this is only for 1d data
    # here we are initializing sdev with sdev of data. This helps when the 
    # data has large values
    
    n,dims = data.shape
    #n = data.size
    # EM
    LL = np.Inf # log likelihood variable initialized to infinity
    itercount = 0
    converged = False
    
    while (itercount<max_iter):
        theta = zip(lbda,mean,sdev)
        
        # Expectation Step
        q_kn = np.hstack([ss.norm.pdf(data,mu,sig)*pv 
                            for pv,mu,sig in theta])
        q_n = np.sum(q_kn,1,keepdims = True)
        LL_now = np.sum(np.log(q_n))
        
        LL_diff = np.abs(LL_now - LL)
        if verbose == True:
            print 'Iteration', itercount, \
                    ', Inertia', LL_diff
                    
        if LL_diff < tol:
            if verbose == True:
                print 'Converged at iteration',itercount
            
            converged = True
            break  
        LL = LL_now
        
        p_kn = q_kn/q_n
        p_k = np.sum(p_kn,0)
        
        # Maximization step
        lbda = p_k/n
        mean = np.sum((p_kn*data),0)/p_k
        sdev = np.sqrt(np.sum(p_kn*(np.square(data-mean)),0)/p_k)
        
        itercount = itercount + 1
    
    if verbose == True:
        print 'Converged: ',converged
    
    if len(mean.shape) == 1:
        mean.resize([mean.size,1])
    if len(sdev.shape) == 1:
        sdev.resize([sdev.size,1])
        
    # Returning sklearn format GMM.
    

    from sklearn.mixture import GaussianMixture

    model = GaussianMixture(n_components = n_clusters,
                verbose = True, 
                max_iter = 0, 
                init_params = '', 
                covariance_type = '')
                
    model.means_ = mean
    model.covars_ = np.square(sdev)
    model.weights_ = lbda
    model.converged_ = converged
    
    return model
    
#%% Main function
def main():
    g1 = ss.norm.rvs(10,2,10000)
    g2 = ss.norm.rvs(15,1,20000)
    g3 = ss.norm.rvs(20,3,34000)
    
    data = np.concatenate((g1,g2,g3))
    # displaying data
    cc=plt.hist(data,100)
    plt.title('Data')
    
    # fitting GMM    
    model = fit_gmm(data = data.reshape([-1,1]),
                    n_clusters = 3,tol = 0.01,
                    max_iter = 100,verbose = True)   
    
    # displaying model
    x_disp = np.linspace(0,35,100)
    theta_disp = zip(model.weights_,
                     model.means_.flatten(),
                     model.covars_.flatten()) 
    plt.figure()
    
    for pv,mu,sig in theta_disp:
        q_kn_disp = ss.norm.pdf(x_disp,mu,np.sqrt(sig))*pv
        plt.plot(x_disp,q_kn_disp)
        
    cc=plt.hist(data, normed = 1,bins = 100)
    plt.title('gmm1d')    
    
#%% testing   
    
if __name__ == '__main__':
    main()

