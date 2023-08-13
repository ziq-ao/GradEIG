import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy import stats
import scipy.special as spl

import sys

#import ml.trainers as trainers
#import ml.step_strategies as ss


def kl(y, n=None, k=1, shuffle=True, standardize=True, rng=np.random):
    
    y = np.asarray(y, float)
    
    if standardize == True:
        y_std = np.std(y, axis=0)
        y = y/y_std
        
    N, dim = y.shape
    
    if n is None:
        n = N
    else:
        n = min(n, N)
    
    # permute y
    if shuffle is True:
        rng.shuffle(y)
    
    # knn search
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='chebyshev').fit(y)
    dist, idx = nbrs.kneighbors(y)
    
    if standardize == True:
        hh = dim*np.log(2*dist[:,k])+np.sum(np.log(y_std))
    else:
        hh = dim*np.log(2*dist[:,k])
        
    h = -spl.digamma(k)+spl.digamma(N)+np.mean(hh)
    
    return h


def tkl(y, n=None, k=1, shuffle=True, rng=np.random):
    
    y = np.asarray(y, float)
    N, dim = y.shape
    
    if n is None:
        n = N
    else:
        n = min(n, N)
    
    # permute y
    if shuffle is True:
        rng.shuffle(y)
    
    # knn search
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='chebyshev').fit(y)
    dist, idx = nbrs.kneighbors(y)
    
    r = dist[:,k]
    r = np.tile(r[:, np.newaxis], (1, dim))
    lb = (y-r >= 0)*(y-r) + (y-r < 0)*0
    ub = (y+r <= 1)*(y+r) + (y+r > 1)*1
    hh = np.log(np.prod(ub-lb, axis=1))
        
    h = -spl.digamma(k)+spl.digamma(N)+np.mean(hh)
    
    return h


def mi_kl(y, dim_x, n=None, k=1, shuffle=True, rng=np.random):
    
    y = np.asarray(y, float)
    
    #standardize
    y = y/np.std(y, axis=0)
    
    N, dim = y.shape
    
    if n is None:
        n = N
    else:
        n = min(n, N)
    
    # permute y
    if shuffle is True:
        rng.shuffle(y)
    
    x1 = y[:, :dim_x]
    x2 = y[:, dim_x:]
    # knn search
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='chebyshev').fit(y)
    dist, idx = nbrs.kneighbors(y)
    
    nbrs_1 = NearestNeighbors(n_neighbors=N, algorithm='auto', metric='chebyshev').fit(x1)
    dist_1, idx_1 = nbrs_1.kneighbors(x1)
    
    nbrs_2 = NearestNeighbors(n_neighbors=N, algorithm='auto', metric='chebyshev').fit(x2)
    dist_2, idx_2 = nbrs_2.kneighbors(x2)
    
    n_x = np.empty(n)
    n_y = np.empty(n)
    for i in range(n):
        n_x[i] = np.sum(dist_1[i,1:] < dist[i,k])
        n_y[i] = np.sum(dist_2[i,1:] < dist[i,k])
    
    mi = spl.digamma(k)-np.mean(spl.digamma(n_x+1)+spl.digamma(n_y+1))+spl.digamma(N)
    
    return mi



def ksg(y, n=None, k=1, shuffle=True, standardize=True, rng=np.random):
    """
    Implements the KSG entropy estimation in m-dimensional case, as discribed by:
    Alexander Kraskov, Harald Stogbauer, and Peter Grassberger, "Estimating Mutual Information", Physical review E, 2004 
    """
    
    y = np.asarray(y, float)
    
    if standardize == True:
        y_std = np.std(y, axis=0)
        y = y/y_std
        
    N, dim = y.shape
    
    if n is None:
        n = N
    else:
        n = min(n, N)
    
    # permute y
    if shuffle is True:
        rng.shuffle(y)
    
    # knn search
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='chebyshev').fit(y)
    dist, idx = nbrs.kneighbors(y)
    
    hh = np.empty(n)
    
    if standardize == True:
        for j in range(n):
            r = np.max(np.abs(y[j]-y[idx[j,1:k+1]]), axis=0)
            hh[j] = np.log(np.prod(2*r*y_std))
            
    else:
        for j in range(n):
            r = np.max(np.abs(y[j]-y[idx[j,1:k+1]]), axis=0)
            hh[j] = np.log(np.prod(2*r))
        
    h = -spl.digamma(k)+spl.digamma(N)+(dim-1)/k+np.mean(hh)
    
    return h


def tksg(y, n=None, k=1, shuffle=True, rng=np.random):
    """
    Implements the KSG entropy estimation in m-dimensional case, as discribed by:
    Alexander Kraskov, Harald Stogbauer, and Peter Grassberger, "Estimating Mutual Information", Physical review E, 2004 
    """
    
    y = np.asarray(y, float)
    N, dim = y.shape
    
    if n is None:
        n = N
    else:
        n = min(n, N)
    
    # permute y
    if shuffle is True:
        rng.shuffle(y)
    
    # knn search
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='chebyshev').fit(y)
    dist, idx = nbrs.kneighbors(y)
    
    hh = np.empty(n)
    for j in range(n):
        r = np.max(np.abs(y[j]-y[idx[j,1:k+1]]), axis=0)
        lb = (y[j]-r >=0)*(y[j]-r) + (y[j]-r < 0)*0
        ub = (y[j]+r <=1)*(y[j]+r) + (y[j]+r > 1)*1
        hh[j] = np.log(np.prod(ub-lb))
        
    h = -spl.digamma(k)+spl.digamma(N)+(dim-1)/k+np.mean(hh)
    
    return h


def mi_ksg(y, dim_x, n=None, k=1, shuffle=True, rng=np.random):
    
    y = np.asarray(y, float)
    
    #standardize
    y = y/np.std(y, axis=0)
    
    N, dim = y.shape
    
    if n is None:
        n = N
    else:
        n = min(n, N)
    
    # permute y
    if shuffle is True:
        rng.shuffle(y)
    
    x1 = y[:, :dim_x]
    x2 = y[:, dim_x:]
    # knn search
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='chebyshev').fit(y)
    dist, idx = nbrs.kneighbors(y)
    
    nbrs_1 = NearestNeighbors(n_neighbors=N, algorithm='auto', metric='chebyshev').fit(x1)
    dist_1, idx_1 = nbrs_1.kneighbors(x1)
    
    nbrs_2 = NearestNeighbors(n_neighbors=N, algorithm='auto', metric='chebyshev').fit(x2)
    dist_2, idx_2 = nbrs_2.kneighbors(x2)
    
    n_x = np.empty(n)
    n_y = np.empty(n)
    for i in range(n):
        r_1 = np.max(np.abs(y[i,:dim_x]-y[idx[i,1:k+1],:dim_x]))
        r_2 = np.max(np.abs(y[i,dim_x:]-y[idx[i,1:k+1],dim_x:]))
        n_x[i] = np.sum(dist_1[i,1:] <= r_1)
        n_y[i] = np.sum(dist_2[i,1:] <= r_2)
    
    mi = spl.digamma(k)-1/k-np.mean(spl.digamma(n_x)+spl.digamma(n_y))+spl.digamma(N)
    
    return mi



def lnc(y, n=None, k=1, alpha=None, shuffle=True, rng=np.random):
    """
    Implements the Local Nonuniformity Correction (LNC) estimator in
    Shuyang Gao, Greg Ver Steeg, and Aram Galstyan, "Efficient Estimation of Mutual Information for
    Strongly Dependent Variables", AISTATS, 2015 
    """
    
    y = np.asarray(y, float)
    N, dim = y.shape
    
    # Determine alpha            
    if alpha is None:
        alpha = est_alpha_for_lnc(dim, k)
    
    if n is None:
        n = N
    else:
        n = min(n, N)
    
    # permute y
    if shuffle is True:
        rng.shuffle(y)
    
    # knn search
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='chebyshev').fit(y)
    dist, idx = nbrs.kneighbors(y)
    
    hh = np.empty(n)
    for j in range(n):
        y_loc = y[idx[j,:k+1]]
        l_edge = np.max(np.abs(y_loc[1:k+1]-y_loc[0]))
        logV = dim*np.log(2*l_edge)
        pca = PCA(n_components=dim, whiten = True)
        pca.fit(y_loc)
        y_loc = pca.transform(y_loc)
        l_edge = np.max(np.abs(y_loc[1:k+1]-y_loc[0]))
        logV_loc = np.log(np.prod(2*l_edge*np.sqrt(pca.explained_variance_)))
        if logV_loc-logV < np.log(alpha):
            hh[j] = logV_loc
        else:
            hh[j] = logV
        
    h = -spl.digamma(k)+spl.digamma(N)+np.mean(hh)
    
    return h


def mi_lnc(y, dim_x, n=None, k=1, alpha=None, shuffle=True, rng=np.random):
    
    y = np.asarray(y, float)
    N, dim = y.shape
    
    if n is None:
        n = N
    else:
        n = min(n, N)
    
    # permute y
    if shuffle is True:
        rng.shuffle(y)
    
    x1 = y[:, :dim_x]
    x2 = y[:, dim_x:]
    # knn search
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='chebyshev').fit(y)
    dist, idx = nbrs.kneighbors(y)
    
    nbrs_1 = NearestNeighbors(n_neighbors=N, algorithm='auto', metric='chebyshev').fit(x1)
    dist_1, idx_1 = nbrs_1.kneighbors(x1)
    
    nbrs_2 = NearestNeighbors(n_neighbors=N, algorithm='auto', metric='chebyshev').fit(x2)
    dist_2, idx_2 = nbrs_2.kneighbors(x2)
    
    n_x = np.empty(n)
    n_y = np.empty(n)
    LNC = np.empty(n)
    for i in range(n):
        y_loc = y[idx[i,:k+1]]
        r_1 = np.max(np.abs(y[i,:dim_x]-y[idx[i,1:k+1],:dim_x]))
        r_2 = np.max(np.abs(y[i,dim_x:]-y[idx[i,1:k+1],dim_x:]))
        logV = dim_x*np.log(2*r_1)+(dim-dim_x)*np.log(2*r_2)
        pca = PCA(n_components=dim, whiten = True)
        pca.fit(y_loc)
        y_loc = pca.transform(y_loc)
        l_edge = np.max(np.abs(y_loc[1:k+1]-y_loc[0]))
        logV_loc = np.log(np.prod(2*l_edge*np.sqrt(pca.explained_variance_)))
        if logV_loc-logV < np.log(alpha):
            LNC[i] = logV_loc-logV
        else:
            LNC[i] = 0.0
        
        n_x[i] = np.sum(dist_1[i,:] <= r_1)
        n_y[i] = np.sum(dist_2[i,:] <= r_2)
    
    mi = spl.digamma(k)-1/k-np.mean(spl.digamma(n_x)+spl.digamma(n_y))+spl.digamma(N)-np.mean(LNC)
    
    return mi


def est_alpha_for_lnc(dim, k, N=5e5, eps=5e-3, rng=np.random):
    N = int(N)
    a = np.empty(N)
    for i in range(N):
        y_loc = rng.rand(k, dim)
        pca = PCA(n_components=dim, whiten = True)
        pca.fit(y_loc)
        y_loc = pca.transform(y_loc)
        l_edge = np.max(y_loc)
        V_tilde = np.prod(2*l_edge*np.sqrt(pca.explained_variance_))
        a[i] = V_tilde/1
    return np.sort(a)[int(eps*N)]
    
    
    