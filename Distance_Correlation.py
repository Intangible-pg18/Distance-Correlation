import numpy as np
from scipy.spatial.distance import pdist, squareform

def Distance_corr(X, Y): 
    
    X = np.array(X)[:, None]
    Y = np.array(Y)[:, None]

    if X.shape[0] != Y.shape[0]:
        raise ValueError('Found attributes of different lengths')
        
    dist_mat_X = squareform(pdist(X))
    dist_mat_Y = squareform(pdist(Y))
    
    cent_dist_mat_X = dist_mat_X - dist_mat_X.mean(axis=0)[None, :] - dist_mat_X.mean(axis=1)[:, None] + dist_mat_X.mean()
    cent_dist_mat_Y = dist_mat_Y - dist_mat_Y.mean(axis=0)[None, :] - dist_mat_Y.mean(axis=1)[:, None] + dist_mat_Y.mean()
    
    #to check if we got the correct centered distance matrices or not (all rows and all columns sum to zero)
    #ch = [cent_dist_mat_X.sum(axis=0), cent_dist_mat_X.sum(axis=1), cent_dist_mat_Y.sum(axis=0), cent_dist_mat_Y.sum(axis=1)]
    #for i in ch:
    #    for j in i:
    #        print(int(j))
    

    n = X.shape[0]
    dist_var_X = (cent_dist_mat_X * cent_dist_mat_X).sum() / float(n * n)
    dist_var_Y = (cent_dist_mat_Y * cent_dist_mat_Y).sum() / float(n * n)
    dist_covar = (cent_dist_mat_X * cent_dist_mat_Y).sum() / float(n * n)
    distance_correlation = np.sqrt(dist_covar) / np.sqrt(np.sqrt(dist_var_X) * np.sqrt(dist_var_Y))
    
    return distance_correlation
