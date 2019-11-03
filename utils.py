import numpy as np
from scipy import linalg

def make_pos_definite(C, n_iter=200, a=0.001):
    """Convert the input matrix into positive definite
          Algorithm: Dykstra's Correction
    """
    if C.ndim != 3:
        C = np.expand_dims(C, axis=0)
    C_pos = np.zeros(C.shape)
    for c in range(C.shape[0]):
        cov = C[c]
        delta_S = 0
        for i in range(n_iter):
            if np.all(linalg.eigvalsh(cov) > 0):
                C_pos[c,:,:] = cov
                break
            R = cov - delta_S
            l, V = linalg.eig(R)
            L = np.diag(l)
            L[L <= 0] = a  # replace negetive eigen vals with a pos value
            cov = np.dot(np.dot(V, L), V.T)
            delta_S = cov - R
            np.fill_diagonal(cov, 1) # in-place

    return C_pos

def calc_covars(X, labels, n_clusters):
    n_features = X.shape[1]
    covs = np.zeros((n_clusters, n_features, n_features))
    for c in range(n_clusters):
        cluster_samples = X[labels==c,:].T
        if cluster_samples.shape[1] == 1:
            cluster_samples = np.tile(cluster_samples, (1,2))
        covs[c] = np.cov(cluster_samples)

    return covs
