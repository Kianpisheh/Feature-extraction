import numpy as np
import matplotlib.pyplot as plt

def show(data, n1=None, n2=None):
    
    # data is a ndarray
    n_features = data.shape[1]
    # determine the plots arrengments
    if n_features < 5:
        n1, n2 = n_features, 1

    for i in range(n_features):
        plt.subplot(n1,n2, i+1)
        plt.plot(data[:,i])
    plt.show()
