import numpy as np

def replaceNAN(X): # assume that we receive a numpy.ndarray
    means = np.nanmean(a=X, axis=0) # we have the variables on the columns
    pos = np.where(np.isnan(X))
    X[pos] = means[pos[1]]
    return X