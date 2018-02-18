def normalization(X):

    mu = X.mean(axis=0)
    std = X.std(axis=0)

    X_nor = (X - mu) / std
        # X[:,i] = (X[:,i] - mu) / np.std(X[:,i])
    return X_nor, mu, std
    