import numpy as np
from sklearn.decomposition import PCA

X = np.load("data600.npy")
# due to computation efficiency, we consider dimension reduction with PCA
pca = PCA()
X = pca.fit(X)
cumvar = np.cumsum(pca.explained_variance_ratio_)
d = np.searchsorted(cumvar, 0.95) + 1
print(f"In order to maintin 95% variance, we need to reduce the dimension to {d}.\n")