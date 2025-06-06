import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from DPMM import DPGMM
from BIRCHM import BIRCH

def main():
    parser = argparse.ArgumentParser(
        description="Run DPGMM with normal-inverse-whishart prior distribution"
    )
    parser.add_argument("--device", type = str, default = "cuda", help = "Device CUDA/CPU")
    parser.add_argument("--model", type = str, default = "DPGMM", help = "Model")
    parser.add_argument("--pcacomponents", type = int, default = 325, help = "Number of components for PCA")
    parser.add_argument("--alpha", type = float, default = 5.0, help = "DP concentration parameter, when alpha = 1.0, the expecation of number of cluster = lnN")
    parser.add_argument("--iters", type = int, default = 10, help="Number of Gibbs sampling iterations")
    parser.add_argument("--seed", type = int, default = 0, help = "Random seed")
    parser.add_argument("--threshold", type = float, default= 10, help = "Radius Threshold for Clusters")
    parser.add_argument("--B", type = int, default = 2000, help = "Node Threshold for Non-leaf Node")
    parser.add_argument("--L", type = int, default = 2000, help = "Node Threshold for Leaf Node")
    parser.add_argument("--nclusters", type = int, default = 5, help = "Number of Clustering")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    X = np.load("data600.npy")

    # standardization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)    

    if args.model == "DPGMM":
        # due to computation efficiency, we consider dimension reduction with PCA
        pca = PCA(n_components = args.pcacomponents)
        X = pca.fit_transform(X)
        N, D = X.shape
        model  = DPGMM(X, alpha = args.alpha, nu0 = D + 2, lambda0 = np.eye(D) * 0.1, mu0 = np.zeros(D), kappa0 = 0.01, device = args.device)
        # nu_0 = D + 2 ensures that the expecation of covariance exists
        model.sample(iterations = args.iters)
    if args.model == "BIRCH":
        model = BIRCH(threshold = args.threshold, B = args.B, L = args.L, nclusters = args.nclusters, seed = args.seed)
        for x in X:
            model.data_insertion(x)
        print(f"The data has been inserted.\n")
        # limit the number of clusters
        model.global_clustering(n_clusters = args.nclusters)
        model.sample(X)

if __name__ == "__main__":
    main()
