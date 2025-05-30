import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse
from DPMM import DPGMM

def main():
    parser = argparse.ArgumentParser(
        description="Run DPGMM with normal-inverse-whishart prior distribution"
    )
    parser.add_argument("--device", type = str, default = "cuda", help = "Device CUDA/CPU")
    parser.add_argument("--pcacomponents", type = int, default = 325, help = "Number of components for PCA")
    parser.add_argument("--alpha", type = float, default = 1.0, help = "DP concentration parameter")
    parser.add_argument("--iters", type = int, default = 500, help="Number of Gibbs sampling iterations")
    parser.add_argument("--seed", type = int, default = 0, help = "Random seed")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    X = np.load("data600.npy")

    # standardization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)    

    # due to computation efficiency, we consider dimension reduction with PCA
    pca = PCA(n_components = args.pcacomponents)
    X = pca.fit_transform(X)
    N, D = X.shape

    model  = DPGMM(X, alpha = args.alpha, nu0 = D + 2, lambda0 = np.eye(D, D), mu0 = np.zeros(D), kappa0 = 1, device = args.device)
    # nu_0 = D + 2 ensures that the expecation of covariance exists
    labels, clusters = model._reassign_data_to_cluster(iterations = args.iters)

    # visualization
    plt.scatter(X.cpu().numpy(), np.zeros_like(X.cpu().numpy()), c = labels.cpu().numpy(), s = 10, cmap = "tab10")
    plt.yticks([])
    plt.xlabel("x")
    plt.title(f"DPGMM Clustering with Normal-inv-Whishart (K={len(set(labels))})")
    plt.show()
    plt.savefig("DPGMM Clustering with Normal-inv-Whishart.png")

if __name__ == "__main__":
    main()
