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
    parser.add_argument("--model", type = str, default = "DPGMM", help = "Model")
    parser.add_argument("--pcacomponents", type = int, default = 325, help = "Number of components for PCA")
    parser.add_argument("--alpha", type = float, default = 1.0, help = "DP concentration parameter, when alpha = 1.0, the expecation of number of cluster = lnN")
    parser.add_argument("--iters", type = int, default = 10, help="Number of Gibbs sampling iterations")
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

    if args.model == "DPGMM":
        model  = DPGMM(X, alpha = args.alpha, nu0 = D + 2, lambda0 = np.eye(D, D), mu0 = np.zeros(D), kappa0 = 1, device = args.device)
        # nu_0 = D + 2 ensures that the expecation of covariance exists
        labels, clusters = model._reassign_data_to_cluster(iterations = args.iters)
        
        # save the labels
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok = True)
        labels_path = out_dir / "dpgmm_labels.csv"
        np.savetxt(labels_path, np.column_stack([np.arange(len(labels)), labels.cpu().numpy()]), fmt='%d', delimiter=',', header='index,label', comments='')
        print(f"DPGMM labels written to {labels_path}.\n")

        #print the cluster summary
        cluster_sizes = {idx: len(value) for idx, value in clusters.items()}
        for idx, size in sorted(cluster_sizes.items()):
            print(f"Cluster {idx}: {size} points")


if __name__ == "__main__":
    main()
