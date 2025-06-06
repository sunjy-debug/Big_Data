import numpy as np
import torch
from torch.distributions import Chi2, MultivariateNormal, StudentT
import math
from sklearn.metrics import silhouette_score, calinski_harabasz_score

class DPGMM:
    def __init__(self, X, alpha, nu0, lambda0, mu0, kappa0, device):
        # X: (N, D) data follow gaussian distribution (mu, sigma), mu, sigma both unknown
        # alpha: concentration parameter
        # G_0: base distribution for parameter
        # assume G_0 normal-inverse-wishart distribution for mu parameter (mu0, sigma0)
        # advantage: simple model, conjugate prior for efficient calculation
        
        self.X = torch.from_numpy(X).float().to(device)
        self.N, self.D = self.X.shape
        self.alpha = torch.tensor(alpha, device = device, dtype=self.X.dtype)
        self.lambda0 = torch.from_numpy(lambda0).float().to(device)
        self.mu0     = torch.from_numpy(mu0).float().to(device)
        self.nu0    = nu0
        self.kappa0 = kappa0
        self.device = device

        self.labels   = torch.zeros(self.N, device = self.device, dtype = torch.int) # the cluster labels of each data point
        self.clusters = {0: list(range(self.N))} # the data points belong to each cluster, we initialize that all data points belong to the same cluster

        # here we want to speed up the calculation of the matrix covariance
        # we instead of calculating the matrix covariance after one data point is removed, calculate three sufficient statistics
        self.stats = {}
        for idx, value in self.clusters.items():
            X = self.X[value]
            n = len(X)
            s = X.sum(dim = 0) # sum
            ss = X.T @ X # sum of squares
            self.stats[idx] = {'n': n, "s": s, "ss": ss}
            
        # initialize theta
        self.thetas = {}
        for idx in self.clusters.keys():
            self.thetas[idx] = self._sample_cluster_parameter(idx)
    
    def invwishart(self, df: int, scale: torch.Tensor) -> torch.Tensor:
        p = scale.shape[0]
        chi2 = torch.sqrt(Chi2(df - torch.arange(p)).sample().to(device = self.device, dtype = scale.dtype))
        A = torch.diag(chi2)
        A += torch.triu(torch.randn(p, p, device = self.device, dtype = scale.dtype), diagonal=1)
        L = torch.linalg.cholesky(torch.linalg.inv(scale))
        W = L @ A @ A.T @ L.T
        sigma = torch.linalg.inv(W)
        return sigma

    def multivariatenorm_logpdf(self, x: torch.Tensor, mu: torch.Tensor, invL: torch.tensor, logdet: torch.Tensor) -> torch.Tensor:
        D    = x.numel()
        z = torch.mv(invL, x - mu)
        return -0.5 * (D * math.log(2.0 * math.pi) + logdet + torch.dot(z, z))
    
    def multivariatet_logpdf(self, x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, df: float) -> torch.Tensor:
        L = torch.linalg.cholesky(sigma)
        z = torch.linalg.solve_triangular(L, (x - mu).unsqueeze(-1), upper = False).squeeze(-1)
        t = StudentT(df, torch.tensor(0., device = self.device, dtype = x.dtype), torch.tensor(1., device = self.device, dtype = x.dtype))
        logpdf_z = t.log_prob(z).sum(-1)
        logdet = 2.0 * torch.log(torch.diag(L)).sum()
        return logpdf_z - 0.5 * logdet
    
    def _transform_psd(self, matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        matrix = (matrix + matrix.transpose(-1, -2)) / 2
        eigval, eigvec = torch.linalg.eigh(matrix)
        eigval = torch.clamp(eigval, min = eps * eigval.max(dim = -1, keepdim = True).values)
        matrix = eigvec @ (eigval[..., None] * eigvec.transpose(-1, -2))
        return matrix
    
    def _sample_cluster_parameter(self, idx):
        # normal-inverse-wishart conjugate prior, gaussian likelihood
        n = self.stats[idx]['n']
        X_mean = self.stats[idx]['s'] / self.stats[idx]['n'] if n > 0 else torch.zeros(self.D, device = self.device)
        X_var = (self.stats[idx]['ss'] - n * torch.outer(self.stats[idx]['s'] / self.stats[idx]['n'], self.stats[idx]['s'] / self.stats[idx]['n'])) \
            if n > 1  else torch.zeros((self.D, self.D), device = self.device)
        nun = self.nu0 + n
        kappan = self.kappa0 + n
        mun = (self.mu0 * self.kappa0 + X_mean * n) / kappan
        lambdan = self.lambda0 + X_var + self.kappa0 * n / kappan * torch.outer(X_mean - self.mu0, X_mean - self.mu0)

        sigma = self.invwishart(nun, lambdan) # sigma follows inv-wishart(nu_n, lambda_n)
        # due to computational error, sigma might not be a symmetric positive-definite matrix
        # here we do the transformation
        sigma = self._transform_psd(sigma)
        sigma_mu = sigma / kappan
        sigma_mu = self._transform_psd(sigma_mu)
        mu = MultivariateNormal(mun, sigma_mu).sample()
        L = torch.linalg.cholesky(sigma)
        invL = torch.cholesky_inverse(L)
        logdet = 2.0 * torch.log(torch.diag(L)).sum()

        return {'mu': mu, 'sigma': sigma, 'L': L, "invL": invL, 'logdet': logdet}

    
    def sample(self, iterations):
        D = self.D
        for itr in range(iterations):
            print(f"{itr + 1}-th iteration has beginned.\n")
            
            cluster_changed = set() # we monitor the changed clusters and resample the parameters for the changed clusters
           
            for i in range(self.N):
                # reassign the data point
                # remove i from its cluster
                cluster_i = int(self.labels[i].item()) # the cluster data point i used to belong to
                self.clusters[cluster_i].remove(i) # remove the data point from the cluster
                self.stats[cluster_i]['n'] -= 1 # remove the sufficient statistics from the cluster parameters
                self.stats[cluster_i]['s'] -= self.X[i]
                self.stats[cluster_i]['ss'] -= torch.outer(self.X[i], self.X[i])
                if self.stats[cluster_i]['n'] == 0:
                    del self.clusters[cluster_i] # if the cluster data point i used to belong to is empty, we delete it
                    del self.stats[cluster_i]
                    del self.thetas[cluster_i]
                cluster_changed.add(cluster_i)

                # calculating the resampling probability of k(i)
                # log_prob, the probability k(i) belongs to each cluster
                log_probs = []
                for idx, value in self.clusters.items():
                    theta = self.thetas[idx]
                    mu, sigma, L, invL, logdet = theta['mu'], theta['sigma'], theta['L'], theta['invL'], theta['logdet']
                    n = torch.tensor(len(value), device = self.device, dtype = self.X.dtype)
                    log_probs.append(torch.log(n) + self.multivariatenorm_logpdf(self.X[i], mu, invL, logdet)) # the probability of existing clusters
                df_new = self.nu0 - D + 1
                mu_new = self.mu0
                sigma_new = (self.kappa0 + 1) / (self.kappa0 * df_new) * self.lambda0
                log_probs.append(torch.log(self.alpha) + self.multivariatet_logpdf(self.X[i], mu_new, sigma_new, df_new)) # the probability of a new cluster
                log_probs = torch.stack(log_probs, dim = 0)
                # the intergral of gaussian(x_i| mu_k, sigma_k) * gaussian(mu_k| mu_0, sigma / kappa_0) * inv-wishart(sigma_k| nu_0, lambda_0)
                # is a multivariate-t distribution, with df = nu_0 - d + 1, loc = mu_0, scale = (kappa_0 + 1) / (kappa_0 * (nu_0 - d + 1)) * lambda_0

                # standarization - softmax
                prob = torch.softmax(log_probs, dim = 0)

                # resample k(i)
                choice = torch.multinomial(prob, 1).item()
                cluster_idxs = list(self.clusters.keys())
                # if the choice belongs to a new cluster
                if choice == len(cluster_idxs): # since the index begins with 0
                    new_idx = max(self.clusters.keys(), default = -1) + 1
                    self.clusters[new_idx] = [i]
                    self.stats[new_idx] = {
                        'n':  1,
                        's':  self.X[i],
                        'ss': torch.outer(self.X[i], self.X[i])
                        }
                    self.thetas[new_idx] = self._sample_cluster_parameter(new_idx)
                    self.labels[i] = torch.tensor(new_idx, device = self.device, dtype = torch.long)
                    cluster_changed.add(new_idx)
                else:
                    idx = cluster_idxs[choice]
                    self.clusters[idx].append(i)
                    self.stats[idx]['n'] += 1
                    self.stats[idx]['s'] += self.X[i]
                    self.stats[idx]['ss'] += torch.outer(self.X[i], self.X[i])
                    self.labels[i] = torch.tensor(idx, device = self.device, dtype = torch.long)
                    cluster_changed.add(idx)          

            # resample cluster parameter for changed clusters
            for idx in cluster_changed:
                if idx in self.clusters: 
                    if len(self.clusters[idx]) > 0:
                        self.thetas[idx] = self._sample_cluster_parameter(idx)
                    else:
                        del self.clusters[idx]
                        del self.thetas[idx]

            print(f"{itr + 1}-th iteration has completed.\n")

        # save the labels
        labels_path = "alg1_out.txt"
        np.savetxt(labels_path, self.labels.cpu().numpy(), fmt='%d')
        print(f"DPMM labels written to {labels_path}.\n")

        #print the cluster summary
        cluster_sizes = {idx: len(value) for idx, value in self.clusters.items()}
        for idx, size in sorted(cluster_sizes.items()):
            print(f"Cluster {idx}: {size} points")
        
        # evaluation
        s_score = silhouette_score(self.X.cpu().numpy(), self.labels.cpu().numpy(), metric='euclidean')
        print(f"Silhouette Score: {s_score:.4f}")
        ch_score = calinski_harabasz_score(self.X.cpu().numpy(), self.labels.cpu().numpy())
        print(f"Calinski-Harabasz Index: {ch_score:.4f}")
