import torch
from torch.distributions import Chi2, MultivariateNormal, StudentT


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

        self.labels = torch.arange(self.N, device = self.device, dtype = torch.long) # the cluster labels of each data point
        self.clusters = {i:[i] for i in range(self.N)} # the data points belong to each cluster

        # initialize theta
        self.thetas = {}
        for idx, value in self.clusters.items():
            self.thetas[idx] = self._resample_cluster_parameter(value)
    
    def invwishart(self, df: int, scale: torch.Tensor) -> torch.Tensor:
        p = scale.shape[0]
        A = torch.zeros((p, p), device = self.device, dtype = scale.dtype)
        for i in range(p):
            chi2 = Chi2(df - i).sample().to(device = self.device, dtype = scale.dtype)
            A[i, i] = torch.sqrt(chi2)
            if i > 0:
                A[i, :i] = torch.randn(i, device = self.device, dtype = scale.dtype)

        L = torch.linalg.cholesky(torch.linalg.inv(scale))
        W = L @ A @ A.T @ L.T
        sigma = torch.linalg.inv(W)
        return sigma
    
    def multivariatet_logpdf(self, x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, df: float) -> torch.Tensor:
        L = torch.linalg.cholesky(sigma)
        z = torch.linalg.solve_triangular(L, (x - mu).T, upper = False).squeeze(-1)
        t = StudentT(df, torch.zeros((), device = self.device, dtype = torch.long), torch.ones((), device = self.device, dtype = torch.long))
        logpdf_z = t.log_prob(z).sum(-1)
        logdet = 2.0 * torch.log(torch.diag(L)).sum()
        return logpdf_z - 0.5 * logdet
    
    def _transform_psd(self, matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        matrix = (matrix + matrix.transpose(-1, -2)) / 2
        eigval, eigvec = torch.linalg.eigh(matrix)
        eigval = torch.clamp(eigval, min = eps * eigval.max(dim = -1, keepdim = True).values)
        matrix = eigvec @ (eigval[..., None] * eigvec.transpose(-1, -2))
        return matrix
    
    def _resample_cluster_parameter(self, idx):
        # normal-inverse-wishart conjugate prior, gaussian likelihood
        X = self.X[idx]
        n, _ = X.shape
        X_mean = X.mean(axis = 0) if n > 0 else torch.zeros(self.D, device = self.device, dtype = torch.float)
        X_var = torch.cov(X.T) if n > 1  else torch.zeros((self.D, self.D), device=self.device, dtype = torch.float)
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

        return mu, sigma

    
    def _reassign_data_to_cluster(self, iterations = 1000):
        D = self.D
        for itr in range(iterations):
            for i in range(self.N):
                # remove i from its cluster
                cluster_i = int(self.labels[i].item()) # the cluster data point i used to belong to
                self.clusters[cluster_i].remove(i)
                if len(self.clusters[cluster_i]) == 0:
                    del self.clusters[cluster_i] # if the cluster data point i used to belong to is empty, we delete it
                    del self.thetas[cluster_i]

                # calculating the resampling probability of k(i)
                log_probs = [] # the probability k(i) belongs to each cluster
                cluster_idxs = torch.tensor(list(self.clusters.keys()), device = self.device, dtype = torch.long) # the clusters
                for idx, value in self.clusters.items():
                    mu, sigma = self.thetas[idx]
                    n = torch.tensor(len(value), device = self.device, dtype = torch.long)
                    log_probs.append(torch.log(n) + MultivariateNormal(mu, sigma).log_prob(self.X[i])) # the probability of existing clusters
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
                # if the choice belongs to a new cluster
                if choice == K: # since the index begins with 0
                    new_idx = max(self.clusters.keys(), default = -1) + 1
                    self.clusters[new_idx] = [i]
                    self.thetas[new_idx] = self._resample_cluster_parameter(self.clusters[new_idx])
                    self.labels[i] = torch.tensor(new_idx, device = self.device, dtype = torch.float)
                else:
                    idx = cluster_idxs[choice]
                    self.clusters[idx].append(i)
                    self.thetas[idx] = self._resample_cluster_parameter(self.clusters[idx])
                    self.labels[i] = torch.tensor(idx, device = self.device, dtype = torch.float)            

            for idx, value in self.clusters.items():
                self.thetas[idx] = self._resample_cluster_parameter(value)

            # in order to monitor the process, here we print the procedure
            if (itr + 1) % 50 == 0:
                print(f"Iteration {itr + 1: 4d}: Number of clusters = {len(self.clusters)}\n")

        return self.labels, self.clusters
