This project implements and compares two clustering techniques on high-dimensional, sparse datasets:

A self-implemented Dirichlet Process Gaussian Mixture Model (DPGMM) using Gibbs sampling

A custom BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) model

A comparison with sklearn's built-in BayesianGaussianMixture and Birch clustering methods

File Structure
.
├── project-big-data.ipynb   # Implement in Kaggle Notebook
├── main.py                  # Main entry point
├── PCA.py                   # PCA exploration
├── DPMM.py                  # Self-implemented DPGMM with Gibbs sampler
├── BIRCHM.py                # Custom BIRCH clustering module
├── outputs/                 # Output directory for cluster labels
├── requirements.txt         # Python package dependencies
└── README.md                # This file

How to Run
python main.py --model DPGMM --iters 1 --alpha 5.0
python main.py --model BIRCH --threshold 0.5 --nclusters 50

Dependencies
pip install -r requirements.txt