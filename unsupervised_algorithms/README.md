# Unsupervised Learning Algorithms

This module implements 7 unsupervised machine learning algorithms with interactive visualizations.

## 🧭 Algorithms Implemented

### Clustering Algorithms
- **K-Means** - Partitions data into k clusters by minimizing within-cluster sum of squares
- **DBSCAN** - Density-based clustering that identifies outliers
- **Hierarchical Clustering** - Creates tree of clusters using linkage criteria
- **Gaussian Mixture Model** - Soft clustering using probability distributions
- **Spectral Clustering** - Uses eigenvalues for non-convex cluster shapes

### Dimensionality Reduction
- **PCA** - Principal Component Analysis for linear dimensionality reduction
- **t-SNE** - Non-linear dimensionality reduction preserving local structure

## 📁 Files

- `unsupervised_module.py` - Main implementation with all algorithms
- `README.md` - This documentation file

## 🎮 Usage

Each algorithm has two modes:
- **Overview** - Theory and use cases
- **Playground** - Interactive parameter tuning and visualization

### Data Sources
1. **Upload CSV** - Use your own dataset from sidebar
2. **Generate Sample** - Create synthetic data for each algorithm

### Key Features
- Interactive parameter controls (sliders, selectboxes)
- Real-time visualizations using matplotlib
- Algorithm-specific metrics and insights
- Session state management for data persistence

## 🔧 Integration

Import in `app.py`:
```python
from unsupervised_algorithms.unsupervised_module import unsupervised
unsupervised()
```

## 📊 Algorithm Details

| Algorithm | Best For | Key Parameters | Output |
|-----------|----------|----------------|---------|
| K-Means | Spherical clusters | k (clusters) | Centroids + Inertia |
| DBSCAN | Irregular shapes | eps, min_samples | Noise detection |
| Hierarchical | Tree structure | linkage method | Dendrogram |
| GMM | Overlapping clusters | n_components | AIC/BIC scores |
| Spectral | Non-convex shapes | n_clusters | Eigenvalue-based |
| PCA | Linear reduction | n_components | Explained variance |
| t-SNE | Visualization | perplexity | 2D embedding |

## 🛠️ Dependencies

- scikit-learn
- matplotlib
- numpy
- pandas
- scipy
- streamlit