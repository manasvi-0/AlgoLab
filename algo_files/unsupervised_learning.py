import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

def unsupervised():
    st.header("🧭 Unsupervised Learning Algorithms")
    
    # Algorithm selection
    algorithm = st.selectbox(
        "Choose Algorithm:",
        ["K-Means Clustering", "DBSCAN", "PCA (Dimensionality Reduction)", "Hierarchical Clustering"]
    )
    
    # Generate sample data
    if st.button("Generate Sample Data"):
        if algorithm in ["K-Means Clustering", "DBSCAN", "Hierarchical Clustering"]:
            X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
        else:  # PCA
            X, _ = make_blobs(n_samples=300, centers=4, n_features=4, random_state=0)
        
        st.session_state.unsupervised_data = X
        st.success("Sample data generated!")
    
    if 'unsupervised_data' in st.session_state:
        X = st.session_state.unsupervised_data
        
        if algorithm == "K-Means Clustering":
            st.subheader("K-Means Clustering")
            
            # Parameters
            n_clusters = st.slider("Number of Clusters", 2, 8, 3)
            
            if st.button("Run K-Means"):
                # Apply K-Means
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(X)
                centers = kmeans.cluster_centers_
                
                # Visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
                ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
                ax.set_title(f'K-Means Clustering (k={n_clusters})')
                ax.legend()
                plt.colorbar(scatter)
                st.pyplot(fig)
                
                # Metrics
                st.write(f"**Inertia (Within-cluster sum of squares):** {kmeans.inertia_:.2f}")
        
        elif algorithm == "DBSCAN":
            st.subheader("DBSCAN Clustering")
            
            # Parameters
            eps = st.slider("Epsilon (neighborhood radius)", 0.1, 2.0, 0.5, 0.1)
            min_samples = st.slider("Minimum samples", 2, 20, 5)
            
            if st.button("Run DBSCAN"):
                # Apply DBSCAN
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X)
                
                # Visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                unique_labels = set(labels)
                colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
                
                for k, col in zip(unique_labels, colors):
                    if k == -1:
                        col = [0, 0, 0, 1]  # Black for noise
                    
                    class_member_mask = (labels == k)
                    xy = X[class_member_mask]
                    ax.scatter(xy[:, 0], xy[:, 1], c=[col], alpha=0.7, 
                              label=f'Cluster {k}' if k != -1 else 'Noise')
                
                ax.set_title(f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})')
                ax.legend()
                st.pyplot(fig)
                
                # Metrics
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                st.write(f"**Number of clusters:** {n_clusters}")
                st.write(f"**Number of noise points:** {n_noise}")
        
        elif algorithm == "PCA (Dimensionality Reduction)":
            st.subheader("Principal Component Analysis (PCA)")
            
            if X.shape[1] < 3:
                st.warning("PCA works best with higher dimensional data. Generating 4D sample data...")
                X, _ = make_blobs(n_samples=300, centers=4, n_features=4, random_state=0)
                st.session_state.unsupervised_data = X
            
            # Parameters
            n_components = st.slider("Number of Components", 2, min(X.shape[1], 4), 2)
            
            if st.button("Run PCA"):
                # Standardize the data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Apply PCA
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X_scaled)
                
                # Visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Original data (first 2 dimensions)
                ax1.scatter(X[:, 0], X[:, 1], alpha=0.7)
                ax1.set_title('Original Data (First 2 Dimensions)')
                ax1.set_xlabel('Feature 1')
                ax1.set_ylabel('Feature 2')
                
                # PCA transformed data
                ax2.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, c='red')
                ax2.set_title('PCA Transformed Data')
                ax2.set_xlabel('First Principal Component')
                ax2.set_ylabel('Second Principal Component')
                
                st.pyplot(fig)
                
                # Explained variance
                st.write("**Explained Variance Ratio:**")
                for i, ratio in enumerate(pca.explained_variance_ratio_):
                    st.write(f"PC{i+1}: {ratio:.3f} ({ratio*100:.1f}%)")
                st.write(f"**Total Explained Variance:** {sum(pca.explained_variance_ratio_):.3f} ({sum(pca.explained_variance_ratio_)*100:.1f}%)")
        
        elif algorithm == "Hierarchical Clustering":
            st.subheader("Hierarchical Clustering")
            
            # Parameters
            n_clusters = st.slider("Number of Clusters", 2, 8, 3)
            linkage_method = st.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
            
            if st.button("Run Hierarchical Clustering"):
                # Apply Hierarchical Clustering
                hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
                labels = hierarchical.fit_predict(X)
                
                # Create linkage matrix for dendrogram
                linkage_matrix = linkage(X, method=linkage_method)
                
                # Visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Cluster visualization
                scatter = ax1.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
                ax1.set_title(f'Hierarchical Clustering ({linkage_method} linkage)')
                plt.colorbar(scatter, ax=ax1)
                
                # Dendrogram
                dendrogram(linkage_matrix, ax=ax2, truncate_mode='level', p=5)
                ax2.set_title('Dendrogram')
                ax2.set_xlabel('Sample Index')
                ax2.set_ylabel('Distance')
                
                st.pyplot(fig)
                
                st.write(f"**Number of clusters:** {n_clusters}")
                st.write(f"**Linkage method:** {linkage_method}")
    
    else:
        st.info("Click 'Generate Sample Data' to start exploring unsupervised learning algorithms!")
        
        # Show algorithm descriptions
        st.subheader("Algorithm Overview")
        
        if algorithm == "K-Means Clustering":
            st.write("""
            **K-Means** partitions data into k clusters by minimizing within-cluster sum of squares.
            - **Use case:** Market segmentation, image compression
            - **Key parameter:** Number of clusters (k)
            """)
        
        elif algorithm == "DBSCAN":
            st.write("""
            **DBSCAN** groups together points in high-density areas and marks outliers in low-density areas.
            - **Use case:** Anomaly detection, clustering irregular shapes
            - **Key parameters:** Epsilon (neighborhood radius), minimum samples
            """)
        
        elif algorithm == "PCA (Dimensionality Reduction)":
            st.write("""
            **PCA** reduces dimensionality by finding principal components that explain maximum variance.
            - **Use case:** Data visualization, feature reduction, noise reduction
            - **Key parameter:** Number of components to retain
            """)
        
        elif algorithm == "Hierarchical Clustering":
            st.write("""
            **Hierarchical Clustering** creates a tree of clusters using linkage criteria.
            - **Use case:** Taxonomy creation, social network analysis
            - **Key parameters:** Number of clusters, linkage method
            """)

if __name__ == "__main__":
    st.set_page_config(page_title="Unsupervised Learning", page_icon="🧭")
    st.title("🧭 Unsupervised Learning Algorithms")
    unsupervised()