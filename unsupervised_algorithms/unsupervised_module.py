import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA, TSNE
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

def unsupervised():
    st.write("Unsupervised Learning")
    
    options = ["K-Means", "DBSCAN", "PCA", "Hierarchical Clustering", "Gaussian Mixture", "Spectral Clustering", "t-SNE"]
    selected_option = st.selectbox("Choose an algorithm:", options)
    
    st.write("You have selected:", selected_option)
    
    if selected_option == "K-Means":
        view = st.radio("Choose View", ["K-Means Overview", "K-Means Playground"])
        if view == "K-Means Overview":
            st.subheader("K-Means Clustering Overview")
            st.write("**K-Means** partitions data into k clusters by minimizing within-cluster sum of squares.")
            st.write("- **Use case:** Market segmentation, image compression")
            st.write("- **Key parameter:** Number of clusters (k)")
        elif view == "K-Means Playground":
            kmeans_playground()
    
    elif selected_option == "DBSCAN":
        view = st.radio("Choose View", ["DBSCAN Overview", "DBSCAN Playground"])
        if view == "DBSCAN Overview":
            st.subheader("DBSCAN Overview")
            st.write("**DBSCAN** groups together points in high-density areas and marks outliers.")
            st.write("- **Use case:** Anomaly detection, clustering irregular shapes")
            st.write("- **Key parameters:** Epsilon (neighborhood radius), minimum samples")
        elif view == "DBSCAN Playground":
            dbscan_playground()
    
    elif selected_option == "PCA":
        view = st.radio("Choose View", ["PCA Overview", "PCA Playground"])
        if view == "PCA Overview":
            st.subheader("PCA Overview")
            st.write("**PCA** reduces dimensionality by finding principal components that explain maximum variance.")
            st.write("- **Use case:** Data visualization, feature reduction, noise reduction")
            st.write("- **Key parameter:** Number of components to retain")
        elif view == "PCA Playground":
            pca_playground()
    
    elif selected_option == "Hierarchical Clustering":
        view = st.radio("Choose View", ["Hierarchical Overview", "Hierarchical Playground"])
        if view == "Hierarchical Overview":
            st.subheader("Hierarchical Clustering Overview")
            st.write("**Hierarchical Clustering** creates a tree of clusters using linkage criteria.")
            st.write("- **Use case:** Taxonomy creation, social network analysis")
            st.write("- **Key parameters:** Number of clusters, linkage method")
        elif view == "Hierarchical Playground":
            hierarchical_playground()
    
    elif selected_option == "Gaussian Mixture":
        view = st.radio("Choose View", ["GMM Overview", "GMM Playground"])
        if view == "GMM Overview":
            st.subheader("Gaussian Mixture Model Overview")
            st.write("**GMM** assumes data comes from a mixture of Gaussian distributions.")
            st.write("- **Use case:** Soft clustering, density estimation")
            st.write("- **Key parameter:** Number of components")
        elif view == "GMM Playground":
            gmm_playground()
    
    elif selected_option == "Spectral Clustering":
        view = st.radio("Choose View", ["Spectral Overview", "Spectral Playground"])
        if view == "Spectral Overview":
            st.subheader("Spectral Clustering Overview")
            st.write("**Spectral Clustering** uses eigenvalues of similarity matrix.")
            st.write("- **Use case:** Non-convex clusters, image segmentation")
            st.write("- **Key parameter:** Number of clusters")
        elif view == "Spectral Playground":
            spectral_playground()
    
    elif selected_option == "t-SNE":
        view = st.radio("Choose View", ["t-SNE Overview", "t-SNE Playground"])
        if view == "t-SNE Overview":
            st.subheader("t-SNE Overview")
            st.write("**t-SNE** reduces dimensionality while preserving local structure.")
            st.write("- **Use case:** Data visualization, exploratory analysis")
            st.write("- **Key parameter:** Perplexity")
        elif view == "t-SNE Playground":
            tsne_playground()

def kmeans_playground():
    st.subheader("K-Means Clustering Playground")
    
    # Check if data exists from sidebar upload
    X = None
    if 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            X = df[numeric_cols[:2]].values
            st.info(f"Using uploaded data: {X.shape[0]} samples, 2 features")
    
    if st.button("Generate Sample Data", key="kmeans_data"):
        X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
        st.session_state.kmeans_data = X
        st.success("Sample data generated!")
    
    if 'kmeans_data' in st.session_state and X is None:
        X = st.session_state.kmeans_data
    
    if X is not None:
        n_clusters = st.slider("Number of Clusters", 2, 8, 3)
        
        if st.button("Run K-Means"):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(X)
            centers = kmeans.cluster_centers_
            
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
            ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
            ax.set_title(f'K-Means Clustering (k={n_clusters})')
            ax.legend()
            plt.colorbar(scatter)
            st.pyplot(fig)
            st.write(f"**Inertia:** {kmeans.inertia_:.2f}")

def dbscan_playground():
    st.subheader("DBSCAN Playground")
    
    # Check if data exists from sidebar upload
    X = None
    if 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            X = df[numeric_cols[:2]].values
            st.info(f"Using uploaded data: {X.shape[0]} samples, 2 features")
    
    if st.button("Generate Sample Data", key="dbscan_data"):
        X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
        st.session_state.dbscan_data = X
        st.success("Sample data generated!")
    
    if 'dbscan_data' in st.session_state and X is None:
        X = st.session_state.dbscan_data
    
    if X is not None:
        eps = st.slider("Epsilon", 0.1, 2.0, 0.5, 0.1)
        min_samples = st.slider("Minimum samples", 2, 20, 5)
        
        if st.button("Run DBSCAN"):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            unique_labels = set(labels)
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
            
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    col = [0, 0, 0, 1]
                class_member_mask = (labels == k)
                xy = X[class_member_mask]
                ax.scatter(xy[:, 0], xy[:, 1], c=[col], alpha=0.7, 
                          label=f'Cluster {k}' if k != -1 else 'Noise')
            
            ax.set_title(f'DBSCAN (eps={eps}, min_samples={min_samples})')
            ax.legend()
            st.pyplot(fig)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            st.write(f"**Clusters:** {n_clusters}")
            st.write(f"**Noise points:** {n_noise}")

def pca_playground():
    st.subheader("PCA Playground")
    
    # Check if data exists from sidebar upload
    X = None
    if 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            X = df[numeric_cols].values
            st.info(f"Using uploaded data: {X.shape[0]} samples, {X.shape[1]} features")
    
    if st.button("Generate Sample Data", key="pca_data"):
        X, _ = make_blobs(n_samples=300, centers=4, n_features=4, random_state=0)
        st.session_state.pca_data = X
        st.success("4D sample data generated!")
    
    if 'pca_data' in st.session_state and X is None:
        X = st.session_state.pca_data
    
    if X is not None:
        n_components = st.slider("Number of Components", 2, min(X.shape[1], 4), 2)
        
        if st.button("Run PCA"):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            ax1.scatter(X[:, 0], X[:, 1], alpha=0.7)
            ax1.set_title('Original Data')
            
            ax2.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, c='red')
            ax2.set_title('PCA Transformed')
            ax2.set_xlabel('PC1')
            ax2.set_ylabel('PC2')
            
            st.pyplot(fig)
            
            st.write("**Explained Variance:**")
            for i, ratio in enumerate(pca.explained_variance_ratio_):
                st.write(f"PC{i+1}: {ratio:.3f} ({ratio*100:.1f}%)")

def hierarchical_playground():
    st.subheader("Hierarchical Clustering Playground")
    
    # Check if data exists from sidebar upload
    X = None
    if 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            X = df[numeric_cols[:2]].values
            st.info(f"Using uploaded data: {X.shape[0]} samples, 2 features")
    
    if st.button("Generate Sample Data", key="hier_data"):
        X, _ = make_blobs(n_samples=100, centers=4, cluster_std=0.60, random_state=0)
        st.session_state.hier_data = X
        st.success("Sample data generated!")
    
    if 'hier_data' in st.session_state and X is None:
        X = st.session_state.hier_data
    
    if X is not None:
        n_clusters = st.slider("Number of Clusters", 2, 8, 3)
        linkage_method = st.selectbox("Linkage Method", ["ward", "complete", "average"])
        
        if st.button("Run Hierarchical Clustering"):
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
            labels = hierarchical.fit_predict(X)
            
            linkage_matrix = linkage(X, method=linkage_method)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            scatter = ax1.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
            ax1.set_title(f'Hierarchical Clustering ({linkage_method})')
            plt.colorbar(scatter, ax=ax1)
            
            dendrogram(linkage_matrix, ax=ax2, truncate_mode='level', p=5)
            ax2.set_title('Dendrogram')
            
            st.pyplot(fig)
            st.write(f"**Clusters:** {n_clusters}")

def gmm_playground():
    st.subheader("Gaussian Mixture Model Playground")
    
    X = None
    if 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            X = df[numeric_cols[:2]].values
            st.info(f"Using uploaded data: {X.shape[0]} samples, 2 features")
    
    if st.button("Generate Sample Data", key="gmm_data"):
        X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=0)
        st.session_state.gmm_data = X
        st.success("Sample data generated!")
    
    if 'gmm_data' in st.session_state and X is None:
        X = st.session_state.gmm_data
    
    if X is not None:
        n_components = st.slider("Number of Components", 2, 6, 3)
        
        if st.button("Run GMM"):
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            labels = gmm.fit_predict(X)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
            ax.set_title(f'Gaussian Mixture Model (components={n_components})')
            plt.colorbar(scatter)
            st.pyplot(fig)
            st.write(f"**AIC:** {gmm.aic(X):.2f}")
            st.write(f"**BIC:** {gmm.bic(X):.2f}")

def spectral_playground():
    st.subheader("Spectral Clustering Playground")
    
    X = None
    if 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            X = df[numeric_cols[:2]].values
            st.info(f"Using uploaded data: {X.shape[0]} samples, 2 features")
    
    if st.button("Generate Sample Data", key="spectral_data"):
        X, _ = make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=0)
        st.session_state.spectral_data = X
        st.success("Circular data generated!")
    
    if 'spectral_data' in st.session_state and X is None:
        X = st.session_state.spectral_data
    
    if X is not None:
        n_clusters = st.slider("Number of Clusters", 2, 6, 2)
        
        if st.button("Run Spectral Clustering"):
            spectral = SpectralClustering(n_clusters=n_clusters, random_state=42)
            labels = spectral.fit_predict(X)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
            ax.set_title(f'Spectral Clustering (clusters={n_clusters})')
            plt.colorbar(scatter)
            st.pyplot(fig)

def tsne_playground():
    st.subheader("t-SNE Playground")
    
    X = None
    if 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 3:
            X = df[numeric_cols].values
            st.info(f"Using uploaded data: {X.shape[0]} samples, {X.shape[1]} features")
    
    if st.button("Generate Sample Data", key="tsne_data"):
        X, _ = make_blobs(n_samples=300, centers=4, n_features=10, random_state=0)
        st.session_state.tsne_data = X
        st.success("10D sample data generated!")
    
    if 'tsne_data' in st.session_state and X is None:
        X = st.session_state.tsne_data
    
    if X is not None:
        perplexity = st.slider("Perplexity", 5, 50, 30)
        
        if st.button("Run t-SNE"):
            with st.spinner("Running t-SNE (this may take a moment)..."):
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                X_tsne = tsne.fit_transform(X)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7)
            ax.set_title(f't-SNE Visualization (perplexity={perplexity})')
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            st.pyplot(fig)

if __name__ == "__main__":
    st.set_page_config(page_title="Unsupervised Learning", page_icon="🧭")
    st.title("🧭 Unsupervised Learning Algorithms")
    unsupervised()