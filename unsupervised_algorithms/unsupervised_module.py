import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

def unsupervised():
    st.write("Unsupervised Learning")
    
    options = ["K-Means", "DBSCAN", "PCA", "Hierarchical Clustering", "Gaussian Mixture", "Spectral Clustering", "t-SNE", "UMAP", "Isolation Forest", "Locally Linear Embedding", "Factor Analysis", "Birch Clustering"]
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
    
    elif selected_option == "UMAP":
        view = st.radio("Choose View", ["UMAP Overview", "UMAP Playground"])
        if view == "UMAP Overview":
            st.subheader("UMAP Overview")
            st.write("**UMAP** (Uniform Manifold Approximation) is a dimensionality reduction technique.")
            st.write("- **Use case:** Better than t-SNE for preserving global structure")
            st.write("- **Key parameters:** n_neighbors, min_dist")
        elif view == "UMAP Playground":
            umap_playground()
    
    elif selected_option == "Isolation Forest":
        view = st.radio("Choose View", ["Isolation Forest Overview", "Isolation Forest Playground"])
        if view == "Isolation Forest Overview":
            st.subheader("Isolation Forest Overview")
            st.write("**Isolation Forest** detects anomalies by isolating data points.")
            st.write("- **Use case:** Fraud detection, outlier detection")
            st.write("- **Key parameters:** Contamination rate, number of estimators")
        elif view == "Isolation Forest Playground":
            isolation_forest_playground()
    
    elif selected_option == "Locally Linear Embedding":
        view = st.radio("Choose View", ["LLE Overview", "LLE Playground"])
        if view == "LLE Overview":
            st.subheader("LLE Overview")
            st.write("**Locally Linear Embedding** preserves local neighborhood relationships.")
            st.write("- **Use case:** Non-linear dimensionality reduction")
            st.write("- **Key parameters:** Number of neighbors")
        elif view == "LLE Playground":
            lle_playground()
    
    elif selected_option == "Factor Analysis":
        view = st.radio("Choose View", ["Factor Analysis Overview", "Factor Analysis Playground"])
        if view == "Factor Analysis Overview":
            st.subheader("Factor Analysis Overview")
            st.write("**Factor Analysis** identifies latent variables underlying observed variables.")
            st.write("- **Use case:** Psychometrics, survey analysis")
            st.write("- **Key parameters:** Number of factors")
        elif view == "Factor Analysis Playground":
            factor_analysis_playground()
    
    elif selected_option == "Birch Clustering":
        view = st.radio("Choose View", ["Birch Overview", "Birch Playground"])
        if view == "Birch Overview":
            st.subheader("Birch Overview")
            st.write("**Birch** is a hierarchical clustering algorithm optimized for large datasets.")
            st.write("- **Use case:** Streaming data, memory-efficient clustering")
            st.write("- **Key parameters:** Branching factor, threshold")
        elif view == "Birch Playground":
            birch_playground()

def kmeans_playground():
    st.subheader("K-Means Clustering Playground")
    
    # Check if data exists from sidebar upload or session state
    X = None
    data_source = "No data loaded"
    
    # Priority: uploaded data > generated data > session data
    if 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            X = df[numeric_cols[:2]].values
            data_source = f"Uploaded dataset: {X.shape[0]} samples, {len(numeric_cols)} features"
            st.success(data_source)
        elif len(numeric_cols) == 1:
            st.warning("Need at least 2 numeric features for visualization")
            return
        else:
            st.warning("No numeric features found in uploaded data")
            return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("Generate Sample Data", key="kmeans_data_btn"):
            with st.spinner("Generating sample data..."):
                X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
                st.session_state.kmeans_data = X
                st.session_state.kmeans_source = "Sample data generated"
                st.rerun()
    
    with col2:
        if st.button("Clear Data", key="kmeans_clear"):
            if 'kmeans_data' in st.session_state:
                del st.session_state.kmeans_data
            st.rerun()
    
    # Load data from session state if no uploaded data
    if X is None and 'kmeans_data' in st.session_state:
        X = st.session_state.kmeans_data
        data_source = st.session_state.get('kmeans_source', 'Sample data')
        st.info(data_source)
    
    if X is not None:
        st.write(f"**Data shape:** {X.shape}")
        
        # Show data preview
        with st.expander("📊 Data Preview"):
            df_preview = pd.DataFrame(X, columns=[f'Feature {i+1}' for i in range(X.shape[1])])
            st.dataframe(df_preview.head())
            st.write(f"**Range:** X: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}], Y: [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")
        
        n_clusters = st.slider("Number of Clusters", 2, 8, 3)
        max_iter = st.slider("Max Iterations", 10, 300, 100)
        
        if st.button("Run K-Means", key="run_kmeans"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Initializing K-Means...")
                progress_bar.progress(10)
                
                kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42, verbose=0)
                
                status_text.text("Fitting model...")
                progress_bar.progress(30)
                
                labels = kmeans.fit_predict(X)
                
                progress_bar.progress(70)
                status_text.text("Generating visualization...")
                
                centers = kmeans.cluster_centers_
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Original data
                ax1.scatter(X[:, 0], X[:, 1], alpha=0.3, color='gray')
                ax1.set_title('Original Data')
                
                # Clustered data
                scatter = ax2.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
                ax2.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
                ax2.set_title(f'K-Means Clustering (k={n_clusters})')
                ax2.legend()
                plt.colorbar(scatter)
                
                st.pyplot(fig)
                
                progress_bar.progress(90)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Inertia", f"{kmeans.inertia_:.2f}")
                with col2:
                    st.metric("Iterations", kmeans.n_iter_)
                with col3:
                    st.metric("Clusters", n_clusters)
                
                # Cluster statistics
                st.subheader("📈 Cluster Statistics")
                cluster_stats = pd.DataFrame({
                    'Cluster': range(n_clusters),
                    'Size': [np.sum(labels == i) for i in range(n_clusters)],
                    'Percentage': [f"{np.sum(labels == i)/len(labels)*100:.1f}%" for i in range(n_clusters)]
                })
                st.dataframe(cluster_stats)
                
                progress_bar.progress(100)
                status_text.text("Complete!")
                
            except Exception as e:
                st.error(f"Error running K-Means: {str(e)}")
            finally:
                progress_bar.empty()
                status_text.empty()

def dbscan_playground():
    st.subheader("DBSCAN Playground")
    
    # Check if data exists from sidebar upload or session state
    X = None
    data_source = "No data loaded"
    
    # Priority: uploaded data > generated data > session data
    if 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            X = df[numeric_cols[:2]].values
            data_source = f"Uploaded dataset: {X.shape[0]} samples, 2 features"
            st.success(data_source)
        elif len(numeric_cols) == 1:
            st.warning("Need at least 2 numeric features for visualization")
            return
        else:
            st.warning("No numeric features found in uploaded data")
            return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("Generate Sample Data", key="dbscan_data_btn"):
            with st.spinner("Generating sample data..."):
                X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
                st.session_state.dbscan_data = X
                st.session_state.dbscan_source = "Sample data generated"
                st.rerun()
    
    with col2:
        if st.button("Clear Data", key="dbscan_clear"):
            if 'dbscan_data' in st.session_state:
                del st.session_state.dbscan_data
            st.rerun()
    
    # Load data from session state if no uploaded data
    if X is None and 'dbscan_data' in st.session_state:
        X = st.session_state.dbscan_data
        data_source = st.session_state.get('dbscan_source', 'Sample data')
        st.info(data_source)
    
    if X is not None:
        st.write(f"**Data shape:** {X.shape}")
        
        with st.expander("📊 Data Preview"):
            df_preview = pd.DataFrame(X, columns=[f'Feature {i+1}' for i in range(X.shape[1])])
            st.dataframe(df_preview.head())
        
        eps = st.slider("Epsilon", 0.1, 2.0, 0.5, 0.1)
        min_samples = st.slider("Minimum samples", 2, 20, 5)
        
        if st.button("Run DBSCAN", key="run_dbscan"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Initializing DBSCAN...")
                progress_bar.progress(10)
                
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                
                status_text.text("Fitting model...")
                progress_bar.progress(30)
                
                labels = dbscan.fit_predict(X)
                
                progress_bar.progress(70)
                status_text.text("Generating visualization...")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Original data
                ax1.scatter(X[:, 0], X[:, 1], alpha=0.3, color='gray')
                ax1.set_title('Original Data')
                
                # DBSCAN results
                unique_labels = set(labels)
                colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
                
                for k, col in zip(unique_labels, colors):
                    if k == -1:
                        col = [0, 0, 0, 1]
                    class_member_mask = (labels == k)
                    xy = X[class_member_mask]
                    ax2.scatter(xy[:, 0], xy[:, 1], c=[col], alpha=0.7, 
                              label=f'Cluster {k}' if k != -1 else f'Noise ({len(xy)} points)')
                
                ax2.set_title(f'DBSCAN (eps={eps}, min_samples={min_samples})')
                ax2.legend()
                
                st.pyplot(fig)
                
                progress_bar.progress(90)
                
                # Display metrics
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Clusters", n_clusters)
                with col2:
                    st.metric("Noise Points", n_noise)
                with col3:
                    st.metric("Total Points", len(X))
                
                # Cluster statistics
                st.subheader("📈 Cluster Statistics")
                cluster_data = []
                for label in unique_labels:
                    mask = labels == label
                    cluster_data.append({
                        'Cluster': f'Cluster {label}' if label != -1 else 'Noise',
                        'Size': np.sum(mask),
                        'Percentage': f"{np.sum(mask)/len(labels)*100:.1f}%",
                        'Centroid X': f"{X[mask, 0].mean():.2f}" if np.sum(mask) > 0 else "N/A",
                        'Centroid Y': f"{X[mask, 1].mean():.2f}" if np.sum(mask) > 0 else "N/A"
                    })
                
                cluster_df = pd.DataFrame(cluster_data)
                st.dataframe(cluster_df)
                
                progress_bar.progress(100)
                status_text.text("Complete!")
                
            except Exception as e:
                st.error(f"Error running DBSCAN: {str(e)}")
            finally:
                progress_bar.empty()
                status_text.empty()

def pca_playground():
    st.subheader("PCA Playground")
    
    # Check if data exists from sidebar upload or session state
    X = None
    data_source = "No data loaded"
    
    # Priority: uploaded data > generated data > session data
    if 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            X = df[numeric_cols].values
            data_source = f"Uploaded dataset: {X.shape[0]} samples, {len(numeric_cols)} features"
            st.success(data_source)
        else:
            st.warning("Need at least 2 numeric features for PCA")
            return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("Generate Sample Data", key="pca_data_btn"):
            with st.spinner("Generating high-dimensional sample data..."):
                X, _ = make_blobs(n_samples=300, centers=4, n_features=8, random_state=0)
                st.session_state.pca_data = X
                st.session_state.pca_source = "8D sample data generated"
                st.rerun()
    
    with col2:
        if st.button("Clear Data", key="pca_clear"):
            if 'pca_data' in st.session_state:
                del st.session_state.pca_data
            st.rerun()
    
    # Load data from session state if no uploaded data
    if X is None and 'pca_data' in st.session_state:
        X = st.session_state.pca_data
        data_source = st.session_state.get('pca_source', 'Sample data')
        st.info(data_source)
    
    if X is not None:
        st.write(f"**Data shape:** {X.shape}")
        
        with st.expander("📊 Data Preview"):
            df_preview = pd.DataFrame(X, columns=[f'Feature {i+1}' for i in range(X.shape[1])])
            st.dataframe(df_preview.head())
        
        max_components = min(X.shape[1], 10)  # Limit to prevent too many sliders
        n_components = st.slider("Number of Components", 2, max_components, 2)
        
        if st.button("Run PCA", key="run_pca"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Standardizing data...")
                progress_bar.progress(20)
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                status_text.text("Computing PCA...")
                progress_bar.progress(50)
                
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X_scaled)
                
                progress_bar.progress(80)
                status_text.text("Creating visualizations...")
                
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                
                # Original data (first 2 dimensions)
                ax1.scatter(X[:, 0], X[:, 1], alpha=0.7)
                ax1.set_title('Original Data (First 2 Features)')
                ax1.set_xlabel('Feature 1')
                ax1.set_ylabel('Feature 2')
                
                # PCA transformed data
                scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, c='red')
                ax2.set_title('PCA Transformed Data')
                ax2.set_xlabel('PC1')
                ax2.set_ylabel('PC2')
                
                # Scree plot
                explained_var = pca.explained_variance_ratio_
                ax3.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7)
                ax3.set_xlabel('Principal Component')
                ax3.set_ylabel('Explained Variance Ratio')
                ax3.set_title('Scree Plot')
                ax3.set_xticks(range(1, len(explained_var) + 1))
                
                # Cumulative explained variance
                cumulative_var = np.cumsum(explained_var)
                ax4.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'bo-')
                ax4.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% variance')
                ax4.set_xlabel('Number of Components')
                ax4.set_ylabel('Cumulative Explained Variance')
                ax4.set_title('Cumulative Explained Variance')
                ax4.legend()
                ax4.grid(True)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                progress_bar.progress(90)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Dimensions", X.shape[1])
                with col2:
                    st.metric("Reduced Dimensions", n_components)
                with col3:
                    st.metric("Variance Retained", f"{np.sum(pca.explained_variance_ratio_)*100:.1f}%")
                
                # Detailed explained variance
                st.subheader("📊 Explained Variance Analysis")
                var_df = pd.DataFrame({
                    'Component': [f'PC{i+1}' for i in range(n_components)],
                    'Explained Variance': [f"{ratio:.4f}" for ratio in pca.explained_variance_ratio_],
                    'Percentage': [f"{ratio*100:.1f}%" for ratio in pca.explained_variance_ratio_],
                    'Cumulative': [f"{sum(pca.explained_variance_ratio_[:i+1])*100:.1f}%" for i in range(n_components)]
                })
                st.dataframe(var_df)
                
                progress_bar.progress(100)
                status_text.text("Complete!")
                
            except Exception as e:
                st.error(f"Error running PCA: {str(e)}")
            finally:
                progress_bar.empty()
                status_text.empty()

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
    
    if st.button("Generate Sample Data", key="hier_data_btn"):
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
    
    if st.button("Generate Sample Data", key="gmm_data_btn"):
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
    
    if st.button("Generate Sample Data", key="spectral_data_btn"):
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
    
    if st.button("Generate Sample Data", key="tsne_data_btn"):
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

def umap_playground():
    st.subheader("UMAP Playground")
    
    try:
        import umap
        UMAP_AVAILABLE = True
    except ImportError:
        UMAP_AVAILABLE = False
        st.error("UMAP not installed. Install with: pip install umap-learn")
        return
    
    # Data handling
    X = None
    data_source = "No data loaded"
    
    if 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            X = df[numeric_cols].values
            data_source = f"Uploaded dataset: {X.shape[0]} samples, {len(numeric_cols)} features"
            st.success(data_source)
    
    if X is None:
        st.info("Generating sample data...")
        X, _ = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)
        data_source = "Generated sample data: 300 samples, 2 features"
        st.success(data_source)
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        n_neighbors = st.slider("n_neighbors", 2, 50, 15)
        min_dist = st.slider("min_dist", 0.0, 1.0, 0.1)
    with col2:
        metric = st.selectbox("Metric", ["euclidean", "manhattan", "cosine"])
        random_state = st.number_input("Random State", 42)
    
    if st.button("Apply UMAP", key="umap_apply"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Initializing UMAP...")
            progress_bar.progress(20)
            
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, 
                              metric=metric, random_state=random_state)
            
            status_text.text("Fitting UMAP...")
            progress_bar.progress(60)
            
            X_umap = reducer.fit_transform(X)
            
            status_text.text("Creating visualization...")
            progress_bar.progress(100)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Original data
            if X.shape[1] >= 2:
                ax1.scatter(X[:, 0], X[:, 1], alpha=0.6)
                ax1.set_title("Original Data")
                ax1.set_xlabel("Feature 1")
                ax1.set_ylabel("Feature 2")
            
            # UMAP result
            ax2.scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.6)
            ax2.set_title("UMAP Projection")
            ax2.set_xlabel("UMAP 1")
            ax2.set_ylabel("UMAP 2")
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.success(f"UMAP completed! Reduced from {X.shape[1]} to 2 dimensions")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()

def isolation_forest_playground():
    st.subheader("Isolation Forest Playground")
    
    # Data handling
    X = None
    data_source = "No data loaded"
    
    if 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            X = df[numeric_cols].values
            data_source = f"Uploaded dataset: {X.shape[0]} samples, {len(numeric_cols)} features"
            st.success(data_source)
    
    if X is None:
        st.info("Generating sample data with outliers...")
        X, _ = make_blobs(n_samples=300, centers=2, n_features=2, random_state=42)
        # Add outliers
        outliers = np.random.uniform(low=-10, high=10, size=(20, 2))
        X = np.vstack([X, outliers])
        data_source = "Generated sample data with outliers: 320 samples, 2 features"
        st.success(data_source)
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        contamination = st.slider("Contamination", 0.01, 0.5, 0.1)
        n_estimators = st.slider("n_estimators", 50, 500, 100)
    with col2:
        max_samples = st.selectbox("max_samples", ["auto", 100, 256, 512])
        random_state = st.number_input("Random State", 42)
    
    if st.button("Detect Anomalies", key="isolation_apply"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Initializing Isolation Forest...")
            progress_bar.progress(20)
            
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=contamination, 
                                       n_estimators=n_estimators,
                                       max_samples=max_samples,
                                       random_state=random_state)
            
            status_text.text("Fitting Isolation Forest...")
            progress_bar.progress(60)
            
            anomaly_labels = iso_forest.fit_predict(X)
            anomaly_scores = iso_forest.decision_function(X)
            
            status_text.text("Creating visualization...")
            progress_bar.progress(100)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Original data
            ax1.scatter(X[:, 0], X[:, 1], alpha=0.6)
            ax1.set_title("Original Data")
            ax1.set_xlabel("Feature 1")
            ax1.set_ylabel("Feature 2")
            
            # Anomaly detection results
            normal_mask = anomaly_labels == 1
            anomaly_mask = anomaly_labels == -1
            
            ax2.scatter(X[normal_mask, 0], X[normal_mask, 1], c='blue', alpha=0.6, label='Normal')
            ax2.scatter(X[anomaly_mask, 0], X[anomaly_mask, 1], c='red', alpha=0.8, label='Anomaly')
            ax2.set_title(f"Anomaly Detection ({np.sum(anomaly_mask)} anomalies found)")
            ax2.set_xlabel("Feature 1")
            ax2.set_ylabel("Feature 2")
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.success(f"Detected {np.sum(anomaly_mask)} anomalies out of {len(X)} samples")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()

def lle_playground():
    st.subheader("Locally Linear Embedding Playground")
    
    # Data handling
    X = None
    data_source = "No data loaded"
    
    if 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 3:
            X = df[numeric_cols].values
            data_source = f"Uploaded dataset: {X.shape[0]} samples, {len(numeric_cols)} features"
            st.success(data_source)
    
    if X is None:
        st.info("Generating Swiss roll sample data...")
        from sklearn.datasets import make_swiss_roll
        X, _ = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)
        data_source = "Generated Swiss roll data: 1000 samples, 3 features"
        st.success(data_source)
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        n_neighbors = st.slider("n_neighbors", 5, 30, 12)
        n_components = st.slider("n_components", 2, 3, 2)
    with col2:
        method = st.selectbox("Method", ["standard", "modified", "ltsa"])
        random_state = st.number_input("Random State", 42)
    
    if st.button("Apply LLE", key="lle_apply"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Initializing LLE...")
            progress_bar.progress(20)
            
            from sklearn.manifold import LocallyLinearEmbedding
            lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, 
                                       n_components=n_components,
                                       method=method,
                                       random_state=random_state)
            
            status_text.text("Fitting LLE...")
            progress_bar.progress(60)
            
            X_lle = lle.fit_transform(X)
            
            status_text.text("Creating visualization...")
            progress_bar.progress(100)
            
            fig = plt.figure(figsize=(15, 6))
            
            if X.shape[1] >= 3:
                # 3D original data
                ax1 = fig.add_subplot(121, projection='3d')
                ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 2], cmap='viridis', alpha=0.6)
                ax1.set_title("Original 3D Data")
                
                # LLE result
                ax2 = fig.add_subplot(122)
                ax2.scatter(X_lle[:, 0], X_lle[:, 1], c=X[:, 2], cmap='viridis', alpha=0.6)
                ax2.set_title(f"LLE Projection ({n_components}D)")
            else:
                # 2D data
                ax1 = fig.add_subplot(121)
                ax1.scatter(X[:, 0], X[:, 1], alpha=0.6)
                ax1.set_title("Original Data")
                
                ax2 = fig.add_subplot(122)
                ax2.scatter(X_lle[:, 0], X_lle[:, 1], alpha=0.6)
                ax2.set_title(f"LLE Projection ({n_components}D)")
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.success(f"LLE completed! Reduced from {X.shape[1]} to {n_components} dimensions")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()

def factor_analysis_playground():
    st.subheader("Factor Analysis Playground")
    
    # Data handling
    X = None
    data_source = "No data loaded"
    
    if 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 3:
            X = df[numeric_cols].values
            data_source = f"Uploaded dataset: {X.shape[0]} samples, {len(numeric_cols)} features"
            st.success(data_source)
    
    if X is None:
        st.info("Generating sample data...")
        X, _ = make_blobs(n_samples=500, centers=3, n_features=5, random_state=42)
        data_source = "Generated sample data: 500 samples, 5 features"
        st.success(data_source)
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        n_factors = st.slider("Number of factors", 1, min(X.shape[1], 10), 2)
        rotation = st.selectbox("Rotation", ["varimax", "quartimax", "promax"])
    with col2:
        random_state = st.number_input("Random State", 42)
    
    if st.button("Apply Factor Analysis", key="factor_apply"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Initializing Factor Analysis...")
            progress_bar.progress(20)
            
            from sklearn.decomposition import FactorAnalysis
            fa = FactorAnalysis(n_components=n_factors, random_state=random_state)
            
            status_text.text("Fitting Factor Analysis...")
            progress_bar.progress(60)
            
            X_fa = fa.fit_transform(X)
            
            status_text.text("Creating visualization...")
            progress_bar.progress(100)
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Original data (first 2 features)
            ax1.scatter(X[:, 0], X[:, 1], alpha=0.6)
            ax1.set_title("Original Data (First 2 Features)")
            ax1.set_xlabel("Feature 1")
            ax1.set_ylabel("Feature 2")
            
            # Factor loadings
            loadings = fa.components_
            im = ax2.imshow(loadings, cmap='coolwarm', aspect='auto')
            ax2.set_title("Factor Loadings")
            ax2.set_xlabel("Original Features")
            ax2.set_ylabel("Factors")
            plt.colorbar(im, ax=ax2)
            
            # Factor scores
            ax3.scatter(X_fa[:, 0], X_fa[:, 1], alpha=0.6)
            ax3.set_title("Factor Scores")
            ax3.set_xlabel("Factor 1")
            ax3.set_ylabel("Factor 2")
            
            # Explained variance
            explained_var = np.var(X_fa, axis=0) / np.var(X, axis=0).sum()
            ax4.bar(range(len(explained_var)), explained_var)
            ax4.set_title("Explained Variance by Factor")
            ax4.set_xlabel("Factor")
            ax4.set_ylabel("Proportion of Variance")
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.success(f"Factor Analysis completed! Identified {n_factors} latent factors")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()

def birch_playground():
    st.subheader("Birch Clustering Playground")
    
    # Data handling
    X = None
    data_source = "No data loaded"
    
    if 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            X = df[numeric_cols].values
            data_source = f"Uploaded dataset: {X.shape[0]} samples, {len(numeric_cols)} features"
            st.success(data_source)
    
    if X is None:
        st.info("Generating sample data...")
        X, _ = make_blobs(n_samples=500, centers=4, n_features=2, random_state=42)
        data_source = "Generated sample data: 500 samples, 2 features"
        st.success(data_source)
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        branching_factor = st.slider("Branching factor", 10, 100, 50)
        threshold = st.slider("Threshold", 0.1, 2.0, 0.5)
    with col2:
        n_clusters = st.slider("n_clusters", 2, 10, 3)
        compute_labels = st.checkbox("Compute labels", True)
    
    if st.button("Apply Birch", key="birch_apply"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Initializing Birch...")
            progress_bar.progress(20)
            
            from sklearn.cluster import Birch
            birch = Birch(branching_factor=branching_factor, 
                        threshold=threshold,
                        n_clusters=n_clusters if n_clusters > 0 else None,
                        compute_labels=compute_labels)
            
            status_text.text("Fitting Birch...")
            progress_bar.progress(60)
            
            birch.fit(X)
            labels = birch.labels_
            
            status_text.text("Creating visualization...")
            progress_bar.progress(100)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Original data
            ax1.scatter(X[:, 0], X[:, 1], alpha=0.6)
            ax1.set_title("Original Data")
            ax1.set_xlabel("Feature 1")
            ax1.set_ylabel("Feature 2")
            
            # Birch clustering results
            unique_labels = np.unique(labels)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax2.scatter(X[mask, 0], X[mask, 1], 
                           color=colors[i], alpha=0.6, label=f'Cluster {label}')
            
            ax2.set_title(f"Birch Clustering ({len(unique_labels)} clusters)")
            ax2.set_xlabel("Feature 1")
            ax2.set_ylabel("Feature 2")
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Cluster statistics
            st.subheader("Cluster Statistics")
            for label in unique_labels:
                mask = labels == label
                st.write(f"Cluster {label}: {np.sum(mask)} samples ({np.sum(mask)/len(X)*100:.1f}%)")
            
            st.success(f"Birch clustering completed! Found {len(unique_labels)} clusters")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()

if __name__ == "__main__":
    st.set_page_config(page_title="Unsupervised Learning", page_icon="🧭")
    st.title("🧭 Unsupervised Learning Algorithms")
    unsupervised()