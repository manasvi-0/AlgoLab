import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

def unsupervised():
    """
    Main function for unsupervised learning module
    Provides implementations of clustering and dimensionality reduction algorithms
    """
    st.header("🧠 Unsupervised Learning Algorithms")
    st.markdown("Explore clustering and dimensionality reduction techniques with interactive visualizations.")
    
    # Algorithm selection
    algorithm = st.sidebar.selectbox(
        "Select Algorithm",
        ["K-Means Clustering", "DBSCAN Clustering", "Hierarchical Clustering", 
         "PCA (Principal Component Analysis)", "Algorithm Comparison"]
    )
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Choose Data Source",
        ["Generate Clustering Data", "Upload CSV", "Use Current Data"]
    )
    
    # Get data
    df = get_data_for_unsupervised(data_source)
    
    if df is not None:
        if algorithm == "K-Means Clustering":
            kmeans_clustering(df)
        elif algorithm == "DBSCAN Clustering":
            dbscan_clustering(df)
        elif algorithm == "Hierarchical Clustering":
            hierarchical_clustering(df)
        elif algorithm == "PCA (Principal Component Analysis)":
            pca_analysis(df)
        elif algorithm == "Algorithm Comparison":
            algorithm_comparison(df)
    else:
        st.warning("⚠️ Please provide data to proceed with unsupervised learning analysis.")

def get_data_for_unsupervised(data_source):
    """Get data based on user selection"""
    
    if data_source == "Generate Clustering Data":
        return generate_clustering_data()
    elif data_source == "Upload CSV":
        return upload_csv_for_unsupervised()
    elif data_source == "Use Current Data":
        st.info("💡 This would use data from the 'Visualize Data' tab if available.")
        # For now, generate sample data
        return generate_clustering_data()
    
    return None

def generate_clustering_data():
    """Generate synthetic clustering data"""
    st.subheader("🔧 Generate Clustering Data")
    
    with st.expander("Data Generation Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.slider("Number of Samples", 100, 1000, 300, key="unsup_samples")
            n_centers = st.slider("Number of Clusters", 2, 8, 3, key="unsup_centers")
        with col2:
            n_features = st.slider("Number of Features", 2, 10, 2, key="unsup_features")
            cluster_std = st.slider("Cluster Standard Deviation", 0.5, 3.0, 1.0, key="unsup_std")
    
    if st.button("Generate Data", type="primary", key="unsup_generate"):
        # Generate synthetic clustering data
        X, y_true = make_blobs(
            n_samples=n_samples,
            centers=n_centers,
            n_features=n_features,
            cluster_std=cluster_std,
            random_state=42
        )
        
        # Create DataFrame
        feature_names = [f'Feature_{i+1}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['True_Cluster'] = y_true
        
        st.success(f"✅ Generated clustering data with {n_samples} samples and {n_centers} clusters!")
        
        # Show data preview
        st.subheader("📊 Data Preview")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df.head(), use_container_width=True)
        with col2:
            st.write("**Data Shape:**", df.shape)
            st.write("**Features:**", feature_names)
            st.write("**True Clusters:**", sorted(df['True_Cluster'].unique()))
        
        return df
    
    return None

def upload_csv_for_unsupervised():
    """Handle CSV upload for unsupervised learning"""
    st.subheader("📁 Upload CSV for Clustering")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="unsupervised_upload")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Select numeric columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                st.error("❌ No numeric columns found in the dataset!")
                return None
            
            st.success(f"✅ Successfully loaded {df.shape[0]} rows and {len(numeric_cols)} numeric columns!")
            
            # Feature selection
            selected_features = st.multiselect(
                "Select features for clustering",
                numeric_cols,
                default=list(numeric_cols)[:min(5, len(numeric_cols))]
            )
            
            if selected_features:
                return df[selected_features]
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    return None

def kmeans_clustering(df):
    """Implement K-Means clustering with visualization"""
    st.subheader("🎯 K-Means Clustering")
    
    # Remove non-numeric columns for clustering
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if 'True_Cluster' in df.columns:
        numeric_cols = [col for col in numeric_cols if col != 'True_Cluster']
    
    X = df[numeric_cols].values
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        n_clusters = st.slider("Number of Clusters (k)", 2, 10, 3, key="kmeans_clusters")
    with col2:
        max_iter = st.slider("Max Iterations", 100, 1000, 300, key="kmeans_iter")
    with col3:
        random_state = st.slider("Random State", 0, 100, 42, key="kmeans_random")
    
    # Standardization option
    standardize = st.checkbox("Standardize Features", value=True)
    
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    # Run K-Means
    if st.button("Run K-Means Clustering"):
        kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Add results to dataframe
        df_result = df.copy()
        df_result['KMeans_Cluster'] = cluster_labels
        
        # Calculate metrics
        inertia = kmeans.inertia_
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Within-cluster Sum of Squares (WCSS)", f"{inertia:.2f}")
        with col2:
            st.metric("Number of Iterations", kmeans.n_iter_)
        
        # Visualizations
        if len(numeric_cols) >= 2:
            # 2D visualization
            fig = px.scatter(
                df_result, 
                x=numeric_cols[0], 
                y=numeric_cols[1],
                color='KMeans_Cluster',
                title=f"K-Means Clustering Results (k={n_clusters})",
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            
            # Add cluster centers
            if not standardize:
                centers = kmeans.cluster_centers_
            else:
                centers = scaler.inverse_transform(kmeans.cluster_centers_)
            
            fig.add_scatter(
                x=centers[:, 0],
                y=centers[:, 1],
                mode='markers',
                marker=dict(size=15, symbol='x', color='black'),
                name='Centroids'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Show cluster comparison if true clusters exist
        if 'True_Cluster' in df.columns:
            st.subheader("🔍 Cluster Comparison")
            col1, col2 = st.columns(2)
            
            with col1:
                fig_true = px.scatter(
                    df_result,
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    color='True_Cluster',
                    title="True Clusters"
                )
                st.plotly_chart(fig_true, use_container_width=True)
            
            with col2:
                fig_kmeans = px.scatter(
                    df_result,
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    color='KMeans_Cluster',
                    title="K-Means Clusters"
                )
                st.plotly_chart(fig_kmeans, use_container_width=True)
        
        # Elbow method
        st.subheader("📈 Elbow Method for Optimal k")
        elbow_method(X_scaled)

def elbow_method(X):
    """Implement elbow method to find optimal number of clusters"""
    k_range = range(1, 11)
    wcss = []
    
    progress_bar = st.progress(0)
    for i, k in enumerate(k_range):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        progress_bar.progress((i + 1) / len(k_range))
    
    # Plot elbow curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(k_range),
        y=wcss,
        mode='lines+markers',
        name='WCSS',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    fig.update_layout(
        title="Elbow Method for Optimal Number of Clusters",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Within-cluster Sum of Squares (WCSS)"
    )
    st.plotly_chart(fig, use_container_width=True)

def dbscan_clustering(df):
    """Implement DBSCAN clustering with visualization"""
    st.subheader("🌐 DBSCAN Clustering")
    st.markdown("DBSCAN (Density-Based Spatial Clustering) can find clusters of arbitrary shape and identify outliers.")
    
    # Remove non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if 'True_Cluster' in df.columns:
        numeric_cols = [col for col in numeric_cols if col != 'True_Cluster']
    
    X = df[numeric_cols].values
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        eps = st.slider("Epsilon (ε) - Neighborhood radius", 0.1, 2.0, 0.5, 0.1, key="dbscan_eps")
    with col2:
        min_samples = st.slider("Min Samples - Minimum points in neighborhood", 2, 20, 5, key="dbscan_min")
    
    # Standardization
    standardize = st.checkbox("Standardize Features", value=True, key="dbscan_std")
    
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    # Run DBSCAN
    if st.button("Run DBSCAN Clustering"):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(X_scaled)
        
        # Add results to dataframe
        df_result = df.copy()
        df_result['DBSCAN_Cluster'] = cluster_labels
        
        # Calculate metrics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Clusters", n_clusters)
        with col2:
            st.metric("Number of Outliers", n_noise)
        with col3:
            st.metric("Outlier Percentage", f"{n_noise/len(cluster_labels)*100:.1f}%")
        
        # Visualization
        if len(numeric_cols) >= 2:
            # Create color map (outliers in gray)
            unique_labels = sorted(set(cluster_labels))
            colors = px.colors.qualitative.Set1[:len(unique_labels)]
            color_map = {label: colors[i] if label != -1 else 'gray' 
                        for i, label in enumerate(unique_labels)}
            
            fig = px.scatter(
                df_result,
                x=numeric_cols[0],
                y=numeric_cols[1],
                color='DBSCAN_Cluster',
                title=f"DBSCAN Clustering Results (ε={eps}, min_samples={min_samples})",
                color_discrete_map=color_map
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Cluster summary
        st.subheader("📊 Cluster Summary")
        cluster_summary = df_result.groupby('DBSCAN_Cluster').size().reset_index()
        cluster_summary.columns = ['Cluster', 'Size']
        cluster_summary['Cluster'] = cluster_summary['Cluster'].apply(lambda x: 'Outliers' if x == -1 else f'Cluster {x}')
        st.dataframe(cluster_summary, use_container_width=True)

def hierarchical_clustering(df):
    """Implement Hierarchical clustering with dendrogram"""
    st.subheader("🌳 Hierarchical Clustering")
    st.markdown("Hierarchical clustering creates a tree of clusters, useful for understanding data structure.")
    
    # Remove non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if 'True_Cluster' in df.columns:
        numeric_cols = [col for col in numeric_cols if col != 'True_Cluster']
    
    X = df[numeric_cols].values
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        linkage_method = st.selectbox("Linkage Method", 
                                    ["ward", "complete", "average", "single"])
    with col2:
        n_clusters = st.slider("Number of Clusters", 2, 10, 3, key="hier_clusters")
    
    # Standardization
    standardize = st.checkbox("Standardize Features", value=True, key="hier_std")
    
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    # Limit samples for dendrogram (performance)
    max_samples_dendrogram = 100
    if len(X_scaled) > max_samples_dendrogram:
        st.warning(f"⚠️ Using first {max_samples_dendrogram} samples for dendrogram visualization (performance optimization)")
        X_dendrogram = X_scaled[:max_samples_dendrogram]
    else:
        X_dendrogram = X_scaled
    
    # Run Hierarchical Clustering
    if st.button("Run Hierarchical Clustering"):
        # Calculate linkage matrix
        linkage_matrix = linkage(X_dendrogram, method=linkage_method)
        
        # Create dendrogram
        st.subheader("🌳 Dendrogram")
        fig, ax = plt.subplots(figsize=(12, 6))
        dendrogram(linkage_matrix, ax=ax, truncate_mode='level', p=5)
        ax.set_title(f"Hierarchical Clustering Dendrogram ({linkage_method} linkage)")
        ax.set_xlabel("Sample Index or (Cluster Size)")
        ax.set_ylabel("Distance")
        st.pyplot(fig)
        
        # Perform clustering
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        cluster_labels = hierarchical.fit_predict(X_scaled)
        
        # Add results to dataframe
        df_result = df.copy()
        df_result['Hierarchical_Cluster'] = cluster_labels
        
        # Display cluster distribution
        st.subheader("📊 Cluster Distribution")
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        fig = px.bar(x=cluster_counts.index, y=cluster_counts.values, 
                    title="Cluster Size Distribution")
        fig.update_xaxis(title="Cluster")
        fig.update_yaxis(title="Number of Points")
        st.plotly_chart(fig, use_container_width=True)
        
        # 2D visualization
        if len(numeric_cols) >= 2:
            fig = px.scatter(
                df_result,
                x=numeric_cols[0],
                y=numeric_cols[1],
                color='Hierarchical_Cluster',
                title=f"Hierarchical Clustering Results ({linkage_method} linkage, {n_clusters} clusters)"
            )
            st.plotly_chart(fig, use_container_width=True)

def pca_analysis(df):
    """Implement PCA (Principal Component Analysis)"""
    st.subheader("🔍 PCA (Principal Component Analysis)")
    st.markdown("PCA reduces dimensionality while preserving variance in the data.")
    
    # Remove non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if 'True_Cluster' in df.columns:
        numeric_cols = [col for col in numeric_cols if col != 'True_Cluster']
    
    if len(numeric_cols) < 2:
        st.error("❌ Need at least 2 numeric features for PCA analysis.")
        return
    
    X = df[numeric_cols].values
    
    # Parameters
    n_components = st.slider("Number of Components", 2, min(len(numeric_cols), 10), 2, key="pca_components")
    standardize = st.checkbox("Standardize Features", value=True, key="pca_std")
    
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    # Run PCA
    if st.button("Run PCA Analysis"):
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create results dataframe
        pca_columns = [f'PC{i+1}' for i in range(n_components)]
        df_pca = pd.DataFrame(X_pca, columns=pca_columns)
        
        # Add original target if exists
        if 'True_Cluster' in df.columns:
            df_pca['True_Cluster'] = df['True_Cluster'].values
        
        # Display explained variance
        st.subheader("📊 Explained Variance")
        col1, col2 = st.columns(2)
        
        with col1:
            explained_var = pca.explained_variance_ratio_
            cumulative_var = np.cumsum(explained_var)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=pca_columns,
                y=explained_var,
                name='Individual',
                marker_color='lightblue'
            ))
            fig.add_trace(go.Scatter(
                x=pca_columns,
                y=cumulative_var,
                mode='lines+markers',
                name='Cumulative',
                yaxis='y2',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ))
            fig.update_layout(
                title="Explained Variance by Component",
                xaxis_title="Principal Component",
                yaxis_title="Explained Variance Ratio",
                yaxis2=dict(title="Cumulative Explained Variance", overlaying='y', side='right')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Variance table
            variance_df = pd.DataFrame({
                'Component': pca_columns,
                'Explained Variance': explained_var,
                'Cumulative Variance': cumulative_var
            })
            st.dataframe(variance_df, use_container_width=True)
        
        # 2D visualization of first two components
        if n_components >= 2:
            st.subheader("🎯 PCA Visualization")
            
            if 'True_Cluster' in df_pca.columns:
                fig = px.scatter(
                    df_pca,
                    x='PC1',
                    y='PC2',
                    color='True_Cluster',
                    title="PCA: First Two Principal Components"
                )
            else:
                fig = px.scatter(
                    df_pca,
                    x='PC1',
                    y='PC2',
                    title="PCA: First Two Principal Components"
                )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature loadings
        st.subheader("🔗 Feature Loadings")
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        loadings_df = pd.DataFrame(
            loadings,
            columns=pca_columns,
            index=numeric_cols
        )
        st.dataframe(loadings_df, use_container_width=True)
        
        # Biplot for first two components
        if n_components >= 2:
            st.subheader("📈 PCA Biplot")
            fig = go.Figure()
            
            # Add points
            if 'True_Cluster' in df_pca.columns:
                for cluster in df_pca['True_Cluster'].unique():
                    cluster_data = df_pca[df_pca['True_Cluster'] == cluster]
                    fig.add_trace(go.Scatter(
                        x=cluster_data['PC1'],
                        y=cluster_data['PC2'],
                        mode='markers',
                        name=f'Cluster {cluster}',
                        opacity=0.7
                    ))
            else:
                fig.add_trace(go.Scatter(
                    x=df_pca['PC1'],
                    y=df_pca['PC2'],
                    mode='markers',
                    name='Data Points',
                    opacity=0.7
                ))
            
            # Add loading vectors
            scale_factor = 3
            for i, feature in enumerate(numeric_cols):
                fig.add_annotation(
                    x=loadings[i, 0] * scale_factor,
                    y=loadings[i, 1] * scale_factor,
                    ax=0, ay=0,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="red"
                )
                fig.add_annotation(
                    x=loadings[i, 0] * scale_factor,
                    y=loadings[i, 1] * scale_factor,
                    text=feature,
                    showarrow=False,
                    font=dict(color="red", size=10)
                )
            
            fig.update_layout(
                title="PCA Biplot",
                xaxis_title=f"PC1 ({explained_var[0]:.1%} variance)",
                yaxis_title=f"PC2 ({explained_var[1]:.1%} variance)"
            )
            st.plotly_chart(fig, use_container_width=True)

def algorithm_comparison(df):
    """Compare different clustering algorithms"""
    st.subheader("⚖️ Algorithm Comparison")
    st.markdown("Compare the performance of different clustering algorithms on the same dataset.")
    
    # Remove non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if 'True_Cluster' in df.columns:
        numeric_cols = [col for col in numeric_cols if col != 'True_Cluster']
    
    X = df[numeric_cols].values
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Parameters
    n_clusters = st.slider("Number of Clusters", 2, 8, 3, key="comp_clusters")
    
    if st.button("Compare Algorithms"):
        # Initialize algorithms
        algorithms = {
            'K-Means': KMeans(n_clusters=n_clusters, random_state=42),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
            'Hierarchical': AgglomerativeClustering(n_clusters=n_clusters)
        }
        
        # Run algorithms and collect results
        results = {}
        for name, algorithm in algorithms.items():
            cluster_labels = algorithm.fit_predict(X_scaled)
            results[name] = cluster_labels
            
            # Add to dataframe
            df[f'{name}_Cluster'] = cluster_labels
        
        # Visualize results
        if len(numeric_cols) >= 2:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Original Data'] + list(algorithms.keys()),
                specs=[[{"colspan": 2}, None],
                       [{}, {}]]
            )
            
            # Original data
            if 'True_Cluster' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df[numeric_cols[0]],
                        y=df[numeric_cols[1]],
                        mode='markers',
                        marker=dict(color=df['True_Cluster'], colorscale='viridis'),
                        name='True Clusters'
                    ),
                    row=1, col=1
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=df[numeric_cols[0]],
                        y=df[numeric_cols[1]],
                        mode='markers',
                        name='Data Points'
                    ),
                    row=1, col=1
                )
            
            # Algorithm results
            positions = [(2, 1), (2, 2)]
            for i, (name, labels) in enumerate(results.items()):
                if i < 2:  # Only show first two algorithms in subplots
                    row, col = positions[i]
                    fig.add_trace(
                        go.Scatter(
                            x=df[numeric_cols[0]],
                            y=df[numeric_cols[1]],
                            mode='markers',
                            marker=dict(color=labels, colorscale='Set1'),
                            name=name,
                            showlegend=False
                        ),
                        row=row, col=col
                    )
            
            fig.update_layout(height=800, title_text="Algorithm Comparison")
            st.plotly_chart(fig, use_container_width=True)
        
        # Algorithm metrics comparison
        st.subheader("📈 Algorithm Metrics")
        metrics_data = []
        
        for name, labels in results.items():
            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
            n_outliers = list(labels).count(-1)
            
            metrics_data.append({
                'Algorithm': name,
                'Clusters Found': n_clusters_found,
                'Outliers': n_outliers,
                'Outlier %': f"{n_outliers/len(labels)*100:.1f}%"
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Individual algorithm visualizations
        st.subheader("🔍 Detailed Algorithm Results")
        for name, labels in results.items():
            with st.expander(f"{name} Results"):
                if len(numeric_cols) >= 2:
                    fig = px.scatter(
                        x=df[numeric_cols[0]],
                        y=df[numeric_cols[1]],
                        color=labels,
                        title=f"{name} Clustering Results"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Cluster summary
                cluster_summary = pd.Series(labels).value_counts().sort_index()
                cluster_summary.name = 'Count'
                cluster_summary.index.name = 'Cluster'
                st.dataframe(cluster_summary, use_container_width=True)
