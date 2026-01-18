import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, Birch
)
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.datasets import make_blobs, make_circles, make_swiss_roll
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from scipy.cluster.hierarchy import dendrogram, linkage

import warnings
warnings.filterwarnings("ignore")


# =====================================================
# 🔹 SINGLE SOURCE OF TRUTH (SIDEBAR DATA)
# =====================================================
def get_sidebar_data():
    if "df" in st.session_state and st.session_state.df is not None:
        return st.session_state.df
    return None


def get_numeric_X(df, min_cols=2):
    if df is None:
        return None
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) < min_cols:
        return None
    return df[num_cols].values


# =====================================================
# 🔹 MAIN ENTRY
# =====================================================
def unsupervised():
    st.subheader("🧭 Unsupervised Learning Playground")

    df = get_sidebar_data()
    if df is not None:
        st.caption(f"📊 Dataset from sidebar | Shape: {df.shape}")
        df = st.session_state.df
        st.write("### Dataset Preview")
        st.dataframe(df.head())
    else:
        st.warning("⚠️ No dataset from sidebar — using sample data")

    algo = st.selectbox(
        "Choose Algorithm",
        [
            "K-Means",
            "DBSCAN",
            "PCA",
            "Hierarchical Clustering",
            "Gaussian Mixture",
            "Spectral Clustering",
            "t-SNE",
            "UMAP",
            "Isolation Forest",
            "Locally Linear Embedding",
            "Factor Analysis",
            "Birch Clustering",
        ],
        key="unsup_algo"
    )

    if algo == "K-Means":
        kmeans_playground(df)
    elif algo == "DBSCAN":
        dbscan_playground(df)
    elif algo == "PCA":
        pca_playground(df)
    elif algo == "Hierarchical Clustering":
        hierarchical_playground(df)
    elif algo == "Gaussian Mixture":
        gmm_playground(df)
    elif algo == "Spectral Clustering":
        spectral_playground(df)
    elif algo == "t-SNE":
        tsne_playground(df)
    elif algo == "UMAP":
        umap_playground(df)
    elif algo == "Isolation Forest":
        isolation_forest_playground(df)
    elif algo == "Locally Linear Embedding":
        lle_playground(df)
    elif algo == "Factor Analysis":
        factor_analysis_playground(df)
    elif algo == "Birch Clustering":
        birch_playground(df)


# =====================================================
# 🔹 K-MEANS
# =====================================================
def kmeans_playground(df):
    st.subheader("K-Means Clustering")

    X = get_numeric_X(df)
    if X is None:
        X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
        st.info("Using sample blob data")

    k = st.slider("Clusters", 2, 10, 3)
    if st.button("Run K-Means"):
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X[:, :2])

        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
        ax.scatter(
            model.cluster_centers_[:, 0],
            model.cluster_centers_[:, 1],
            c="red",
            s=150,
            marker="X"
        )
        ax.set_title("K-Means Result")
        st.pyplot(fig)


# =====================================================
# 🔹 DBSCAN
# =====================================================
def dbscan_playground(df):
    st.subheader("DBSCAN")

    X = get_numeric_X(df)
    if X is None:
        X, _ = make_blobs(n_samples=300, centers=3, random_state=42)
        st.info("Using sample data")

    eps = st.slider("Epsilon", 0.1, 2.0, 0.5)
    min_samples = st.slider("Min Samples", 2, 20, 5)

    if st.button("Run DBSCAN"):
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X[:, :2])
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10")
        ax.set_title("DBSCAN Result")
        st.pyplot(fig)


# =====================================================
# 🔹 PCA
# =====================================================
def pca_playground(df):
    st.subheader("PCA")

    X = get_numeric_X(df, min_cols=2)
    if X is None:
        X, _ = make_blobs(n_samples=300, n_features=5, random_state=42)
        st.info("Using sample high-dimensional data")

    n = st.slider("Components", 2, min(5, X.shape[1]), 2)

    if st.button("Run PCA"):
        X_scaled = StandardScaler().fit_transform(X)
        X_pca = PCA(n_components=n).fit_transform(X_scaled)

        fig, ax = plt.subplots()
        ax.scatter(X_pca[:, 0], X_pca[:, 1])
        ax.set_title("PCA Projection")
        st.pyplot(fig)


# =====================================================
# 🔹 HIERARCHICAL
# =====================================================
def hierarchical_playground(df):
    st.subheader("Hierarchical Clustering")

    X = get_numeric_X(df)
    if X is None:
        X, _ = make_blobs(n_samples=200, random_state=42)

    k = st.slider("Clusters", 2, 8, 3)

    if st.button("Run Hierarchical"):
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(X[:, :2])
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], c=labels)
        st.pyplot(fig)


# =====================================================
# 🔹 GMM
# =====================================================
def gmm_playground(df):
    st.subheader("Gaussian Mixture")

    X = get_numeric_X(df)
    if X is None:
        X, _ = make_blobs(n_samples=300, random_state=42)

    k = st.slider("Components", 2, 6, 3)

    if st.button("Run GMM"):
        labels = GaussianMixture(n_components=k).fit_predict(X[:, :2])
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], c=labels)
        st.pyplot(fig)


# =====================================================
# 🔹 SPECTRAL
# =====================================================
def spectral_playground(df):
    st.subheader("Spectral Clustering")

    X = get_numeric_X(df)
    if X is None:
        X, _ = make_circles(n_samples=300, noise=0.05)

    k = st.slider("Clusters", 2, 6, 2)

    if st.button("Run Spectral"):
        labels = SpectralClustering(n_clusters=k).fit_predict(X[:, :2])
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], c=labels)
        st.pyplot(fig)


# =====================================================
# 🔹 TSNE
# =====================================================
def tsne_playground(df):
    st.subheader("t-SNE")

    X = get_numeric_X(df, min_cols=3)
    if X is None:
        X, _ = make_blobs(n_samples=300, n_features=5)

    if st.button("Run t-SNE"):
        X_tsne = TSNE(n_components=2).fit_transform(X)
        fig, ax = plt.subplots()
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1])
        st.pyplot(fig)


# =====================================================
# 🔹 UMAP
# =====================================================
def umap_playground(df):
    st.subheader("UMAP")

    try:
        import umap
    except ImportError:
        st.error("Install umap-learn")
        return

    X = get_numeric_X(df)
    if X is None:
        X, _ = make_blobs(n_samples=300)

    if st.button("Run UMAP"):
        X_umap = umap.UMAP().fit_transform(X)
        fig, ax = plt.subplots()
        ax.scatter(X_umap[:, 0], X_umap[:, 1])
        st.pyplot(fig)


# =====================================================
# 🔹 ISOLATION FOREST
# =====================================================
def isolation_forest_playground(df):
    st.subheader("Isolation Forest")

    X = get_numeric_X(df)
    if X is None:
        X, _ = make_blobs(n_samples=300)

    if st.button("Detect Anomalies"):
        labels = IsolationForest().fit_predict(X)
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], c=labels)
        st.pyplot(fig)


# =====================================================
# 🔹 LLE
# =====================================================
def lle_playground(df):
    st.subheader("Locally Linear Embedding")

    X = get_numeric_X(df, min_cols=3)
    if X is None:
        X, _ = make_swiss_roll(n_samples=1000)

    if st.button("Run LLE"):
        X_lle = LocallyLinearEmbedding(n_components=2).fit_transform(X)
        fig, ax = plt.subplots()
        ax.scatter(X_lle[:, 0], X_lle[:, 1])
        st.pyplot(fig)


# =====================================================
# 🔹 FACTOR ANALYSIS
# =====================================================
def factor_analysis_playground(df):
    st.subheader("Factor Analysis")

    X = get_numeric_X(df, min_cols=3)
    if X is None:
        X, _ = make_blobs(n_samples=300, n_features=5)

    if st.button("Run Factor Analysis"):
        X_fa = FactorAnalysis(n_components=2).fit_transform(X)
        fig, ax = plt.subplots()
        ax.scatter(X_fa[:, 0], X_fa[:, 1])
        st.pyplot(fig)


# =====================================================
# 🔹 BIRCH
# =====================================================
def birch_playground(df):
    st.subheader("Birch Clustering")

    X = get_numeric_X(df)
    if X is None:
        X, _ = make_blobs(n_samples=300)

    if st.button("Run Birch"):
        labels = Birch(n_clusters=3).fit_predict(X)
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], c=labels)
        st.pyplot(fig)
