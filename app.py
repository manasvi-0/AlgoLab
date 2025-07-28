import streamlit as st
import pandas as pd
from unsupervised_algos.kmeans_clustering import run_kmeans
from unsupervised_algos.dbscan_clustering import run_dbscan

st.title("🧠 Clustering Playground")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview", data.head())

    # Session state initialization for selected algo
    if 'selected_algo' not in st.session_state:
        st.session_state.selected_algo = "KMeans"

    # Algorithm dropdown
    st.session_state.selected_algo = st.selectbox(
        "Choose Clustering Algorithm",
        ["KMeans", "DBSCAN"],
        index=["KMeans", "DBSCAN"].index(st.session_state.selected_algo)
    )

    # Algo caller
    if st.session_state.selected_algo == "KMeans":
        run_kmeans(data)
    elif st.session_state.selected_algo == "DBSCAN":
        run_dbscan(data)
