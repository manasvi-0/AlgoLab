import streamlit as st
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import uuid
import pandas as pd

def run_dbscan(data: pd.DataFrame):
    st.subheader("DBSCAN Clustering")

    # Session state variables
    if 'dbscan_eps' not in st.session_state:
        st.session_state.dbscan_eps = 0.5
    if 'dbscan_min_samples' not in st.session_state:
        st.session_state.dbscan_min_samples = 5
    if 'dbscan_score' not in st.session_state:
        st.session_state.dbscan_score = None
    if 'dbscan_plot_fig' not in st.session_state:
        st.session_state.dbscan_plot_fig = None
    if 'dbscan_clustered_data' not in st.session_state:
        st.session_state.dbscan_clustered_data = None

    # UI form
    with st.form(key="dbscan_form"):
        eps = st.slider("Epsilon (eps)", 0.1, 5.0, st.session_state.dbscan_eps, step=0.1)
        min_samples = st.slider("Min Samples", 1, 20, st.session_state.dbscan_min_samples)
        submitted = st.form_submit_button("Run DBSCAN")

        if submitted:
            st.write("✅ Inside DBSCAN submit")

            input_data = data.copy()
            input_data = input_data.apply(pd.to_numeric, errors='coerce')
            input_data.dropna(inplace=True)

            # Drop old cluster column
            if "Cluster" in input_data.columns:
                input_data.drop("Cluster", axis=1, inplace=True)

            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(input_data)
            input_data["Cluster"] = labels

            st.session_state.dbscan_eps = eps
            st.session_state.dbscan_min_samples = min_samples
            st.session_state.dbscan_clustered_data = input_data.head()

            if len(set(labels)) > 1 and -1 not in set(labels):
                score = silhouette_score(input_data.iloc[:, :-1], labels)
                st.session_state.dbscan_score = score
            else:
                st.session_state.dbscan_score = "Not enough clusters to compute score"

            # Plot
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(input_data.iloc[:, 0], input_data.iloc[:, 1], c=labels, cmap='plasma')
            ax.set_title("DBSCAN Clustering")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            st.session_state.dbscan_plot_fig = fig

    # Result display (outside submit)
    if st.session_state.dbscan_score is not None:
        if isinstance(st.session_state.dbscan_score, str):
            st.warning(st.session_state.dbscan_score)
        else:
            st.write(f"### Silhouette Score: {st.session_state.dbscan_score:.2f}")

    if st.session_state.dbscan_plot_fig:
        st.pyplot(st.session_state.dbscan_plot_fig)
        plt.close(st.session_state.dbscan_plot_fig)

    if st.session_state.dbscan_clustered_data is not None:
        st.write("### Clustered Data", st.session_state.dbscan_clustered_data)
