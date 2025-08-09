import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd # Added for type hinting and clarity

def run_kmeans(data: pd.DataFrame):
    st.subheader("KMeans Clustering")

    # Session state variables को इनिशियलाइज़ करें ताकि परिणाम बने रहें
    if 'silhouette_score' not in st.session_state:
        st.session_state.silhouette_score = None
    if 'kmeans_plot_fig' not in st.session_state:
        st.session_state.kmeans_plot_fig = None
    if 'clustered_data_head' not in st.session_state:
        st.session_state.clustered_data_head = None
    if 'kmeans_n_clusters' not in st.session_state: # पिछली बार उपयोग किए गए n_clusters को स्टोर करने के लिए
        st.session_state.kmeans_n_clusters = 3 # डिफ़ॉल्ट मान

    # st.form के लिए एक स्थिर key का उपयोग करें
    with st.form(key="kmeans_params_form"):
        # स्लाइडर का डिफ़ॉल्ट मान session_state से लें
        n_clusters = st.slider("Select number of clusters", 2, 10, st.session_state.kmeans_n_clusters, key="kmeans_n_clusters_slider")
        submitted = st.form_submit_button("Run KMeans Clustering")
        st.write("Data Types:", data.dtypes)

        if submitted:
            st.write("✅ Inside submit block")

            # इनपुट डेटा की कॉपी बनाएं
            input_data = data.copy()

            # यदि 'Cluster' कॉलम पहले से मौजूद है तो उसे हटा दें
            if "Cluster" in input_data.columns:
                input_data.drop("Cluster", axis=1, inplace=True)

            # KMeans फिट करें
            # KMeans के लिए n_init='auto' जोड़ा गया ताकि भविष्य की चेतावनी से बचा जा सके
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            kmeans.fit(input_data)

            input_data["Cluster"] = kmeans.labels_

            # सिलुएट स्कोर की गणना करें और session_state में स्टोर करें
            score = silhouette_score(input_data.iloc[:, :-1], input_data["Cluster"])
            st.session_state.silhouette_score = score
            st.session_state.kmeans_n_clusters = n_clusters # चुने गए n_clusters को स्टोर करें

            # प्लॉट बनाएं और figure को session_state में स्टोर करें
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(input_data.iloc[:, 0], input_data.iloc[:, 1], c=input_data["Cluster"], cmap='viridis')
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.set_title("KMeans Clustering")
            st.session_state.kmeans_plot_fig = fig

            # क्लस्टर किए गए डेटा का head session_state में स्टोर करें
            st.session_state.clustered_data_head = input_data.head()

    # परिणाम 'if submitted' ब्लॉक के बाहर प्रदर्शित करें, लेकिन तभी जब वे session_state में मौजूद हों
    if st.session_state.silhouette_score is not None:
        st.write(f"### Silhouette Score: {st.session_state.silhouette_score:.2f}")

    if st.session_state.kmeans_plot_fig is not None:
        st.pyplot(st.session_state.kmeans_plot_fig)
        plt.close(st.session_state.kmeans_plot_fig) # मेमोरी लीक से बचने के लिए figure को बंद करें

    if st.session_state.clustered_data_head is not None:
        st.write("### Clustered Data", st.session_state.clustered_data_head)