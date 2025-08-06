import streamlit as st
import pandas as pd
from unsupervised_algos.kmeans_clustering import run_kmeans
from unsupervised_algos.dbscan_clustering import run_dbscan
from data_handler.upload_validate import upload_and_validate
from sklearn.datasets import make_classification
from supervised_module import interactive_model_tuning

st.set_page_config(
    page_title="Algo Lab",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🔬 Algo Labs - Visualize and Learn")

st.markdown("""
<div style='padding: 12px; border-left: 5px solid black;
            background-color: rgba(74, 144, 226, 0.1);
            font-style: italic;'>
"You are the average of the five people you spend time with"<br>
<b>— Jim Rohn</b>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Home Page", "Supervised Learning", "Unsupervised Learning"])

df = None

with st.sidebar:
    st.header("📂 Dataset Options")
    options = ["Upload Dataset", "Generate Dataset"]
    selected_option = st.radio("Choose your preferred option:", options, index=0)

    if selected_option == "Upload Dataset":
        df = upload_and_validate()

    elif selected_option == "Generate Dataset":
        no_of_sample = st.slider("No. of Samples", 10, 2000, 100)
        no_of_feature = st.slider("No. of Features", 2, 20, 2)
        noise_level = st.slider("Noise Level (%)", 0.0, 50.0, 5.0)
        no_of_class = st.number_input("No. of Classes", min_value=2, max_value=10, value=2)
        class_separation = st.slider("Class Separation", 0.50, 2.00, 1.0)

        if st.button("Generate Dataset"):
            X, y = make_classification(
                n_samples=no_of_sample,
                n_features=no_of_feature,
                n_classes=no_of_class,
                n_clusters_per_class=1,
                class_sep=class_separation,
                flip_y=noise_level/100,
                random_state=42
            )
            df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])
            df["Target"] = y
            st.success("✅ Dataset Generated Successfully!")
            st.dataframe(df.head())

with tab1:
    st.write("Welcome to AlgoLab! 👋")
    if df is not None:
        st.subheader("📄 Current Dataset Preview")
        st.dataframe(df.head())
    else:
        st.info("Upload or generate a dataset to preview here.")

with tab2:
    st.write("### Supervised Learning Playground")
    if df is not None:
        interactive_model_tuning(df)
    else:
        st.info("Upload or generate a dataset first to start tuning models.")

with tab3:
    st.write("### Unsupervised Learning")

    if df is not None:
        st.write("### Data Preview", df.head())

        if 'selected_algo' not in st.session_state:
            st.session_state.selected_algo = "KMeans"

        st.session_state.selected_algo = st.selectbox(
            "Choose Clustering Algorithm",
            ["KMeans", "DBSCAN"],
            index=["KMeans", "DBSCAN"].index(st.session_state.selected_algo)
        )

        if st.session_state.selected_algo == "KMeans":
            run_kmeans(df)
        elif st.session_state.selected_algo == "DBSCAN":
            run_dbscan(df)
    else:
        st.info("Please upload or generate a dataset from the sidebar to use Unsupervised Learning algorithms.")


st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f0f2f6;
    color: black;
    text-align: right;
    padding: 10px;
    border-top: 1px solid #e0e0e0;
    height: 50px;
}
</style>
<div class="footer">
    <p>© 2025 GGSOC ❤️</p>
</div>
""", unsafe_allow_html=True)
