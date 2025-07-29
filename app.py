import streamlit as st
import pandas as pd
from unsupervised_algos.kmeans_clustering import run_kmeans
from unsupervised_algos.dbscan_clustering import run_dbscan

# import upload_validate() from data validation
from data_handler.upload_validate import upload_and_validate

# Page config
st.set_page_config(
    page_title="Algo Lab",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# title of the page
st.title("🔬 Algo Labs Visualize and Learn")

# Quote box
st.markdown(f"""
<div style='
    padding: 12px;
    border-left: 5px solid black;
    background-color: rgba(74, 144, 226, 0.1);
    color: inherit;
    font-style: italic;
'>
"You are the average of the five people you spend time with"<br><b>— Jim Rohn</b>
</div>
""", unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3 = st.tabs(["Home Page", "Supervised Learning", "Unsupervised Learning"])

with tab1:
    st.write("View Dataframe")

# Supervised Learning
with tab2:
    st.write("Supervised Learning")
    options = ["KNN", "Decision Tree", "Logestic Regression", "SVM"]
    selected_option = st.selectbox("Choose an option:", options)

    st.write("You have selected:", selected_option)

    if selected_option == "KNN":
        view = st.radio("Choose View", ["KNN Overview", "KNN Playground"])
        if view == "KNN Overview":
            from supervised_algo.KNN import knn_theory
            knn_theory.render()
        elif view == "KNN Playground":
            from supervised_algo.KNN import knn_visualization
            knn_visualization.render()

# Unsupervised Learning
with tab3:
    st.write("### Unsupervised Learning")

    # File upload for clustering
    uploaded_file = st.file_uploader("Upload your CSV file for clustering", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("### Data Preview", data.head())

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

# Sidebar : Data Uploading and Data Generation
with st.sidebar:
    options = ["Upload Dataset", "Generate Dataset"]
    selected_option = st.radio("Choose your preferred option:", options, index=0)

    if selected_option == "Upload Dataset":
        file = st.file_uploader("Choose a CSV file", type="csv")
        from data_handler.upload_validate import upload_file
        with tab1:
            upload_file(file)

    if selected_option == "Upload Dataset":  # modified for data validation feature
        df = upload_and_validate()

    elif selected_option == "Generate Dataset":
        no_of_sample = st.slider("No. of Samples", 10, 2000)
        no_of_feature = st.slider("No. of Features", 2, 20)
        noise_level = st.slider("Noise Level", 0.00, 50.00)
        no_of_class = st.text_input("No. of Classes")
        class_separation = st.slider("Class Separation", 0.50, 2.00)

        def my_callback():
            st.write("Data Generated!")

        st.button("Generate Data", on_click=my_callback)

# Footer
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
    height:50px;
}
</style>
<div class="footer">
    <p>© 2025 GGSOC❤️ </p>
</div>
""", unsafe_allow_html=True)
