# Importing required library
import streamlit as st
from supervised_algo import knn_visualization
from supervised_algo import knn_theory

# Page configuration
st.set_page_config(
    page_title="KNN Visualizer",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# title  of the  page
st.title("🔬Algo Labs Visualize and Learn")

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
tab1, tab2, tab3 = st.tabs(["Visualize Data", "Supervised Learning", "Unsupervised Learning"])

with tab1:
    from algo_files.data_visualize import visualize

    visualize()

with tab2:
    view = st.sidebar.radio("Choose View", ["KNN Overview", "KNN Playground"])
    if view == "KNN Overview":
        from supervised_algo import knn_theory

        knn_theory.render()
    elif view == "KNN Playground":
        from supervised_algo import knn_visualization

        knn_visualization.render()

with tab3:
    from algo_files.unsupervised_learning import unsupervised

    unsupervised()

# Sidebar : Data Uploading and Data Generation
with st.sidebar:
    options = ["Upload Dataset", "Generate Dataset"]
    selected_option = st.radio("Choose your preferred option:", options, index=0)

    if selected_option == "Upload Dataset":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
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

