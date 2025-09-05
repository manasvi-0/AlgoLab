
# Importing required library
import streamlit as st
import pandas as  pd
import numpy as np

#import upload_validate() from data validation

# ==============================
#  app.py - AlgoLab Main Script
#  ----------------------------
#  - Handles UI and navigation
#  - Dataset Upload & Generation
#  - Calls interactive_model_tuning()
# ==============================

import streamlit as st
import pandas as pd
from Supervised_algorithms.supervised_module import interactive_model_tuning

from data_handler.upload_validate import upload_and_validate

from data_handler.upload_validate import generate_dataset
from data_handler.upload_validate import toy_dataset

from sklearn.datasets import make_classification

import cv2
from cnn.kernels import visualiseImage

# Import unsupervised learning module
from unsupervised_algorithms.unsupervised_module import unsupervised


# ✅ Page configuration

st.set_page_config(
    page_title="Algo Lab",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ✅ App Title
st.title("🔬 Algo Labs - Visualize and Learn")

# ✅ Motivational Quote Box
st.markdown("""
<div style='padding: 12px; border-left: 5px solid black;
            background-color: rgba(74, 144, 226, 0.1);
            font-style: italic;'>
"You are the average of the five people you spend time with"<br>
<b>— Jim Rohn</b>
</div>
""", unsafe_allow_html=True)

# ✅ Tabs for navigation
tab1, tab2, tab3, tab4 = st.tabs(["Home Page", "Supervised Learning", "Unsupervised Learning", "Convolutional Neural Networks"])



with tab1:
    st.write("View Dataframe")

#Supervised  Learning
with tab2:
    st.write("Supervised  Learning")
    options = ["KNN", "Decision Tree", "Logestic Regression","SVM"]
    selected_option = st.selectbox("Choose an option:", options)

    st.write("You have  selected:", selected_option)

    # KNN Option selection
    #if selected_option=="KNN":
     #view = st.radio("Choose View", ["KNN Overview", "KNN Playground"])
     #if view == "KNN Overview":
        #from supervised_algo.KNN import knn_theory
        #knn_theory.render()
     #elif view == "KNN Playground":
         #from supervised_algo.KNN import knn_visualization
         #knn_visualization.render()


#Unsupervised Learning
with tab3:
    st.header("🧭 Unsupervised Learning")
    
    # Ensure uploaded data is available in session state
    if 'df' in st.session_state and st.session_state.df is not None:
        st.session_state.uploaded_data = st.session_state.df
    
    # Run the unsupervised module
    unsupervised()

with tab4:
    st.write("Upload an image to get started")

    # st.write(st.session_state)
    if "uploaded_image" in st.session_state:
        
        print("Hi")
        # Call your visualisation function
        visualiseImage(st.session_state.uploaded_image)


    
# ✅ Global variable to store dataset
df = None
image_upload = None
# ==============================
# 📂 Sidebar - Upload or Generate Dataset
# ==============================
with st.sidebar:
    st.header("📂 Dataset Options")

    options = ["Toy Dataset","Upload Dataset", "Generate Dataset"]

    options = ["Upload Dataset", "Generate Dataset", "Upload Image"]

    selected_option = st.radio("Choose your preferred option:", options, index=0)

    # ✅ Importing  Toy Dataset from Scikitlearn
    if selected_option=="Toy Dataset":
        toy_dataset()

    # ✅ Upload dataset with validation
    elif selected_option == "Upload Dataset":
        df = upload_and_validate()

    # ✅ Generate synthetic dataset
    elif selected_option == "Generate Dataset":
        df = generate_dataset()
        

    elif selected_option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image_upload = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                #visualiseImage(image_upload)
                print("success")

            except:
                st.error("The image cannot be opened")

            else:
                st.session_state.uploaded_image = image_upload
                # st.session_state["uploaded_image"] = image

# ==============================
# 🏠 Tab 1: Home Page
# ==============================
with tab1:
    st.write("Welcome to AlgoLab! 👋")
    if 'df' in st.session_state:
        st.subheader("📄 Current Dataset Preview")
        st.dataframe(st.session_state.df.head())
    else:
        st.info("Load, upload or generate a dataset to preview here.")

# ==============================
# 🤖 Tab 2: Supervised Learning
# ==============================
with tab2:
    st.write("### Supervised Learning Playground")
    if 'df' in st.session_state:
        interactive_model_tuning(st.session_state.df)
    else:
        st.info("Upload or generate a dataset first to start tuning models.")

# This section is now handled in the tab3 block above


# Sidebar theme toggle
theme = st.sidebar.radio("Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown(
        """
        <style>
        body {
            background-color: #121212;
            color: white;
        }
        .stButton button {
            background-color: #333333;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

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
    height: 50px;
}
</style>
<div class="footer">
    <p>© 2025 GGSOC ❤️</p>
</div>
""", unsafe_allow_html=True)
