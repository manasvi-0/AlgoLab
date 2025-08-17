
# Importing required library
import streamlit as st
import pandas as  pd

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
tab1, tab2, tab3 = st.tabs(["Home Page", "Supervised Learning", "Unsupervised Learning"])



#Unsupervised Learning
with tab3:
    from unsupervised_algorithms.unsupervised_module import unsupervised
    # Store uploaded data in session state for unsupervised algorithms
    if 'df' in locals() and df is not None:
        st.session_state.uploaded_data = df
    unsupervised()

# ✅ Global variable to store dataset
df = None

# ==============================
# 📂 Sidebar - Upload or Generate Dataset
# ==============================
with st.sidebar:
    st.header("📂 Dataset Options")
    options = ["Toy Dataset","Upload Dataset", "Generate Dataset"]
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

# ==============================
# 🚧 Tab 3: Unsupervised Learning
# ==============================
with tab3:
    st.write("Unsupervised module is under development.")

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
