# ==============================
# app.py - AlgoLab Main Script
# ==============================

import streamlit as st



# Internal imports
import model_comparison
from data_handler.upload_validate import upload_and_validate, generate_dataset, toy_dataset
from unsupervised_algorithms.unsupervised_module import unsupervised
from Supervised_algorithms.Supervised_learning import supervised 
# ==============================
# Page configuration
# ==============================
st.set_page_config(
    page_title="Algo Lab",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==============================
# App Header
# ==============================
st.title("🔬 AlgoLabs – Visualize and Learn")

st.markdown("""
<div style='padding:12px; border-left:5px solid black;
background-color: rgba(74,144,226,0.1); font-style:italic;'>
"You are the average of the five people you spend time with"<br>
<b>— Jim Rohn</b>
</div>
""", unsafe_allow_html=True)

# ==============================
# Tabs
# ==============================
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Home",
    "🤖 Supervised Learning",
    "🧭 Unsupervised Learning",
    "📊 Model Comparison"
  
])

# ==============================
# Sidebar – Dataset / Image
# ==============================
with st.sidebar:
    st.header("📂 Dataset Options")

    choice = st.radio(
        "Choose option:",
        ["Upload Dataset", "Generate Dataset",],
        index=0
    )

    if choice == "Upload Dataset":
        df = upload_and_validate()
        if df is not None:
            st.session_state.df = df

    elif choice == "Generate Dataset":
        df = generate_dataset()
        if df is not None:
            st.session_state.df = df

    
    

# ==============================
# 🏠 Home
# ==============================
with tab1:
    st.subheader("Welcome to AlgoLab 👋")
    if "df" in st.session_state:
        st.dataframe(st.session_state.df.head())
    else:
        st.info("Upload or generate a dataset to begin.")

# ==============================
# 🤖 Supervised Learning (Streamlit)
# ==============================
with tab2:
    st.subheader("Supervised Learning")
    if "df" in st.session_state:
        st.session_state.uploaded_data = st.session_state.df
        supervised()
    else:
        st.info("Upload a dataset first.")

# ==============================
# 🧭 Unsupervised Learning
# ==============================
with tab3:
    st.subheader("Unsupervised Learning")
    if "df" in st.session_state:
        st.session_state.uploaded_data = st.session_state.df
        unsupervised()
    else:
        st.info("Upload a dataset first.")

# ==============================
# 📊 Model Comparison
# ==============================
with tab4:
    model_comparison.show()

    


   
# ==============================
# Footer
# ==============================
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: black;
        text-align: right;
        padding: 8px 16px;
        font-size: 14px;
        z-index: 999;
        border-top: 1px solid #e0e0e0;
    }

    /* Prevent content from being hidden behind footer */
    .block-container {
        padding-bottom: 60px;
    }
    </style>

    <div class="footer">
        © 2025 GGSOC ❤️
    </div>
    """,
    unsafe_allow_html=True
)
