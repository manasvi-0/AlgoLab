
# Importing required library
import streamlit as st
import pandas as  pd

from pages import model_comparison

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
from sklearn.datasets import make_classification



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
# tab1, tab2, tab3 = st.tabs(["Home Page", "Supervised Learning", "Unsupervised Learning"])

tab1, tab2, tab3, tab4 = st.tabs([
    "Home Page", 
    "Supervised Learning", 
    "Unsupervised Learning", 
    "Model Comparison"  # ✅ new tab
])



with tab1:
    st.write("Veiw Dataframe")

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
    options = ["Upload Dataset", "Generate Dataset"]
    selected_option = st.radio("Choose your preferred option:", options, index=0)

    # ✅ Upload dataset with validation
    if selected_option == "Upload Dataset":
        df = upload_and_validate()

    # ✅ Generate synthetic dataset
    elif selected_option == "Generate Dataset":
        no_of_sample = st.slider("No. of Samples", 10, 2000, 100)
        no_of_feature = st.slider("No. of Features", 2, 20, 2)
        noise_level = st.slider("Noise Level (%)", 0.0, 50.0, 5.0)
        no_of_class = st.number_input("No. of Classes", min_value=2, max_value=10, value=2)
        class_separation = st.slider("Class Separation", 0.50, 2.00, 1.0)

        if st.button("Generate Dataset"):
            # Calculate appropriate feature distribution to avoid constraint violation
            n_informative = max(1, min(no_of_feature - 1, no_of_feature // 2))
            n_redundant = max(0, min(no_of_feature - n_informative - 1, no_of_feature // 4))
            n_repeated = max(0, no_of_feature - n_informative - n_redundant - 1)
            
            X, y = make_classification(
                n_samples=no_of_sample,
                n_features=no_of_feature,
                n_informative=n_informative,
                n_redundant=n_redundant,
                n_repeated=n_repeated,
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

# ==============================
# 🏠 Tab 1: Home Page
# ==============================
with tab1:
    st.write("Welcome to AlgoLab! 👋")
    if df is not None:
        st.subheader("📄 Current Dataset Preview")
        st.dataframe(df.head())
    else:
        st.info("Upload or generate a dataset to preview here.")

# ==============================
# 🤖 Tab 2: Supervised Learning
# ==============================
with tab2:
    st.write("### Supervised Learning Playground")
    if df is not None:
        interactive_model_tuning(df)
    else:
        st.info("Upload or generate a dataset first to start tuning models.")

# ==============================
# 🚧 Tab 3: Unsupervised Learning
# ==============================
with tab3:
    st.write("Unsupervised module is under development.")


# ==============================
# 📊 Tab 4: Model Comparison
# ==============================
with tab4:
    model_comparison.show()
   

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
