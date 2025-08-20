import pandas as pd
import numpy as np
import streamlit as st
from sklearn.datasets import *

def toy_dataset():
    dataset=st.selectbox("select  a  dataset",["iris","digits","wine","breast cancer","diabetes","boston"],accept_new_options=False)
    

    if  st.button("Import Toy Dataset",type="primary"):
        st.write("You have choosen: ",dataset)

        all_datasets = {
        'iris': load_iris(),
        'digits': load_digits(),
        'wine': load_wine(),
        'breast cancer': load_breast_cancer(),
        'diabetes': load_diabetes(),
        'boston': fetch_openml(name='boston', version=1)
        }

        df=all_datasets[dataset]
        
        combined = np.c_[df.data, df.target]  # Horizontally stack arrays
        columns = df.feature_names + ['target']
        df = pd.DataFrame(combined, columns=columns)
        st.success("✅Data imported successfully!")
        st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.balloons()
        st.session_state.df=df
    else:
        st.info("Select a Dataset")
        return None


def upload_and_validate():
    """Uploads and validates a CSV file with error handling."""
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            #  Validation: Empty file
            if df.empty:
                st.error(" The uploaded CSV is empty.")
                return None

            #  Validation: Missing Headers
            if df.columns.str.contains("Unnamed").any():
                st.error(" The CSV seems to have missing or unnamed headers.")
                return None

            #  Info block after successful validation
            st.success("✅File uploaded and validated successfully!")
            st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.balloons()

            #  Warning for missing values
            if df.isnull().values.any():
                st.warning(" Missing values detected!")

            #  Storing data  
            st.session_state.df=df

        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None
    else:
        st.info(" Please upload a CSV file.")
        return None
    

def generate_dataset():
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
            st.balloons()
            st.session_state.df=df
