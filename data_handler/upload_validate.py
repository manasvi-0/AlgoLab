import pandas as pd
import streamlit as st

def upload_file(file):
    """Displays uploaded CSV data preview."""
    if file is not None:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
            st.dataframe(df.head())

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
            st.success(" File uploaded and validated successfully!")
            st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

            #  Warning for missing values
            if df.isnull().values.any():
                st.warning(" Missing values detected!")

            #  Data preview
            st.subheader(" Preview (Top 10 Rows)")
            st.dataframe(df.head(10))

            return df

        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None
    else:
        st.info(" Please upload a CSV file.")
        return None
