import streamlit as st
import pandas as pd

def upload_and_validate():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Validation: Empty
            if df.empty:
                st.error(" The uploaded CSV is empty.")
                return None

            # Validation: Missing Headers
            if df.columns.str.contains("Unnamed").any():
                st.error(" The CSV seems to have missing or unnamed headers.")
                return None

            # Info block
            st.success(" File uploaded and validated successfully!")
            st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

            if df.isnull().values.any():
                st.warning(" Missing values detected!")

            # Preview
            st.subheader(" Preview (Top 10 Rows)")
            st.dataframe(df.head(10))

            return df

        except Exception as e:
            st.error(f" Error reading file: {e}")
            return None
    else:
        st.info("Please upload a CSV file.")
        return None
