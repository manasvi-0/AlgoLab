import  pandas  as pd
import streamlit as st


def upload_file(file):  
     if file  is not None:
            if file.name.endswith('.csv'):
                df=pd.read_csv(file)
                st.dataframe(df.head())