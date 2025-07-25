# trial page
import streamlit as st
import  pandas as  pd

def visualize():
    st.title("Visualize the data")


    data = {'col1': [1, 2, 3], 'col2': ['A', 'B', 'C']}
    df = pd.DataFrame(data)

    st.dataframe(df)