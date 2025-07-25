import streamlit as  st

def unsupervised():
    st.title("Unsupervised  Learning")


    option = st.selectbox(
                    "Choose your Algorithm:",
                    ("K-Means", "DBSCAN", "PCA", "Hierarchical Clustering")
                )

    st.write("You have selected:", option)