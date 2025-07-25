import streamlit as  st

def supervised():
    st.title("Supervised  Learning")

  
    option = st.selectbox(
                    "Choose your Algorithm:",
                    ("KNN", "Decision Tree", "Logistic Regression", " SVM ")
                )

    st.write("You have selected:", option)