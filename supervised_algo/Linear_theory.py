import streamlit as st

def render():
    st.title("Linear Regression - Theory")
    st.write("""
    Linear Regression is a **supervised learning algorithm** used for predicting a **continuous** outcome based on one or more input features.
    It tries to find the best-fit straight line through the data points using the equation:

    **y = mx + c**

    Where:
    - `y`: predicted value
    - `m`: slope (weight)
    - `x`: input feature
    - `c`: intercept (bias)

    """)

    st.header("Working of Linear Regression")
    st.markdown("""
    1. Linear regression estimates coefficients (slope and intercept) that minimize the squared difference between actual and predicted values.
    2. The result is a straight line that best fits the training data.
    """)

    st.write("**Use Cases:** Predicting prices, forecasting trends, etc.")
