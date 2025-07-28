import streamlit as st

def render():
    st.title("Logistic Regression - Theory")

    st.write("""
    Logistic Regression is a **supervised learning algorithm** used for **classification** tasks — especially binary classification (yes/no, 0/1, true/false).

    Unlike linear regression which predicts continuous values, logistic regression predicts the **probability** that a given input belongs to a particular class.
    """)

    st.header("How It Works")
    st.markdown("""
    - Instead of fitting a straight line, logistic regression fits a **sigmoid curve**.
    - The output is a value between 0 and 1, interpreted as probability.
    - A threshold (commonly 0.5) is applied to make a final classification.

    **Sigmoid Function:**
    \n
    $$
    \sigma(z) = \\frac{1}{1 + e^{-z}}
    $$

    where `z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ`
    """)

    st.subheader("Use Cases")
    st.markdown("""
    - Email Spam Detection
    - Disease Prediction
    - Customer Churn Prediction
    - Binary Sentiment Classification
    """)
