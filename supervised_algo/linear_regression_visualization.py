import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def render():
    st.title("Linear Regression Playground")
    st.markdown("Adjust the data points and noise level to see how Linear Regression behaves:")

    # Sidebar controls
    num_samples = st.sidebar.slider("Number of Samples", 20, 500, 100)
    noise = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.2)
    run = st.sidebar.button("Run Algorithm")

    if run:
        # Generate synthetic data
        X = 2 * np.random.rand(num_samples, 1)
        y = 4 + 3 * X + np.random.randn(num_samples, 1) * noise

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Plot
        fig, ax = plt.subplots()
        ax.scatter(X_test, y_test, color='blue', label='Test Data')
        ax.plot(X_test, y_pred, color='red', linewidth=2, label='Prediction Line')
        ax.legend()
        st.pyplot(fig)

        st.subheader("Model Performance")
        st.write(f"**R² Score**: {r2_score(y_test, y_pred):.2f}")
        st.write(f"**Mean Squared Error**: {mean_squared_error(y_test, y_pred):.2f}")
