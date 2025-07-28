import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def render():
    st.title("Logistic Regression Playground")
    st.markdown("Use the controls in the sidebar to adjust the data and see how Logistic Regression behaves:")

    # Sidebar controls
    num_samples = st.sidebar.slider("Number of Samples", 100, 1000, 300)
    test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20) / 100
    noise = st.sidebar.slider("Noise Level (Flip Labels)", 0.0, 0.5, 0.1)
    run = st.sidebar.button("Run Algorithm")

    if run:
        # Generate synthetic classification data
        X, y = make_classification(
            n_samples=num_samples,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=1,
            flip_y=noise,
            class_sep=1.0,
            random_state=42
        )

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Train logistic regression
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Plot decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        fig, ax = plt.subplots()
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', label='Train', edgecolor='k')
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='*', cmap='coolwarm', label='Test', edgecolors='k')
        ax.set_title("Decision Boundary")
        ax.legend()
        st.pyplot(fig)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        st.subheader("Model Performance")
        st.metric("Accuracy", f"{acc:.2%}")

        st.subheader("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax_cm)
        st.pyplot(fig_cm)
