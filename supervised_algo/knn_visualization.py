import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def render():
    st.title("K-Nearest Neighbors (KNN) Playground")

    st.markdown("Adjust the hyperparameters below to see how KNN behaves:")

    # Sidebar controls
    k = st.sidebar.slider("Number of Neighbors (k)", 1, 15, 3)
    test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20) / 100
    distance_metric = st.sidebar.selectbox("Distance Metric", ["euclidean", "manhattan", "minkowski"])
    show_boundary = st.sidebar.radio("Show decision boundary", ['Yes', 'No'])
    run = st.sidebar.button("Run Algorithm")
    if run:
        # Generate dataset
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=300,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=2,
            flip_y=0.2,
            class_sep=0.8,
            weights=[0.6, 0.4],
            random_state=42
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Train model
        model = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Plot decision boundaries
        if show_boundary=='Yes':
            h = .02
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            fig, ax = plt.subplots()
            ax.contourf(xx, yy, Z, alpha=0.3)
        else:
            fig, ax = plt.subplots()

        #showing training/testing data
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, label='Train', cmap='coolwarm', edgecolor='k')
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='*', label='Test', cmap='coolwarm', edgecolors='k')
        ax.set_title(f"Decision Boundary (k={k}, Metric={distance_metric})")
        ax.legend()
        st.pyplot(fig)

        # Show metrics
        st.subheader("Model Performance")
        st.metric(label="Accuracy",value=f"{acc:.2%}")

        st.subheader("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax_cm)
        st.pyplot(fig_cm)


