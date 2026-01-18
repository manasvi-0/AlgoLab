import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


# ======================================================
# TASK TYPE DETECTION
# ======================================================
def detect_task_type(df: pd.DataFrame, target_col: str):
    y = df[target_col]

    if y.dtype == "object" or y.dtype.name == "category":
        return "Classification"

    if y.nunique() <= 10:
        return "Classification"

    return "Regression"


# ======================================================
# MAIN ENTRY
# ======================================================
def supervised():
    st.write("Supervised Learning")

    if "uploaded_data" not in st.session_state:
        st.info("Upload a dataset from the sidebar to begin.")
        return
    df = st.session_state.df
    st.write("### Dataset Preview")
    st.dataframe(df.head())
    df = st.session_state.uploaded_data

    # ------------------------------
    # Target selection
    # ------------------------------
    target_col = st.selectbox("Select Target Column", df.columns)

    task_type = detect_task_type(df, target_col)
    st.success(f"Detected Task Type: **{task_type}**")

    # ------------------------------
    # Algorithm options
    # ------------------------------
    if task_type == "Classification":
        algorithms = [
            "Logistic Regression",
            "Decision Tree",
            "Support Vector Machine",
            "K-Nearest Neighbors"
        ]
    else:
        algorithms = [
            "Linear Regression",
            "Decision Tree",
            "Support Vector Regression",
            "K-Nearest Neighbors"
        ]

    selected_algo = st.selectbox("Choose Algorithm", algorithms)

    view = st.radio(
        "Choose View",
        [f"{selected_algo} Overview", f"{selected_algo} Playground"]
    )

    if "Overview" in view:
        show_overview(selected_algo, task_type)
    else:
        playground(selected_algo, task_type, df, target_col)


# ======================================================
# OVERVIEWS
# ======================================================
def show_overview(algo, task):
    st.subheader(f"{algo} Overview")

    descriptions = {
        "Logistic Regression": "Used for classification using probability estimation.",
        "Linear Regression": "Predicts continuous numeric values.",
        "Decision Tree": "Tree-based model for both classification and regression.",
        "Support Vector Machine": "Max-margin classifier.",
        "Support Vector Regression": "Margin-based regression.",
        "K-Nearest Neighbors": "Distance-based learning algorithm."
    }

    st.write(descriptions.get(algo, ""))
    st.write(f"**Task Type:** {task}")


# ======================================================
# PLAYGROUND
# ======================================================
def playground(algo, task, df, target_col):

    # ------------------------------
    # Feature handling
    # ------------------------------
    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number])

    if X.shape[1] < 2:
        st.warning("Need at least 2 numeric feature columns.")
        return

    y = df[target_col]

    X = X.values
    y = y.values

    # ------------------------------
    # Model selection
    # ------------------------------
    if algo == "Logistic Regression":
        model = LogisticRegression()

    elif algo == "Linear Regression":
        model = LinearRegression()

    elif algo == "Decision Tree":
        depth = st.slider("Max Depth", 1, 10, 3)
        model = (
            DecisionTreeClassifier(max_depth=depth)
            if task == "Classification"
            else DecisionTreeRegressor(max_depth=depth)
        )

    elif algo == "Support Vector Machine":
        C = st.slider("C", 0.1, 10.0, 1.0)
        model = SVC(C=C)

    elif algo == "Support Vector Regression":
        C = st.slider("C", 0.1, 10.0, 1.0)
        model = SVR(C=C)

    elif algo == "K-Nearest Neighbors":
        k = st.slider("K Neighbors", 1, 15, 5)
        model = (
            KNeighborsClassifier(n_neighbors=k)
            if task == "Classification"
            else KNeighborsRegressor(n_neighbors=k)
        )

    else:
        st.error("Unsupported algorithm")
        return

    # ------------------------------
    # Train button
    # ------------------------------
    if st.button("Train Model"):

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.25, random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ------------------------------
        # Visualization
        # ------------------------------
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_pred,
            cmap="viridis",
            alpha=0.7
        )
        ax.set_title("Model Predictions")
        plt.colorbar(scatter)
        st.pyplot(fig)

        # ------------------------------
        # Metrics
        # ------------------------------
        if task == "Classification":
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            st.metric("Accuracy", f"{acc:.3f}")
            st.write("Confusion Matrix")
            st.dataframe(cm)

        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.metric("MSE", f"{mse:.2f}")
            st.metric("R²", f"{r2:.3f}")


# ======================================================
# STANDALONE RUN
# ======================================================
if __name__ == "__main__":
    st.set_page_config(page_title="Supervised Learning", page_icon="🤖")
    st.title("🤖 Supervised Learning Algorithms")
    supervised()
