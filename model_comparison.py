import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split

# Classification metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Regression metrics
from sklearn.metrics import mean_squared_error, r2_score

# Classification models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Regression models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def detect_task_type(y):
    """Auto-detect classification vs regression"""
    if y.dtype.kind in "ifu":
        if y.nunique() <= 20:
            return "classification"
        else:
            return "regression"
    return "classification"


def show():
    st.title("🔍 Model Comparison Dashboard")
    st.write("Compare multiple ML models with automatic task detection.")

     # ✅ Use dataset from sidebar
    if "df" not in st.session_state or st.session_state.df is None:
        st.info("Upload or generate a dataset from the sidebar to use Model Comparison.")
        return

    df = st.session_state.df
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Select target column
    target_col = st.selectbox(
        "Select Target Column",
        df.columns,
        key="model_comparison_target_col"
        )


    X = df.drop(columns=[target_col])
    y = df[target_col]


    task_type = detect_task_type(y)

    st.info(f"🧠 Detected Problem Type: **{task_type.upper()}**")

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    if task_type == "classification":
            models = {
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "SVM": SVC(),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier()
            }

            metrics_used = ["Accuracy", "Precision", "Recall", "F1 Score"]

    else:
            models = {
                "KNN": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "SVR": SVR(),
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor()
            }

            metrics_used = ["RMSE", "R² Score"]

    st.sidebar.subheader("Select Models to Compare")
    models_selected = st.sidebar.multiselect(
            "Choose algorithms",
            list(models.keys()),
            default=list(models.keys())[:2]
        )

    results = []

    for model_name in models_selected:
            model = models[model_name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if task_type == "classification":
                results.append({
                    "Model": model_name,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                    "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0)
                })
            else:
                results.append({
                    "Model": model_name,
                    "RMSE": mean_squared_error(y_test, y_pred, squared=False),
                    "R² Score": r2_score(y_test, y_pred)
                })

    results_df = pd.DataFrame(results)
    st.write("### 📊 Model Performance Comparison")
    st.dataframe(results_df, use_container_width=True)
