import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def show():
    st.subheader("📊 Model Comparison")

    if "df" not in st.session_state or st.session_state.df is None:
        st.info("Upload or generate a dataset from the sidebar.")
        return

    df = st.session_state.df.copy()
    st.dataframe(df.head(), use_container_width=True)

    target_col = st.selectbox(
        "Select Target Column",
        df.columns,
        key="model_comp_target"
    )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Convert features
    X = pd.get_dummies(X, drop_first=True)

    # Detect task type
    is_regression = pd.api.types.is_numeric_dtype(y) and y.nunique() > 20
    task_type = "Regression" if is_regression else "Classification"

    st.success(f"Detected task type: **{task_type}**")

    # Encode target if classification
    if task_type == "Classification" and y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if task_type == "Classification":
        models = {
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "SVM": SVC(),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier()
        }
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor()
        }

    selected_models = st.multiselect(
        "Select models",
        list(models.keys()),
        default=list(models.keys()),
        key="model_select"
    )

    results = []

    for name in selected_models:
        model = models[name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if task_type == "Classification":
            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            })
        else:
            results.append({
                "Model": name,
                "RMSE": root_mean_squared_error(y_test, y_pred),
                "MAE": mean_absolute_error(y_test, y_pred),
                "R²": r2_score(y_test, y_pred),
            })

    st.dataframe(pd.DataFrame(results), use_container_width=True)
