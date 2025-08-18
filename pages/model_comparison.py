import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# ------------------------
# Page Title
# ------------------------
st.title("🔍 Model Comparison Dashboard")
st.write("Compare multiple ML models on accuracy, precision, recall, and F1-score.")

# ------------------------
# Upload Dataset
# ------------------------
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data", df.head())

    target_col = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ------------------------
    # Model Selection
    # ------------------------
    st.sidebar.subheader("Select Models to Compare")
    models_selected = st.sidebar.multiselect(
        "Choose algorithms",
        ["KNN", "Decision Tree", "SVM", "Logistic Regression", "Random Forest"],
        default=["KNN", "Decision Tree"]
    )

    # Define models
    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "SVM": SVC(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier()
    }

    results = []

    for model_name in models_selected:
        model = models[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
        })

    results_df = pd.DataFrame(results)
    st.write("### 📊 Model Performance Comparison")
    st.dataframe(results_df, use_container_width=True)
