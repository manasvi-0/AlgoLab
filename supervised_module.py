'''
Contains codes for supervised ML Algorithms:-
    Preprocessing (handling nulls, outliers, categorical encoding)
    Interactive hyperparameter tuning
    Multi-algorithm performance comparison
    Decision boundary visualization (2D datasets)
'''

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

#Data Preprocessing
def preprocess_data(df: pd.DataFrame):
    df = df.copy()

    # Handle nulls
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Encode categorical features
    for col in df.columns:
        if df[col].dtype == 'object':
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])

    # Outlier treatment (IQR method)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    return df

#Decision Boundary

def plot_decision_boundary(model, X, y, ax, title):
    """Plots decision boundaries for 2D datasets."""
    h = 0.02  # step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    ax.set_title(title)
    return scatter

#Model Tuning
def interactive_model_tuning(df):
    """Main function for supervised learning playground."""
    
    if df is None:
        st.warning("Please upload or generate a dataset first.")
        return

    st.subheader(" Preprocessing & Train-Test Split")
    df = preprocess_data(df)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    test_size = st.slider("Test Size (%)", 10, 50, 30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42
    )

    st.subheader("🤖 Select Algorithms to Compare")
    algorithms = st.multiselect(
        "Choose models:", 
        ["KNN", "Decision Tree", "Logistic Regression", "SVM"]
    )

    models = {}
    scores = []

    # 🔹 KNN
    if "KNN" in algorithms:
        from supervised_algos import knn_visualization
        knn_visualization.render(df)
        return

    # 🔹 Decision Tree
    if "Decision Tree" in algorithms:
        max_depth = st.slider("Decision Tree - Max Depth", 1, 20, 5)
        min_samples_split = st.slider("Decision Tree - Min Samples Split", 2, 10, 2)
        models["Decision Tree"] = DecisionTreeClassifier(max_depth=max_depth, 
                                                         min_samples_split=min_samples_split)

    # 🔹 Logistic Regression
    if "Logistic Regression" in algorithms:
        C = st.slider("Logistic Regression - Regularization Strength (C)", 0.01, 10.0, 1.0)
        models["Logistic Regression"] = LogisticRegression(C=C, max_iter=1000)

    # 🔹 SVM
    if "SVM" in algorithms:
        kernel = st.selectbox("SVM - Kernel", ["linear", "poly", "rbf", "sigmoid"])
        C = st.slider("SVM - Regularization Parameter (C)", 0.01, 10.0, 1.0)
        models["SVM"] = SVC(kernel=kernel, C=C)
        
        
    #Training
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append({
            "Algorithm": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted'),
            "Recall": recall_score(y_test, y_pred, average='weighted'),
            "F1-Score": f1_score(y_test, y_pred, average='weighted')
        })

    if scores:
        st.subheader("Performance Comparison")
        score_df = pd.DataFrame(scores)
        st.dataframe(score_df)

        # Bar plot
        st.subheader("Metrics Visualization")
        fig, ax = plt.subplots(figsize=(8, 4))
        melted_df = score_df.melt(id_vars=["Algorithm"], 
                                  value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
                                  var_name="Metric", value_name="Score")
        sns.barplot(data=melted_df, x="Algorithm", y="Score", hue="Metric", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        
        if X.shape[1] == 2:  # Only for 2D
            st.subheader("Decision Boundary Visualization")
            fig, ax = plt.subplots()
            for name, model in models.items():
                plot_decision_boundary(model, X_train, y_train, ax, f"{name} Boundary")
            st.pyplot(fig)
        else:
            st.info("Decision boundary visualization is only available for 2D datasets.")
            
