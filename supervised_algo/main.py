page = st.sidebar.selectbox("Choose Module", [
    "KNN Theory",
    "KNN Visualization",
    "Linear Regression Theory",
    "Linear Regression Visualization",
    "Logistic Regression Theory",
    "Logistic Regression Visualization"
])

if page == "Logistic Regression Theory":
    Logistic_theory.render()
elif page == "Logistic Regression Visualization":
    logistic_regression_visualization.render()
