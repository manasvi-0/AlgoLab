import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def visualize():
    """
    Main function for data visualization module
    Provides interactive data exploration and visualization tools
    """
    st.header("📊 Data Visualization & Exploration")
    st.markdown("Upload your dataset or use sample data to explore various visualization techniques.")
    
    # Sidebar for data source selection
    data_source = st.sidebar.selectbox(
        "Choose Data Source",
        ["Upload CSV", "Sample Dataset", "Generate Synthetic Data"]
    )
    
    df = None
    
    if data_source == "Upload CSV":
        df = upload_csv_section()
    elif data_source == "Sample Dataset":
        df = sample_dataset_section()
    elif data_source == "Generate Synthetic Data":
        df = generate_synthetic_data()
    
    if df is not None:
        # Display basic info
        display_data_info(df)
        
        # Visualization options
        st.subheader("🎨 Visualization Options")
        
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Data Overview", "Correlation Analysis", "Distribution Analysis", 
             "Scatter Plot", "Interactive Plots", "Statistical Summary"]
        )
        
        if viz_type == "Data Overview":
            data_overview(df)
        elif viz_type == "Correlation Analysis":
            correlation_analysis(df)
        elif viz_type == "Distribution Analysis":
            distribution_analysis(df)
        elif viz_type == "Scatter Plot":
            scatter_plot_analysis(df)
        elif viz_type == "Interactive Plots":
            interactive_plots(df)
        elif viz_type == "Statistical Summary":
            statistical_summary(df)

def upload_csv_section():
    """Handle CSV file upload"""
    st.subheader("📁 Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="data_viz_uploader")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns!")
            return df
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return None
    return None

def sample_dataset_section():
    """Provide sample datasets"""
    st.subheader("📚 Sample Datasets")
    
    dataset_choice = st.selectbox(
        "Choose a sample dataset",
        ["Iris Dataset", "Boston Housing", "Wine Quality", "Titanic Sample"]
    )
    
    if st.button("Load Sample Dataset", key="load_sample"):
        if dataset_choice == "Iris Dataset":
            from sklearn.datasets import load_iris
            iris = load_iris()
            df = pd.DataFrame(iris.data, columns=iris.feature_names)
            df['target'] = iris.target
            df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
            st.success("✅ Iris dataset loaded!")
            return df
            
        elif dataset_choice == "Boston Housing":
            # Create a synthetic housing dataset since Boston Housing was removed from sklearn
            np.random.seed(42)
            n_samples = 506
            df = pd.DataFrame({
                'CRIM': np.random.exponential(3, n_samples),
                'ZN': np.random.uniform(0, 100, n_samples),
                'INDUS': np.random.uniform(0, 25, n_samples),
                'NOX': np.random.uniform(0.3, 0.9, n_samples),
                'RM': np.random.normal(6.3, 0.7, n_samples),
                'AGE': np.random.uniform(0, 100, n_samples),
                'DIS': np.random.uniform(1, 12, n_samples),
                'TAX': np.random.uniform(200, 700, n_samples),
                'PTRATIO': np.random.uniform(12, 22, n_samples),
                'PRICE': np.random.uniform(10, 50, n_samples)
            })
            st.success("✅ Housing dataset loaded!")
            return df
            
        elif dataset_choice == "Wine Quality":
            # Generate synthetic wine quality data
            np.random.seed(42)
            n_samples = 1000
            df = pd.DataFrame({
                'fixed_acidity': np.random.normal(7, 1.5, n_samples),
                'volatile_acidity': np.random.normal(0.4, 0.2, n_samples),
                'citric_acid': np.random.normal(0.3, 0.15, n_samples),
                'residual_sugar': np.random.exponential(2, n_samples),
                'chlorides': np.random.normal(0.08, 0.03, n_samples),
                'free_sulfur_dioxide': np.random.normal(30, 15, n_samples),
                'total_sulfur_dioxide': np.random.normal(120, 40, n_samples),
                'density': np.random.normal(0.996, 0.003, n_samples),
                'pH': np.random.normal(3.2, 0.3, n_samples),
                'sulphates': np.random.normal(0.6, 0.2, n_samples),
                'alcohol': np.random.normal(10.5, 1.5, n_samples),
                'quality': np.random.randint(3, 9, n_samples)
            })
            st.success("✅ Wine Quality dataset loaded!")
            return df
            
        elif dataset_choice == "Titanic Sample":
            # Generate synthetic Titanic-like data
            np.random.seed(42)
            n_samples = 800
            df = pd.DataFrame({
                'age': np.random.normal(30, 15, n_samples),
                'fare': np.random.exponential(50, n_samples),
                'pclass': np.random.choice([1, 2, 3], n_samples, p=[0.3, 0.3, 0.4]),
                'sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
                'survived': np.random.choice([0, 1], n_samples, p=[0.62, 0.38])
            })
            st.success("✅ Titanic sample dataset loaded!")
            return df
    
    return None

def generate_synthetic_data():
    """Generate synthetic data based on user parameters"""
    st.subheader("🔧 Generate Synthetic Data")
    
    with st.expander("Data Generation Parameters"):
        n_samples = st.slider("Number of Samples", 100, 2000, 500, key="dataviz_samples")
        n_features = st.slider("Number of Features", 2, 10, 4, key="dataviz_features")
        noise_level = st.slider("Noise Level", 0.0, 1.0, 0.1, key="dataviz_noise")
        data_type = st.selectbox("Data Type", ["Classification", "Regression", "Clustering"])
    
    if st.button("Generate Data", key="dataviz_generate"):
        np.random.seed(42)
        
        if data_type == "Classification":
            from sklearn.datasets import make_classification
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=max(2, n_features-1),
                n_redundant=0,
                n_clusters_per_class=1,
                flip_y=noise_level,
                random_state=42
            )
            df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(n_features)])
            df['target'] = y
            
        elif data_type == "Regression":
            from sklearn.datasets import make_regression
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                noise=noise_level*100,
                random_state=42
            )
            df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(n_features)])
            df['target'] = y
            
        elif data_type == "Clustering":
            from sklearn.datasets import make_blobs
            X, y = make_blobs(
                n_samples=n_samples,
                centers=3,
                n_features=n_features,
                cluster_std=1.0 + noise_level,
                random_state=42
            )
            df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(n_features)])
            df['cluster'] = y
        
        st.success(f"✅ Generated {data_type.lower()} dataset with {n_samples} samples!")
        return df
    
    return None

def display_data_info(df):
    """Display basic information about the dataset"""
    st.subheader("📋 Dataset Information")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Numeric Columns", df.select_dtypes(include=[np.number]).shape[1])
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Display first few rows
    st.subheader("📊 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

def data_overview(df):
    """Provide comprehensive data overview"""
    st.subheader("🔍 Data Overview")
    
    # Data types
    st.write("**Data Types:**")
    st.dataframe(pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum()
    }), use_container_width=True)
    
    # Missing values heatmap
    if df.isnull().sum().sum() > 0:
        st.write("**Missing Values Heatmap:**")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=True, yticklabels=False, ax=ax)
        plt.title("Missing Values Pattern")
        st.pyplot(fig)

def correlation_analysis(df):
    """Analyze correlations between numeric variables"""
    st.subheader("🔗 Correlation Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        st.warning("⚠️ Need at least 2 numeric columns for correlation analysis.")
        return
    
    # Correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Interactive correlation heatmap
    fig = px.imshow(
        corr_matrix,
        color_continuous_scale='RdBu_r',
        aspect='auto',
        title="Correlation Matrix"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Strong correlations
    st.write("**Strong Correlations (|r| > 0.7):**")
    strong_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                strong_corr.append({
                    'Variable 1': corr_matrix.columns[i],
                    'Variable 2': corr_matrix.columns[j],
                    'Correlation': corr_val
                })
    
    if strong_corr:
        st.dataframe(pd.DataFrame(strong_corr), use_container_width=True)
    else:
        st.info("No strong correlations found (|r| > 0.7)")

def distribution_analysis(df):
    """Analyze distributions of variables"""
    st.subheader("📈 Distribution Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Numeric distributions
    if len(numeric_cols) > 0:
        st.write("**Numeric Variable Distributions:**")
        selected_numeric = st.multiselect(
            "Select numeric variables to plot",
            numeric_cols,
            default=list(numeric_cols)[:4]  # Default to first 4
        )
        
        if selected_numeric:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[f'Distribution of {col}' for col in selected_numeric[:4]]
            )
            
            for i, col in enumerate(selected_numeric[:4]):
                row = (i // 2) + 1
                col_pos = (i % 2) + 1
                
                fig.add_trace(
                    go.Histogram(x=df[col], name=col, showlegend=False),
                    row=row, col=col_pos
                )
            
            fig.update_layout(height=600, title_text="Variable Distributions")
            st.plotly_chart(fig, use_container_width=True)
    
    # Categorical distributions
    if len(categorical_cols) > 0:
        st.write("**Categorical Variable Distributions:**")
        selected_cat = st.selectbox("Select categorical variable", categorical_cols)
        
        if selected_cat:
            value_counts = df[selected_cat].value_counts()
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f'Distribution of {selected_cat}'
            )
            st.plotly_chart(fig, use_container_width=True)

def scatter_plot_analysis(df):
    """Create interactive scatter plots"""
    st.subheader("🎯 Scatter Plot Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        st.warning("⚠️ Need at least 2 numeric columns for scatter plot.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Select X-axis variable", numeric_cols)
    with col2:
        y_axis = st.selectbox("Select Y-axis variable", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
    
    # Optional color coding
    color_col = st.selectbox("Color by (optional)", ["None"] + list(df.columns))
    
    if color_col != "None":
        fig = px.scatter(
            df, x=x_axis, y=y_axis, color=color_col,
            title=f'{y_axis} vs {x_axis} (colored by {color_col})'
        )
    else:
        fig = px.scatter(
            df, x=x_axis, y=y_axis,
            title=f'{y_axis} vs {x_axis}'
        )
    
    st.plotly_chart(fig, use_container_width=True)

def interactive_plots(df):
    """Create various interactive plots"""
    st.subheader("🎨 Interactive Plots")
    
    plot_type = st.selectbox(
        "Select plot type",
        ["Box Plot", "Violin Plot", "Pair Plot", "3D Scatter"]
    )
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if plot_type == "Box Plot":
        if len(numeric_cols) > 0:
            selected_cols = st.multiselect("Select variables for box plot", numeric_cols)
            if selected_cols:
                fig = go.Figure()
                for col in selected_cols:
                    fig.add_trace(go.Box(y=df[col], name=col))
                fig.update_layout(title="Box Plot Comparison")
                st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Violin Plot":
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select variable for violin plot", numeric_cols)
            fig = go.Figure(data=go.Violin(y=df[selected_col], box_visible=True, meanline_visible=True))
            fig.update_layout(title=f"Violin Plot of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Pair Plot":
        if len(numeric_cols) >= 2:
            selected_cols = st.multiselect("Select variables for pair plot", numeric_cols, default=list(numeric_cols)[:3])
            if len(selected_cols) >= 2:
                fig = px.scatter_matrix(df[selected_cols])
                fig.update_layout(title="Pair Plot Matrix")
                st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "3D Scatter":
        if len(numeric_cols) >= 3:
            col1, col2, col3 = st.columns(3)
            with col1:
                x_var = st.selectbox("X variable", numeric_cols)
            with col2:
                y_var = st.selectbox("Y variable", numeric_cols, index=1)
            with col3:
                z_var = st.selectbox("Z variable", numeric_cols, index=2)
            
            fig = px.scatter_3d(df, x=x_var, y=y_var, z=z_var)
            fig.update_layout(title=f"3D Scatter: {x_var}, {y_var}, {z_var}")
            st.plotly_chart(fig, use_container_width=True)

def statistical_summary(df):
    """Provide detailed statistical summary"""
    st.subheader("📊 Statistical Summary")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        st.write("**Descriptive Statistics:**")
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        # Skewness and Kurtosis
        st.write("**Distribution Shape Metrics:**")
        shape_stats = pd.DataFrame({
            'Variable': numeric_cols,
            'Skewness': [df[col].skew() for col in numeric_cols],
            'Kurtosis': [df[col].kurtosis() for col in numeric_cols]
        })
        st.dataframe(shape_stats, use_container_width=True)
        
        # Outlier detection using IQR
        st.write("**Outlier Detection (IQR Method):**")
        outlier_summary = []
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            outlier_summary.append({
                'Variable': col,
                'Outlier Count': len(outliers),
                'Outlier Percentage': f"{len(outliers)/len(df)*100:.2f}%",
                'Lower Bound': lower_bound,
                'Upper Bound': upper_bound
            })
        
        st.dataframe(pd.DataFrame(outlier_summary), use_container_width=True)
    
    else:
        st.warning("⚠️ No numeric columns found for statistical analysis.")
