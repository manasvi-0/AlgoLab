import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import io
from sklearn.datasets import make_classification, make_regression, load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    classification_report, confusion_matrix, roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Supervised Learning Visualizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .algorithm-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitSupervisedLearning:
    def __init__(self):
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.target_name = None
        
    def prepare_data(self, X, y, test_size=0.2, random_state=42,
                    feature_names=None, target_name=None):
        """Prepare data with train-test split and scaling"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(X.shape[1])]
        self.target_name = target_name or "Target"
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def visualize_train_test_split(self):
        """Visualize train-test split"""
        if self.X_train is None:
            return
            
        # Create subplot for train-test visualization
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Training Data', 'Test Data'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Training data
        fig.add_trace(
            go.Scatter(
                x=self.X_train[:, 0],
                y=self.y_train,
                mode='markers',
                name='Training',
                marker=dict(color='blue', size=8)
            ),
            row=1, col=1
        )
        
        # Test data
        fig.add_trace(
            go.Scatter(
                x=self.X_test[:, 0],
                y=self.y_test,
                mode='markers',
                name='Test',
                marker=dict(color='red', size=8)
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Train-Test Split Visualization",
            height=400,
            showlegend=True
        )
        
        return fig
    
    def linear_regression_realtime(self, progress_bar, status_text):
        """Real-time Linear Regression with visualization"""
        model = LinearRegression()
        
        # Simulate training with mini-batches
        n_samples = len(self.X_train)
        batch_size = max(1, n_samples // 20)
        
        # Initialize empty lists for tracking
        mse_history = []
        r2_history = []
        predictions_history = []
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = self.X_train[i:end_idx]
            y_batch = self.y_train[i:end_idx]
            
            # Fit model on batch
            model.fit(X_batch, y_batch)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            predictions_history.append(y_pred)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            mse_history.append(mse)
            r2_history.append(r2)
            
            # Update progress
            progress = (i + batch_size) / n_samples
            progress_bar.progress(progress)
            status_text.text(f"Training... {progress:.1%} complete")
            
            # Create real-time visualization
            fig = self._create_linear_regression_plots(
                model, y_pred, mse_history, r2_history
            )
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Small delay for visualization
            time.sleep(0.1)
        
        # Final model
        model.fit(self.X_train, self.y_train)
        final_pred = model.predict(self.X_test)
        final_mse = mean_squared_error(self.y_test, final_pred)
        final_r2 = r2_score(self.y_test, final_pred)
        
        # Display final metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final MSE", f"{final_mse:.4f}")
        with col2:
            st.metric("Final R²", f"{final_r2:.4f}")
        with col3:
            st.metric("Training Samples", len(self.X_train))
        
        return model, final_pred, final_mse, final_r2
    
    def _create_linear_regression_plots(self, model, y_pred, mse_history, r2_history):
        """Create comprehensive Linear Regression plots"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Regression Line vs Actual',
                'Predictions vs Actual',
                'MSE Progression',
                'R² Progression'
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Regression line plot
        x_range = np.linspace(self.X_test[:, 0].min(), self.X_test[:, 0].max(), 100)
        y_line = model.coef_[0] * x_range + model.intercept_
        
        fig.add_trace(
            go.Scatter(
                x=self.X_test[:, 0],
                y=self.y_test,
                mode='markers',
                name='Actual',
                marker=dict(color='blue', size=8)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_line,
                mode='lines',
                name='Regression Line',
                line=dict(color='red', width=3)
            ),
            row=1, col=1
        )
        
        # Predictions vs Actual
        fig.add_trace(
            go.Scatter(
                x=self.y_test,
                y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(color='green', size=8)
            ),
            row=1, col=2
        )
        
        # Perfect prediction line
        min_val = min(self.y_test.min(), y_pred.min())
        max_val = max(self.y_test.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=2
        )
        
        # MSE Progression
        if mse_history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(mse_history))),
                    y=mse_history,
                    mode='lines+markers',
                    name='MSE',
                    line=dict(color='orange', width=2)
                ),
                row=2, col=1
            )
        
        # R² Progression
        if r2_history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(r2_history))),
                    y=r2_history,
                    mode='lines+markers',
                    name='R²',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Linear Regression Real-time Training",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def logistic_regression_realtime(self, progress_bar, status_text):
        """Real-time Logistic Regression with visualization"""
        model = LogisticRegression(max_iter=1000, random_state=42)
        
        # Simulate training with mini-batches
        n_samples = len(self.X_train)
        batch_size = max(1, n_samples // 20)
        
        # Initialize empty lists for tracking
        accuracy_history = []
        predictions_history = []
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = self.X_train[i:end_idx]
            y_batch = self.y_train[i:end_idx]
            
            # Fit model on batch
            model.fit(X_batch, y_batch)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]
            predictions_history.append(y_pred)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            accuracy_history.append(accuracy)
            
            # Update progress
            progress = (i + batch_size) / n_samples
            progress_bar.progress(progress)
            status_text.text(f"Training... {progress:.1%} complete")
            
            # Create real-time visualization
            fig = self._create_logistic_regression_plots(
                model, y_pred, y_proba, accuracy_history
            )
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Small delay for visualization
            time.sleep(0.1)
        
        # Final model
        model.fit(self.X_train, self.y_train)
        final_pred = model.predict(self.X_test)
        final_accuracy = accuracy_score(self.y_test, final_pred)
        
        # Display final metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Accuracy", f"{final_accuracy:.4f}")
        with col2:
            st.metric("Training Samples", len(self.X_train))
        with col3:
            st.metric("Test Samples", len(self.X_test))
        
        return model, final_pred, final_accuracy
    
    def _create_logistic_regression_plots(self, model, y_pred, y_proba, accuracy_history):
        """Create comprehensive Logistic Regression plots"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Decision Boundary',
                'Predictions vs Actual',
                'Accuracy Progression',
                'Probability Distribution'
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
        )
        
        # Decision boundary plot
        x_min, x_max = self.X_test[:, 0].min() - 0.5, self.X_test[:, 0].max() + 0.5
        y_min, y_max = self.X_test[:, 1].min() - 0.5, self.X_test[:, 1].max() + 0.5
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        fig.add_trace(
            go.Contour(
                x=xx[0, :],
                y=yy[:, 0],
                z=Z,
                colorscale='RdBu',
                opacity=0.3,
                showscale=False
            ),
            row=1, col=1
        )
        
        # Scatter plot of test data
        for i, label in enumerate(np.unique(self.y_test)):
            mask = self.y_test == label
            fig.add_trace(
                go.Scatter(
                    x=self.X_test[mask, 0],
                    y=self.X_test[mask, 1],
                    mode='markers',
                    name=f'Class {label}',
                    marker=dict(size=8)
                ),
                row=1, col=1
            )
        
        # Predictions vs Actual
        fig.add_trace(
            go.Scatter(
                x=self.y_test,
                y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(color='green', size=8)
            ),
            row=1, col=2
        )
        
        # Accuracy Progression
        if accuracy_history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(accuracy_history))),
                    y=accuracy_history,
                    mode='lines+markers',
                    name='Accuracy',
                    line=dict(color='blue', width=2)
                ),
                row=2, col=1
            )
        
        # Probability Distribution
        fig.add_trace(
            go.Histogram(
                x=y_proba,
                nbinsx=20,
                name='Probability Distribution',
                marker=dict(color='purple', opacity=0.7)
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Logistic Regression Real-time Training",
            height=600,
            showlegend=True
        )
        
        return fig

    def decision_tree_realtime(self, progress_bar, status_text, max_depth=5, task='classification'):
        """Real-time Decision Tree training with visualization"""
        if self.X_train is None:
            return None, None, None
        
        # Initialize model based on task
        if task == 'classification':
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        else:
            model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        
        # Training steps for visualization
        n_steps = 20
        step_size = len(self.X_train) // n_steps
        
        # Initialize containers for plotting
        placeholder = st.empty()
        
        # Training history
        if task == 'classification':
            accuracy_history = []
        else:
            mse_history = []
            r2_history = []
        
        # Simulate step-by-step training
        for step in range(n_steps):
            # Update progress
            progress = (step + 1) / n_steps
            progress_bar.progress(progress)
            status_text.text(f"Training Decision Tree... Step {step + 1}/{n_steps}")
            
            # Use subset of data for this step
            end_idx = min((step + 1) * step_size, len(self.X_train))
            X_subset = self.X_train[:end_idx]
            y_subset = self.y_train[:end_idx]
            
            # Train model
            model.fit(X_subset, y_subset)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            if task == 'classification':
                accuracy = accuracy_score(self.y_test, y_pred)
                accuracy_history.append(accuracy)
            else:
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                mse_history.append(mse)
                r2_history.append(r2)
            
            # Create visualization
            if task == 'classification':
                fig = self._create_decision_tree_classification_plots(model, y_pred, accuracy_history)
            else:
                fig = self._create_decision_tree_regression_plots(model, y_pred, mse_history, r2_history)
            
            # Update plot
            with placeholder.container():
                st.plotly_chart(fig, use_container_width=True)
            
            time.sleep(0.1)  # Small delay for visualization
        
        # Final metrics
        if task == 'classification':
            final_accuracy = accuracy_history[-1] if accuracy_history else 0
            return model, y_pred, final_accuracy
        else:
            final_mse = mse_history[-1] if mse_history else 0
            final_r2 = r2_history[-1] if r2_history else 0
            return model, y_pred, final_mse, final_r2

    def _create_decision_tree_classification_plots(self, model, y_pred, accuracy_history):
        """Create plots for Decision Tree classification"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Decision Boundary', 'Predictions vs Actual', 'Accuracy Progression', 'Feature Importance'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Decision Boundary (for 2D data)
        if self.X_test.shape[1] >= 2:
            x_min, x_max = self.X_test[:, 0].min() - 0.5, self.X_test[:, 0].max() + 0.5
            y_min, y_max = self.X_test[:, 1].min() - 0.5, self.X_test[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                np.arange(y_min, y_max, 0.1))
            
            # Create mesh points
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            if self.X_test.shape[1] > 2:
                # Pad with zeros for higher dimensions
                padding = np.zeros((mesh_points.shape[0], self.X_test.shape[1] - 2))
                mesh_points = np.hstack([mesh_points, padding])
            
            # Predict on mesh
            mesh_pred = model.predict(mesh_points)
            mesh_pred = mesh_pred.reshape(xx.shape)
            
            # Decision boundary
            fig.add_trace(
                go.Contour(
                    x=xx[0, :],
                    y=yy[:, 0],
                    z=mesh_pred,
                    colorscale='RdYlBu',
                    showscale=False,
                    opacity=0.3
                ),
                row=1, col=1
            )
        
        # Scatter plot of test data
        fig.add_trace(
            go.Scatter(
                x=self.X_test[:, 0],
                y=self.X_test[:, 1] if self.X_test.shape[1] > 1 else self.y_test,
                mode='markers',
                marker=dict(
                    color=self.y_test,
                    colorscale='viridis',
                    size=8
                ),
                name='Test Data',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Predictions vs Actual
        fig.add_trace(
            go.Scatter(
                x=self.y_test,
                y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(color='green', size=8)
            ),
            row=1, col=2
        )
        
        # Accuracy Progression
        if accuracy_history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(accuracy_history))),
                    y=accuracy_history,
                    mode='lines+markers',
                    name='Accuracy',
                    line=dict(color='blue', width=2)
                ),
                row=2, col=1
            )
        
        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            fig.add_trace(
                go.Bar(
                    x=[f"Feature {i}" for i in range(len(model.feature_importances_))],
                    y=model.feature_importances_,
                    name='Feature Importance',
                    marker=dict(color='orange')
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Decision Tree Classification Real-time Training",
            height=600,
            showlegend=True
        )
        
        return fig

    def _create_decision_tree_regression_plots(self, model, y_pred, mse_history, r2_history):
        """Create plots for Decision Tree regression"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Predictions vs Actual', 'Residuals', 'MSE Progression', 'Feature Importance'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Predictions vs Actual
        fig.add_trace(
            go.Scatter(
                x=self.y_test,
                y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(color='green', size=8)
            ),
            row=1, col=1
        )
        
        # Add perfect prediction line
        min_val = min(self.y_test.min(), y_pred.min())
        max_val = max(self.y_test.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        # Residuals
        residuals = self.y_test - y_pred
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(color='purple', size=8)
            ),
            row=1, col=2
        )
        
        # Add zero line for residuals
        fig.add_trace(
            go.Scatter(
                x=[y_pred.min(), y_pred.max()],
                y=[0, 0],
                mode='lines',
                name='Zero Line',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # MSE Progression
        if mse_history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(mse_history))),
                    y=mse_history,
                    mode='lines+markers',
                    name='MSE',
                    line=dict(color='blue', width=2)
                ),
                row=2, col=1
            )
        
        # R² Progression
        if r2_history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(r2_history))),
                    y=r2_history,
                    mode='lines+markers',
                    name='R²',
                    line=dict(color='green', width=2),
                    yaxis='y2'
                ),
                row=2, col=1
            )
        
        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            fig.add_trace(
                go.Bar(
                    x=[f"Feature {i}" for i in range(len(model.feature_importances_))],
                    y=model.feature_importances_,
                    name='Feature Importance',
                    marker=dict(color='orange')
                ),
                row=2, col=2
            )
        
        # Update layout for dual y-axis
        fig.update_layout(
            title="Decision Tree Regression Real-time Training",
            height=600,
            showlegend=True,
            yaxis2=dict(
                title="R²",
                overlaying="y",
                side="right"
            )
        )
        
        return fig
    
    def svm_realtime(self, progress_bar, status_text, task='classification'):
        """Real-time SVM with visualization"""
        if task == 'classification':
            model = SVC(kernel='rbf', probability=True, random_state=42)
        else:
            model = SVR(kernel='rbf')
        
        # Simulate training with mini-batches
        n_samples = len(self.X_train)
        batch_size = max(1, n_samples // 20)
        
        # Initialize empty lists for tracking
        if task == 'classification':
            accuracy_history = []
        else:
            mse_history = []
            r2_history = []
        
        # Training loop
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = self.X_train[i:end_idx]
            y_batch = self.y_train[i:end_idx]
            
            # Update progress
            progress = (i + batch_size) / n_samples
            progress_bar.progress(progress)
            status_text.text(f"Training SVM... {progress:.1%}")
            
            # Fit model on batch
            model.fit(X_batch, y_batch)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            if task == 'classification':
                accuracy = accuracy_score(self.y_test, y_pred)
                accuracy_history.append(accuracy)
            else:
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                mse_history.append(mse)
                r2_history.append(r2)
            
            # Update plots
            if task == 'classification':
                fig = self._create_svm_classification_plots(model, y_pred, accuracy_history)
            else:
                fig = self._create_svm_regression_plots(model, y_pred, mse_history, r2_history)
            
            st.plotly_chart(fig, use_container_width=True)
            time.sleep(0.1)
        
        # Final predictions
        final_pred = model.predict(self.X_test)
        
        if task == 'classification':
            final_accuracy = accuracy_score(self.y_test, final_pred)
            return model, final_pred, final_accuracy
        else:
            final_mse = mean_squared_error(self.y_test, final_pred)
            final_r2 = r2_score(self.y_test, final_pred)
            return model, final_pred, final_mse, final_r2
    
    def _create_svm_classification_plots(self, model, y_pred, accuracy_history):
        """Create plots for SVM classification"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Decision Boundary", "Actual vs Predicted",
                "Accuracy Progression", "Support Vectors"
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Decision boundary visualization (2D only)
        if self.X_test.shape[1] == 2:
            x_min, x_max = self.X_test[:, 0].min() - 0.1, self.X_test[:, 0].max() + 0.1
            y_min, y_max = self.X_test[:, 1].min() - 0.1, self.X_test[:, 1].max() + 0.1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                                np.linspace(y_min, y_max, 50))
            
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Decision boundary
            fig.add_trace(
                go.Contour(
                    x=np.linspace(x_min, x_max, 50),
                    y=np.linspace(y_min, y_max, 50),
                    z=Z,
                    colorscale='RdBu',
                    showscale=False,
                    opacity=0.3
                ),
                row=1, col=1
            )
        
        # Actual vs Predicted
        for class_idx in range(len(np.unique(self.y_test))):
            mask_actual = self.y_test == class_idx
            mask_pred = y_pred == class_idx
            
            # Actual points
            fig.add_trace(
                go.Scatter(
                    x=self.X_test[mask_actual, 0],
                    y=self.X_test[mask_actual, 1],
                    mode='markers',
                    name=f'Class {class_idx} Actual',
                    marker=dict(color=['blue', 'red', 'green', 'orange'][class_idx], size=8)
                ),
                row=1, col=2
            )
            
            # Predicted points
            fig.add_trace(
                go.Scatter(
                    x=self.X_test[mask_pred, 0],
                    y=self.X_test[mask_pred, 1],
                    mode='markers',
                    name=f'Class {class_idx} Predicted',
                    marker=dict(color=['lightblue', 'pink', 'lightgreen', 'yellow'][class_idx], 
                               size=6, symbol='x')
                ),
                row=1, col=2
            )
        
        # Accuracy Progression
        if accuracy_history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(accuracy_history))),
                    y=accuracy_history,
                    mode='lines+markers',
                    name='Accuracy',
                    line=dict(color='green', width=2)
                ),
                row=2, col=1
            )
        
        # Support Vectors (if available)
        if hasattr(model, 'support_vectors_'):
            fig.add_trace(
                go.Scatter(
                    x=model.support_vectors_[:, 0],
                    y=model.support_vectors_[:, 1],
                    mode='markers',
                    name='Support Vectors',
                    marker=dict(color='black', size=10, symbol='diamond')
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="SVM Classification Real-time Training",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def _create_svm_regression_plots(self, model, y_pred, mse_history, r2_history):
        """Create plots for SVM regression"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Actual vs Predicted", "Residuals",
                "MSE Progression", "R² Progression"
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Actual vs Predicted
        fig.add_trace(
            go.Scatter(
                x=self.y_test,
                y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(color='blue', size=8)
            ),
            row=1, col=1
        )
        
        # Perfect prediction line
        min_val = min(self.y_test.min(), y_pred.min())
        max_val = max(self.y_test.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # Residuals
        residuals = self.y_test - y_pred
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(color='green', size=8)
            ),
            row=1, col=2
        )
        
        # Zero line for residuals
        fig.add_trace(
            go.Scatter(
                x=[y_pred.min(), y_pred.max()],
                y=[0, 0],
                mode='lines',
                name='Zero Line',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=2
        )
        
        # MSE Progression
        if mse_history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(mse_history))),
                    y=mse_history,
                    mode='lines+markers',
                    name='MSE',
                    line=dict(color='orange', width=2)
                ),
                row=2, col=1
            )
        
        # R² Progression
        if r2_history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(r2_history))),
                    y=r2_history,
                    mode='lines+markers',
                    name='R²',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="SVM Regression Real-time Training",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def knn_realtime(self, progress_bar, status_text, task='classification', n_neighbors=5):
        """Real-time KNN with visualization"""
        if task == 'classification':
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
        else:
            model = KNeighborsRegressor(n_neighbors=n_neighbors)
        
        # Simulate training with mini-batches
        n_samples = len(self.X_train)
        batch_size = max(1, n_samples // 20)
        
        # Initialize empty lists for tracking
        if task == 'classification':
            accuracy_history = []
        else:
            mse_history = []
            r2_history = []
        
        # Training loop
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = self.X_train[i:end_idx]
            y_batch = self.y_train[i:end_idx]
            
            # Update progress
            progress = (i + batch_size) / n_samples
            progress_bar.progress(progress)
            status_text.text(f"Training KNN... {progress:.1%}")
            
            # Fit model on batch
            model.fit(X_batch, y_batch)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            if task == 'classification':
                accuracy = accuracy_score(self.y_test, y_pred)
                accuracy_history.append(accuracy)
            else:
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                mse_history.append(mse)
                r2_history.append(r2)
            
            # Update plots
            if task == 'classification':
                fig = self._create_knn_classification_plots(model, y_pred, accuracy_history)
            else:
                fig = self._create_knn_regression_plots(model, y_pred, mse_history, r2_history)
            
            st.plotly_chart(fig, use_container_width=True)
            time.sleep(0.1)
        
        # Final predictions
        final_pred = model.predict(self.X_test)
        
        if task == 'classification':
            final_accuracy = accuracy_score(self.y_test, final_pred)
            return model, final_pred, final_accuracy
        else:
            final_mse = mean_squared_error(self.y_test, final_pred)
            final_r2 = r2_score(self.y_test, final_pred)
            return model, final_pred, final_mse, final_r2
    
    def _create_knn_classification_plots(self, model, y_pred, accuracy_history):
        """Create plots for KNN classification"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Decision Boundary", "Actual vs Predicted",
                "Accuracy Progression", "Nearest Neighbors"
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Decision boundary visualization (2D only)
        if self.X_test.shape[1] == 2:
            x_min, x_max = self.X_test[:, 0].min() - 0.1, self.X_test[:, 0].max() + 0.1
            y_min, y_max = self.X_test[:, 1].min() - 0.1, self.X_test[:, 1].max() + 0.1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                                np.linspace(y_min, y_max, 50))
            
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Decision boundary
            fig.add_trace(
                go.Contour(
                    x=np.linspace(x_min, x_max, 50),
                    y=np.linspace(y_min, y_max, 50),
                    z=Z,
                    colorscale='RdBu',
                    showscale=False,
                    opacity=0.3
                ),
                row=1, col=1
            )
        
        # Actual vs Predicted
        for class_idx in range(len(np.unique(self.y_test))):
            mask_actual = self.y_test == class_idx
            mask_pred = y_pred == class_idx
            
            # Actual points
            fig.add_trace(
                go.Scatter(
                    x=self.X_test[mask_actual, 0],
                    y=self.X_test[mask_actual, 1],
                    mode='markers',
                    name=f'Class {class_idx} Actual',
                    marker=dict(color=['blue', 'red', 'green', 'orange'][class_idx], size=8)
                ),
                row=1, col=2
            )
            
            # Predicted points
            fig.add_trace(
                go.Scatter(
                    x=self.X_test[mask_pred, 0],
                    y=self.X_test[mask_pred, 1],
                    mode='markers',
                    name=f'Class {class_idx} Predicted',
                    marker=dict(color=['lightblue', 'pink', 'lightgreen', 'yellow'][class_idx], 
                               size=6, symbol='x')
                ),
                row=1, col=2
            )
        
        # Accuracy Progression
        if accuracy_history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(accuracy_history))),
                    y=accuracy_history,
                    mode='lines+markers',
                    name='Accuracy',
                    line=dict(color='green', width=2)
                ),
                row=2, col=1
            )
        
        # Training points (nearest neighbors)
        fig.add_trace(
            go.Scatter(
                x=self.X_train[:, 0],
                y=self.X_train[:, 1],
                mode='markers',
                name='Training Points',
                marker=dict(color='gray', size=4, opacity=0.6)
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="KNN Classification Real-time Training",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def _create_knn_regression_plots(self, model, y_pred, mse_history, r2_history):
        """Create plots for KNN regression"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Actual vs Predicted", "Residuals",
                "MSE Progression", "R² Progression"
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Actual vs Predicted
        fig.add_trace(
            go.Scatter(
                x=self.y_test,
                y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(color='blue', size=8)
            ),
            row=1, col=1
        )
        
        # Perfect prediction line
        min_val = min(self.y_test.min(), y_pred.min())
        max_val = max(self.y_test.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # Residuals
        residuals = self.y_test - y_pred
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(color='green', size=8)
            ),
            row=1, col=2
        )
        
        # Zero line for residuals
        fig.add_trace(
            go.Scatter(
                x=[y_pred.min(), y_pred.max()],
                y=[0, 0],
                mode='lines',
                name='Zero Line',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=2
        )
        
        # MSE Progression
        if mse_history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(mse_history))),
                    y=mse_history,
                    mode='lines+markers',
                    name='MSE',
                    line=dict(color='orange', width=2)
                ),
                row=2, col=1
            )
        
        # R² Progression
        if r2_history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(r2_history))),
                    y=r2_history,
                    mode='lines+markers',
                    name='R²',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="KNN Regression Real-time Training",
            height=600,
            showlegend=True
        )
        
        return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Supervised Learning Visualizer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📋 Configuration")
    
    # Dataset selection
    dataset_type = st.sidebar.selectbox(
        "Select Dataset Type",
        ["Synthetic Classification", "Synthetic Regression", "Real Dataset", "Upload Custom"]
    )
    
    # Algorithm selection
    algorithm = st.sidebar.selectbox(
        "Select Algorithm",
        ["Linear Regression", "Logistic Regression", "Decision Tree", "SVM", "KNN"]
    )
    
    # Initialize the learning module
    slm = StreamlitSupervisedLearning()
    
    # Initialize real_dataset variable
    real_dataset = None
    
    # Generate or load data based on selection
    if dataset_type == "Synthetic Classification":
        n_samples = st.sidebar.slider("Number of samples", 100, 1000, 500)
        n_features = st.sidebar.slider("Number of features", 2, 10, 2)
        n_classes = st.sidebar.slider("Number of classes", 2, 4, 2)
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_clusters_per_class=1,
            n_redundant=0,
            random_state=42
        )
        
    elif dataset_type == "Synthetic Regression":
        n_samples = st.sidebar.slider("Number of samples", 100, 1000, 500)
        n_features = st.sidebar.slider("Number of features", 1, 10, 1)
        
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=0.1,
            random_state=42
        )
        
    elif dataset_type == "Real Dataset":
        real_dataset = st.sidebar.selectbox(
            "Select Real Dataset",
            ["Iris (Classification)", "California Housing (Regression)"]
        )
        
        if real_dataset == "Iris (Classification)":
            data = load_iris()
            X, y = data.data, data.target
        else:  # California Housing
            data = fetch_california_housing()
            X, y = data.data, data.target
            
    else:  # Upload Custom
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file",
            type=['csv']
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.write("Dataset Preview:")
            st.sidebar.dataframe(df.head())
            
            target_col = st.sidebar.selectbox("Select target column", df.columns)
            feature_cols = [col for col in df.columns if col != target_col]
            
            X = df[feature_cols].values
            y = df[target_col].values
        else:
            st.warning("Please upload a CSV file to continue.")
            return
    
    # Prepare data
    if 'X' in locals() and 'y' in locals():
        X_train, X_test, y_train, y_test = slm.prepare_data(X, y)
        
        # Display dataset info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(X))
        with col2:
            st.metric("Features", X.shape[1])
        with col3:
            st.metric("Training Samples", len(X_train))
        with col4:
            st.metric("Test Samples", len(X_test))
        
        # Show train-test split visualization
        st.subheader("📊 Train-Test Split Visualization")
        split_fig = slm.visualize_train_test_split()
        st.plotly_chart(split_fig, use_container_width=True)
        
        # Run selected algorithm
        st.subheader(f"🎯 {algorithm} Training")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        if algorithm == "Linear Regression":
            # Linear Regression works for regression datasets
            if dataset_type == "Synthetic Regression" or (dataset_type == "Real Dataset" and real_dataset and real_dataset == "California Housing (Regression)"):
                model, predictions, mse, r2 = slm.linear_regression_realtime(progress_bar, status_text)
            else:
                st.warning("Linear Regression requires a regression dataset. Please select 'Synthetic Regression' or 'California Housing (Regression)'.")
                return
            
        elif algorithm == "Logistic Regression":
            # Logistic Regression works for classification datasets
            if dataset_type == "Synthetic Classification" or (dataset_type == "Real Dataset" and real_dataset and real_dataset == "Iris (Classification)"):
                model, predictions, accuracy = slm.logistic_regression_realtime(progress_bar, status_text)
            else:
                st.warning("Logistic Regression requires a classification dataset. Please select 'Synthetic Classification' or 'Iris (Classification)'.")
                return
            
        elif algorithm == "Decision Tree":
            # Add max_depth parameter for Decision Tree
            max_depth = st.sidebar.slider("Max Depth", 1, 10, 5)
            
            # Determine task type based on dataset
            if dataset_type == "Synthetic Classification" or (dataset_type == "Real Dataset" and real_dataset and real_dataset == "Iris (Classification)"):
                task = 'classification'
                model, predictions, accuracy = slm.decision_tree_realtime(progress_bar, status_text, max_depth, task)
            else:
                task = 'regression'
                model, predictions, mse, r2 = slm.decision_tree_realtime(progress_bar, status_text, max_depth, task)
            
        elif algorithm == "SVM":
            # Determine task type based on dataset
            if dataset_type == "Synthetic Classification" or (dataset_type == "Real Dataset" and real_dataset and real_dataset == "Iris (Classification)"):
                task = 'classification'
                model, predictions, accuracy = slm.svm_realtime(progress_bar, status_text, task)
            else:
                task = 'regression'
                model, predictions, mse, r2 = slm.svm_realtime(progress_bar, status_text, task)
            
        elif algorithm == "KNN":
            # Add n_neighbors parameter for KNN
            n_neighbors = st.sidebar.slider("Number of Neighbors", 1, 20, 5)
            
            # Determine task type based on dataset
            if dataset_type == "Synthetic Classification" or (dataset_type == "Real Dataset" and real_dataset and real_dataset == "Iris (Classification)"):
                task = 'classification'
                model, predictions, accuracy = slm.knn_realtime(progress_bar, status_text, task, n_neighbors)
            else:
                task = 'regression'
                model, predictions, mse, r2 = slm.knn_realtime(progress_bar, status_text, task, n_neighbors)
            
        else:
            st.warning(f"{algorithm} is not yet implemented for this dataset type.")
            return
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.success("✅ Training completed!")

if __name__ == "__main__":
    main()
