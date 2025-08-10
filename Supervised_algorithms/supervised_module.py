"""
Supervised Learning Module with Real-time Visualizations
=======================================================

This module implements commonly used supervised machine learning algorithms
with both static and real-time visualizations for better understanding.

Algorithms included:
- Linear Regression
- Logistic Regression  
- Decision Tree
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

Features:
- Static visualizations (train-test split, decision boundaries, etc.)
- Real-time step-by-step training visualizations
- Modular and reusable code structure
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, 
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import StandardScaler
import matplotlib.animation as animation
try:
    from IPython.display import display, clear_output
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
    def display(*args, **kwargs):
        pass
    def clear_output(wait=False):
        pass
import time
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SupervisedLearningModule:
    """
    Main class for supervised learning algorithms with visualizations.
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.target_name = None
        
    def prepare_data(self, X, y, test_size=0.2, random_state=42, 
                    feature_names=None, target_name=None):
        """
        Prepare and split the dataset for training.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target variable
        test_size : float
            Proportion of dataset to include in test split
        random_state : int
            Random state for reproducibility
        feature_names : list
            Names of features for better visualization
        target_name : str
            Name of target variable
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) < 10 else None
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(X.shape[1])]
        self.target_name = target_name or 'Target'
        
        print(f"Dataset prepared:")
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        # Visualize train-test split
        self._visualize_train_test_split()
        
    def _visualize_train_test_split(self):
        """Visualize the train-test split of the data."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot training data
        if self.X_train.shape[1] == 1:
            axes[0].scatter(self.X_train, self.y_train, alpha=0.6, label='Training Data')
            axes[0].set_xlabel(self.feature_names[0])
            axes[0].set_ylabel(self.target_name)
        else:
            scatter = axes[0].scatter(self.X_train[:, 0], self.X_train[:, 1], 
                                    c=self.y_train, alpha=0.6, cmap='viridis')
            axes[0].set_xlabel(self.feature_names[0])
            axes[0].set_ylabel(self.feature_names[1])
            plt.colorbar(scatter, ax=axes[0])
        
        axes[0].set_title('Training Data')
        axes[0].legend()
        
        # Plot test data
        if self.X_test.shape[1] == 1:
            axes[1].scatter(self.X_test, self.y_test, alpha=0.6, label='Test Data', color='red')
            axes[1].set_xlabel(self.feature_names[0])
            axes[1].set_ylabel(self.target_name)
        else:
            scatter = axes[1].scatter(self.X_test[:, 0], self.X_test[:, 1], 
                                    c=self.y_test, alpha=0.6, cmap='viridis')
            axes[1].set_xlabel(self.feature_names[0])
            axes[1].set_ylabel(self.feature_names[1])
            plt.colorbar(scatter, ax=axes[1])
        
        axes[1].set_title('Test Data')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
        
    def linear_regression(self, real_time_viz=True):
        """
        Train Linear Regression model with real-time visualization.
        
        Parameters:
        -----------
        real_time_viz : bool
            Whether to show real-time training visualization
        """
        print("Training Linear Regression model...")
        
        if real_time_viz:
            self._linear_regression_realtime()
        else:
            self._linear_regression_static()
    
    def _linear_regression_static(self):
        """Train Linear Regression with static visualization."""
        model = LinearRegression()
        model.fit(self.X_train_scaled, self.y_train)
        
        # Make predictions
        y_train_pred = model.predict(self.X_train_scaled)
        y_test_pred = model.predict(self.X_test_scaled)
        
        # Calculate metrics
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        # Store model
        self.models['linear_regression'] = {
            'model': model,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        print(f"Training MSE: {train_mse:.4f}")
        print(f"Test MSE: {test_mse:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        
        # Visualize results
        self._visualize_linear_regression_results(y_train_pred, y_test_pred)
    
    def _linear_regression_realtime(self):
        """Train Linear Regression with real-time visualization."""
        # Initialize model
        model = LinearRegression()
        
        # Create figure for real-time visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Linear Regression - Real-time Training', fontsize=16)
        
        # Initialize plots
        axes[0, 0].scatter(self.X_train_scaled[:, 0], self.y_train, alpha=0.6, label='Training Data')
        axes[0, 0].set_xlabel(self.feature_names[0])
        axes[0, 0].set_ylabel(self.target_name)
        axes[0, 0].set_title('Training Progress')
        
        axes[0, 1].scatter(self.X_test_scaled[:, 0], self.y_test, alpha=0.6, label='Test Data', color='red')
        axes[0, 1].set_xlabel(self.feature_names[0])
        axes[0, 1].set_ylabel(self.target_name)
        axes[0, 1].set_title('Test Predictions')
        
        # Error tracking
        train_errors = []
        test_errors = []
        iterations = []
        
        # Simulate training with mini-batches for visualization
        batch_size = max(1, len(self.X_train_scaled) // 20)
        n_batches = len(self.X_train_scaled) // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(self.X_train_scaled))
            
            # Train on batch
            X_batch = self.X_train_scaled[start_idx:end_idx]
            y_batch = self.y_train[start_idx:end_idx]
            
            model.fit(X_batch, y_batch)
            
            # Make predictions
            y_train_pred = model.predict(self.X_train_scaled)
            y_test_pred = model.predict(self.X_test_scaled)
            
            # Calculate errors
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            
            train_errors.append(train_mse)
            test_errors.append(test_mse)
            iterations.append(i)
            
            # Update plots
            axes[0, 0].clear()
            axes[0, 0].scatter(self.X_train_scaled[:, 0], self.y_train, alpha=0.6, label='Training Data')
            
            # Plot regression line
            X_line = np.linspace(self.X_train_scaled[:, 0].min(), self.X_train_scaled[:, 0].max(), 100).reshape(-1, 1)
            if self.X_train_scaled.shape[1] > 1:
                X_line_full = np.zeros((100, self.X_train_scaled.shape[1]))
                X_line_full[:, 0] = X_line.flatten()
                y_line = model.predict(X_line_full)
            else:
                y_line = model.predict(X_line)
            
            axes[0, 0].plot(X_line, y_line, 'r-', linewidth=2, label=f'Regression Line (Iteration {i+1})')
            axes[0, 0].set_xlabel(self.feature_names[0])
            axes[0, 0].set_ylabel(self.target_name)
            axes[0, 0].set_title(f'Training Progress - Iteration {i+1}')
            axes[0, 0].legend()
            
            # Update test predictions
            axes[0, 1].clear()
            axes[0, 1].scatter(self.X_test_scaled[:, 0], self.y_test, alpha=0.6, label='Test Data', color='red')
            axes[0, 1].scatter(self.X_test_scaled[:, 0], y_test_pred, alpha=0.8, label='Predictions', color='green')
            axes[0, 1].set_xlabel(self.feature_names[0])
            axes[0, 1].set_ylabel(self.target_name)
            axes[0, 1].set_title('Test Predictions')
            axes[0, 1].legend()
            
            # Plot error progression
            axes[1, 0].clear()
            axes[1, 0].plot(iterations, train_errors, 'b-', label='Training MSE')
            axes[1, 0].plot(iterations, test_errors, 'r-', label='Test MSE')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Mean Squared Error')
            axes[1, 0].set_title('Error Progression')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Plot residuals
            axes[1, 1].clear()
            residuals = self.y_train - y_train_pred
            axes[1, 1].scatter(y_train_pred, residuals, alpha=0.6)
            axes[1, 1].axhline(y=0, color='r', linestyle='--')
            axes[1, 1].set_xlabel('Predicted Values')
            axes[1, 1].set_ylabel('Residuals')
            axes[1, 1].set_title('Residual Plot')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.pause(0.5)  # Pause to show progress
        
        # Final training on full dataset
        model.fit(self.X_train_scaled, self.y_train)
        y_train_pred = model.predict(self.X_train_scaled)
        y_test_pred = model.predict(self.X_test_scaled)
        
        # Calculate final metrics
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        # Store model
        self.models['linear_regression'] = {
            'model': model,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        print(f"\nFinal Results:")
        print(f"Training MSE: {train_mse:.4f}")
        print(f"Test MSE: {test_mse:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        
        plt.show()
    
    def _visualize_linear_regression_results(self, y_train_pred, y_test_pred):
        """Visualize Linear Regression results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Linear Regression Results', fontsize=16)
        
        # Training data with regression line
        axes[0, 0].scatter(self.X_train_scaled[:, 0], self.y_train, alpha=0.6, label='Training Data')
        
        # Plot regression line
        X_line = np.linspace(self.X_train_scaled[:, 0].min(), self.X_train_scaled[:, 0].max(), 100).reshape(-1, 1)
        if self.X_train_scaled.shape[1] > 1:
            X_line_full = np.zeros((100, self.X_train_scaled.shape[1]))
            X_line_full[:, 0] = X_line.flatten()
            y_line = self.models['linear_regression']['model'].predict(X_line_full)
        else:
            y_line = self.models['linear_regression']['model'].predict(X_line)
        
        axes[0, 0].plot(X_line, y_line, 'r-', linewidth=2, label='Regression Line')
        axes[0, 0].set_xlabel(self.feature_names[0])
        axes[0, 0].set_ylabel(self.target_name)
        axes[0, 0].set_title('Training Data with Regression Line')
        axes[0, 0].legend()
        
        # Test predictions
        axes[0, 1].scatter(self.X_test_scaled[:, 0], self.y_test, alpha=0.6, label='Actual', color='red')
        axes[0, 1].scatter(self.X_test_scaled[:, 0], y_test_pred, alpha=0.8, label='Predicted', color='green')
        axes[0, 1].set_xlabel(self.feature_names[0])
        axes[0, 1].set_ylabel(self.target_name)
        axes[0, 1].set_title('Test Predictions vs Actual')
        axes[0, 1].legend()
        
        # Residuals plot
        residuals_train = self.y_train - y_train_pred
        residuals_test = self.y_test - y_test_pred
        
        axes[1, 0].scatter(y_train_pred, residuals_train, alpha=0.6, label='Training')
        axes[1, 0].scatter(y_test_pred, residuals_test, alpha=0.6, label='Test', color='red')
        axes[1, 0].axhline(y=0, color='black', linestyle='--')
        axes[1, 0].set_xlabel('Predicted Values')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residual Plot')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Metrics comparison
        metrics = ['Training MSE', 'Test MSE', 'Training R²', 'Test R²']
        values = [
            self.models['linear_regression']['train_mse'],
            self.models['linear_regression']['test_mse'],
            self.models['linear_regression']['train_r2'],
            self.models['linear_regression']['test_r2']
        ]
        
        bars = axes[1, 1].bar(metrics, values, color=['blue', 'red', 'green', 'orange'])
        axes[1, 1].set_title('Model Performance Metrics')
        axes[1, 1].set_ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show() 
        
    def logistic_regression(self, real_time_viz=True):
        """
        Train Logistic Regression model with real-time visualization.
        
        Parameters:
        -----------
        real_time_viz : bool
            Whether to show real-time training visualization
        """
        print("Training Logistic Regression model...")
        
        if real_time_viz:
            self._logistic_regression_realtime()
        else:
            self._logistic_regression_static()
    
    def _logistic_regression_static(self):
        """Train Logistic Regression with static visualization."""
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(self.X_train_scaled, self.y_train)
        
        # Make predictions
        y_train_pred = model.predict(self.X_train_scaled)
        y_test_pred = model.predict(self.X_test_scaled)
        y_train_proba = model.predict_proba(self.X_train_scaled)[:, 1]
        y_test_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Calculate metrics
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        
        # Store model
        self.models['logistic_regression'] = {
            'model': model,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'y_train_proba': y_train_proba,
            'y_test_proba': y_test_proba
        }
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_test_pred))
        
        # Visualize results
        self._visualize_logistic_regression_results(y_train_pred, y_test_pred, y_train_proba, y_test_proba)
    
    def _logistic_regression_realtime(self):
        """Train Logistic Regression with real-time visualization."""
        # Initialize model
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        # Create figure for real-time visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Logistic Regression - Real-time Training', fontsize=16)
        
        # Initialize plots
        scatter = axes[0, 0].scatter(self.X_train_scaled[:, 0], self.X_train_scaled[:, 1], 
                                   c=self.y_train, alpha=0.6, cmap='viridis')
        axes[0, 0].set_xlabel(self.feature_names[0])
        axes[0, 0].set_ylabel(self.feature_names[1])
        axes[0, 0].set_title('Training Data')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        axes[0, 1].scatter(self.X_test_scaled[:, 0], self.X_test_scaled[:, 1], 
                          c=self.y_test, alpha=0.6, cmap='viridis')
        axes[0, 1].set_xlabel(self.feature_names[0])
        axes[0, 1].set_ylabel(self.feature_names[1])
        axes[0, 1].set_title('Test Data')
        
        # Error tracking
        train_accuracies = []
        test_accuracies = []
        iterations = []
        
        # Simulate training with mini-batches for visualization
        batch_size = max(1, len(self.X_train_scaled) // 20)
        n_batches = len(self.X_train_scaled) // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(self.X_train_scaled))
            
            # Train on batch
            X_batch = self.X_train_scaled[start_idx:end_idx]
            y_batch = self.y_train[start_idx:end_idx]
            
            model.fit(X_batch, y_batch)
            
            # Make predictions
            y_train_pred = model.predict(self.X_train_scaled)
            y_test_pred = model.predict(self.X_test_scaled)
            
            # Calculate accuracies
            train_accuracy = accuracy_score(self.y_train, y_train_pred)
            test_accuracy = accuracy_score(self.y_test, y_test_pred)
            
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            iterations.append(i)
            
            # Update decision boundary plot
            axes[0, 0].clear()
            scatter = axes[0, 0].scatter(self.X_train_scaled[:, 0], self.X_train_scaled[:, 1], 
                                       c=self.y_train, alpha=0.6, cmap='viridis')
            
            # Plot decision boundary
            x_min, x_max = self.X_train_scaled[:, 0].min() - 0.5, self.X_train_scaled[:, 0].max() + 0.5
            y_min, y_max = self.X_train_scaled[:, 1].min() - 0.5, self.X_train_scaled[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                np.arange(y_min, y_max, 0.02))
            
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            axes[0, 0].contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
            axes[0, 0].scatter(self.X_train_scaled[:, 0], self.X_train_scaled[:, 1], 
                              c=self.y_train, alpha=0.8, cmap='viridis', edgecolors='black')
            axes[0, 0].set_xlabel(self.feature_names[0])
            axes[0, 0].set_ylabel(self.feature_names[1])
            axes[0, 0].set_title(f'Decision Boundary - Iteration {i+1}')
            
            # Update test predictions
            axes[0, 1].clear()
            axes[0, 1].scatter(self.X_test_scaled[:, 0], self.X_test_scaled[:, 1], 
                              c=self.y_test, alpha=0.6, label='Actual', cmap='viridis')
            axes[0, 1].scatter(self.X_test_scaled[:, 0], self.X_test_scaled[:, 1], 
                              c=y_test_pred, alpha=0.8, label='Predicted', cmap='viridis', marker='s')
            axes[0, 1].set_xlabel(self.feature_names[0])
            axes[0, 1].set_ylabel(self.feature_names[1])
            axes[0, 1].set_title('Test Predictions')
            
            # Plot accuracy progression
            axes[1, 0].clear()
            axes[1, 0].plot(iterations, train_accuracies, 'b-', label='Training Accuracy')
            axes[1, 0].plot(iterations, test_accuracies, 'r-', label='Test Accuracy')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_title('Accuracy Progression')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            axes[1, 0].set_ylim(0, 1)
            
            # Plot probability distribution
            axes[1, 1].clear()
            y_proba = model.predict_proba(self.X_train_scaled)[:, 1]
            axes[1, 1].hist(y_proba[self.y_train == 0], alpha=0.5, label='Class 0', bins=20)
            axes[1, 1].hist(y_proba[self.y_train == 1], alpha=0.5, label='Class 1', bins=20)
            axes[1, 1].set_xlabel('Predicted Probability')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Probability Distribution')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.pause(0.5)  # Pause to show progress
        
        # Final training on full dataset
        model.fit(self.X_train_scaled, self.y_train)
        y_train_pred = model.predict(self.X_train_scaled)
        y_test_pred = model.predict(self.X_test_scaled)
        y_train_proba = model.predict_proba(self.X_train_scaled)[:, 1]
        y_test_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Calculate final metrics
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        
        # Store model
        self.models['logistic_regression'] = {
            'model': model,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'y_train_proba': y_train_proba,
            'y_test_proba': y_test_proba
        }
        
        print(f"\nFinal Results:")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_test_pred))
        
        plt.show()
    
    def _visualize_logistic_regression_results(self, y_train_pred, y_test_pred, y_train_proba, y_test_proba):
        """Visualize Logistic Regression results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Logistic Regression Results', fontsize=16)
        
        # Decision boundary
        x_min, x_max = self.X_train_scaled[:, 0].min() - 0.5, self.X_train_scaled[:, 0].max() + 0.5
        y_min, y_max = self.X_train_scaled[:, 1].min() - 0.5, self.X_train_scaled[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        
        Z = self.models['logistic_regression']['model'].predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        axes[0, 0].contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
        axes[0, 0].scatter(self.X_train_scaled[:, 0], self.X_train_scaled[:, 1], 
                          c=self.y_train, alpha=0.8, cmap='viridis', edgecolors='black')
        axes[0, 0].set_xlabel(self.feature_names[0])
        axes[0, 0].set_ylabel(self.feature_names[1])
        axes[0, 0].set_title('Decision Boundary')
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
        axes[0, 1].set_title('Confusion Matrix')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        
        # ROC curve
        fpr, tpr, _ = roc_curve(self.y_test, y_test_proba)
        roc_auc = auc(fpr, tpr)
        
        axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1, 0].set_xlim([0.0, 1.0])
        axes[1, 0].set_ylim([0.0, 1.05])
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('ROC Curve')
        axes[1, 0].legend(loc="lower right")
        axes[1, 0].grid(True)
        
        # Probability distribution
        axes[1, 1].hist(y_test_proba[self.y_test == 0], alpha=0.5, label='Class 0', bins=20)
        axes[1, 1].hist(y_test_proba[self.y_test == 1], alpha=0.5, label='Class 1', bins=20)
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Test Set Probability Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show() 
        
    def decision_tree(self, max_depth=5, real_time_viz=True, task='classification'):
        """
        Train Decision Tree model with real-time visualization.
        
        Parameters:
        -----------
        max_depth : int
            Maximum depth of the tree
        real_time_viz : bool
            Whether to show real-time training visualization
        task : str
            'classification' or 'regression'
        """
        print(f"Training Decision Tree model ({task})...")
        
        if real_time_viz:
            self._decision_tree_realtime(max_depth, task)
        else:
            self._decision_tree_static(max_depth, task)
    
    def _decision_tree_static(self, max_depth, task):
        """Train Decision Tree with static visualization."""
        if task == 'classification':
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        else:
            model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        
        model.fit(self.X_train_scaled, self.y_train)
        
        # Make predictions
        y_train_pred = model.predict(self.X_train_scaled)
        y_test_pred = model.predict(self.X_test_scaled)
        
        # Calculate metrics
        if task == 'classification':
            train_accuracy = accuracy_score(self.y_train, y_train_pred)
            test_accuracy = accuracy_score(self.y_test, y_test_pred)
            
            self.models['decision_tree'] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'task': task
            }
            
            print(f"Training Accuracy: {train_accuracy:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_test_pred))
        else:
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            
            self.models['decision_tree'] = {
                'model': model,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'task': task
            }
            
            print(f"Training MSE: {train_mse:.4f}")
            print(f"Test MSE: {test_mse:.4f}")
            print(f"Training R²: {train_r2:.4f}")
            print(f"Test R²: {test_r2:.4f}")
        
        # Visualize results
        self._visualize_decision_tree_results(y_train_pred, y_test_pred, task)
    
    def _decision_tree_realtime(self, max_depth, task):
        """Train Decision Tree with real-time visualization."""
        if task == 'classification':
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        else:
            model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        
        # Create figure for real-time visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Decision Tree - Real-time Training ({task})', fontsize=16)
        
        # Initialize plots
        if task == 'classification':
            scatter = axes[0, 0].scatter(self.X_train_scaled[:, 0], self.X_train_scaled[:, 1], 
                                       c=self.y_train, alpha=0.6, cmap='viridis')
            axes[0, 0].set_xlabel(self.feature_names[0])
            axes[0, 0].set_ylabel(self.feature_names[1])
            axes[0, 0].set_title('Training Data')
            plt.colorbar(scatter, ax=axes[0, 0])
        else:
            axes[0, 0].scatter(self.X_train_scaled[:, 0], self.y_train, alpha=0.6, label='Training Data')
            axes[0, 0].set_xlabel(self.feature_names[0])
            axes[0, 0].set_ylabel(self.target_name)
            axes[0, 0].set_title('Training Data')
            axes[0, 0].legend()
        
        # Error tracking
        train_metrics = []
        test_metrics = []
        iterations = []
        
        # Train with different depths for visualization
        depths = list(range(1, max_depth + 1))
        
        for i, depth in enumerate(depths):
            # Set current depth
            model.max_depth = depth
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_train_pred = model.predict(self.X_train_scaled)
            y_test_pred = model.predict(self.X_test_scaled)
            
            # Calculate metrics
            if task == 'classification':
                train_accuracy = accuracy_score(self.y_train, y_train_pred)
                test_accuracy = accuracy_score(self.y_test, y_test_pred)
                train_metrics.append(train_accuracy)
                test_metrics.append(test_accuracy)
            else:
                train_mse = mean_squared_error(self.y_train, y_train_pred)
                test_mse = mean_squared_error(self.y_test, y_test_pred)
                train_metrics.append(train_mse)
                test_metrics.append(test_mse)
            
            iterations.append(depth)
            
            # Update decision boundary plot
            axes[0, 0].clear()
            if task == 'classification':
                scatter = axes[0, 0].scatter(self.X_train_scaled[:, 0], self.X_train_scaled[:, 1], 
                                           c=self.y_train, alpha=0.6, cmap='viridis')
                
                # Plot decision boundary
                x_min, x_max = self.X_train_scaled[:, 0].min() - 0.5, self.X_train_scaled[:, 0].max() + 0.5
                y_min, y_max = self.X_train_scaled[:, 1].min() - 0.5, self.X_train_scaled[:, 1].max() + 0.5
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                    np.arange(y_min, y_max, 0.02))
                
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                axes[0, 0].contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
                axes[0, 0].scatter(self.X_train_scaled[:, 0], self.X_train_scaled[:, 1], 
                                  c=self.y_train, alpha=0.8, cmap='viridis', edgecolors='black')
                axes[0, 0].set_xlabel(self.feature_names[0])
                axes[0, 0].set_ylabel(self.feature_names[1])
                axes[0, 0].set_title(f'Decision Boundary - Depth {depth}')
            else:
                axes[0, 0].scatter(self.X_train_scaled[:, 0], self.y_train, alpha=0.6, label='Training Data')
                
                # Sort for plotting
                sort_idx = np.argsort(self.X_train_scaled[:, 0])
                X_sorted = self.X_train_scaled[sort_idx, 0]
                y_pred_sorted = y_train_pred[sort_idx]
                
                axes[0, 0].plot(X_sorted, y_pred_sorted, 'r-', linewidth=2, label=f'Tree Prediction (Depth {depth})')
                axes[0, 0].set_xlabel(self.feature_names[0])
                axes[0, 0].set_ylabel(self.target_name)
                axes[0, 0].set_title(f'Regression Tree - Depth {depth}')
                axes[0, 0].legend()
            
            # Update test predictions
            axes[0, 1].clear()
            if task == 'classification':
                axes[0, 1].scatter(self.X_test_scaled[:, 0], self.X_test_scaled[:, 1], 
                                  c=self.y_test, alpha=0.6, label='Actual', cmap='viridis')
                axes[0, 1].scatter(self.X_test_scaled[:, 0], self.X_test_scaled[:, 1], 
                                  c=y_test_pred, alpha=0.8, label='Predicted', cmap='viridis', marker='s')
                axes[0, 1].set_xlabel(self.feature_names[0])
                axes[0, 1].set_ylabel(self.feature_names[1])
                axes[0, 1].set_title('Test Predictions')
            else:
                axes[0, 1].scatter(self.X_test_scaled[:, 0], self.y_test, alpha=0.6, label='Actual', color='red')
                axes[0, 1].scatter(self.X_test_scaled[:, 0], y_test_pred, alpha=0.8, label='Predicted', color='green')
                axes[0, 1].set_xlabel(self.feature_names[0])
                axes[0, 1].set_ylabel(self.target_name)
                axes[0, 1].set_title('Test Predictions vs Actual')
                axes[0, 1].legend()
            
            # Plot metric progression
            axes[1, 0].clear()
            if task == 'classification':
                axes[1, 0].plot(iterations, train_metrics, 'b-', label='Training Accuracy')
                axes[1, 0].plot(iterations, test_metrics, 'r-', label='Test Accuracy')
                axes[1, 0].set_ylabel('Accuracy')
                axes[1, 0].set_ylim(0, 1)
            else:
                axes[1, 0].plot(iterations, train_metrics, 'b-', label='Training MSE')
                axes[1, 0].plot(iterations, test_metrics, 'r-', label='Test MSE')
                axes[1, 0].set_ylabel('Mean Squared Error')
            
            axes[1, 0].set_xlabel('Tree Depth')
            axes[1, 0].set_title('Performance vs Tree Depth')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Feature importance
            axes[1, 1].clear()
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                axes[1, 1].bar(range(len(importances)), importances[indices])
                axes[1, 1].set_xticks(range(len(importances)))
                axes[1, 1].set_xticklabels([self.feature_names[i] for i in indices], rotation=45)
                axes[1, 1].set_ylabel('Feature Importance')
                axes[1, 1].set_title(f'Feature Importance (Depth {depth})')
            
            plt.tight_layout()
            plt.pause(0.8)  # Pause to show progress
        
        # Final training with specified max_depth
        model.max_depth = max_depth
        model.fit(self.X_train_scaled, self.y_train)
        y_train_pred = model.predict(self.X_train_scaled)
        y_test_pred = model.predict(self.X_test_scaled)
        
        # Calculate final metrics
        if task == 'classification':
            train_accuracy = accuracy_score(self.y_train, y_train_pred)
            test_accuracy = accuracy_score(self.y_test, y_test_pred)
            
            self.models['decision_tree'] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'task': task
            }
            
            print(f"\nFinal Results:")
            print(f"Training Accuracy: {train_accuracy:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_test_pred))
        else:
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            
            self.models['decision_tree'] = {
                'model': model,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'task': task
            }
            
            print(f"\nFinal Results:")
            print(f"Training MSE: {train_mse:.4f}")
            print(f"Test MSE: {test_mse:.4f}")
            print(f"Training R²: {train_r2:.4f}")
            print(f"Test R²: {test_r2:.4f}")
        
        plt.show()
    
    def _visualize_decision_tree_results(self, y_train_pred, y_test_pred, task):
        """Visualize Decision Tree results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Decision Tree Results ({task})', fontsize=16)
        
        model = self.models['decision_tree']['model']
        
        # Decision boundary or regression line
        if task == 'classification':
            x_min, x_max = self.X_train_scaled[:, 0].min() - 0.5, self.X_train_scaled[:, 0].max() + 0.5
            y_min, y_max = self.X_train_scaled[:, 1].min() - 0.5, self.X_train_scaled[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                np.arange(y_min, y_max, 0.02))
            
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            axes[0, 0].contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
            axes[0, 0].scatter(self.X_train_scaled[:, 0], self.X_train_scaled[:, 1], 
                              c=self.y_train, alpha=0.8, cmap='viridis', edgecolors='black')
            axes[0, 0].set_xlabel(self.feature_names[0])
            axes[0, 0].set_ylabel(self.feature_names[1])
            axes[0, 0].set_title('Decision Boundary')
        else:
            axes[0, 0].scatter(self.X_train_scaled[:, 0], self.y_train, alpha=0.6, label='Training Data')
            
            # Sort for plotting
            sort_idx = np.argsort(self.X_train_scaled[:, 0])
            X_sorted = self.X_train_scaled[sort_idx, 0]
            y_pred_sorted = y_train_pred[sort_idx]
            
            axes[0, 0].plot(X_sorted, y_pred_sorted, 'r-', linewidth=2, label='Tree Prediction')
            axes[0, 0].set_xlabel(self.feature_names[0])
            axes[0, 0].set_ylabel(self.target_name)
            axes[0, 0].set_title('Regression Tree')
            axes[0, 0].legend()
        
        # Test predictions
        if task == 'classification':
            axes[0, 1].scatter(self.X_test_scaled[:, 0], self.X_test_scaled[:, 1], 
                              c=self.y_test, alpha=0.6, label='Actual', cmap='viridis')
            axes[0, 1].scatter(self.X_test_scaled[:, 0], self.X_test_scaled[:, 1], 
                              c=y_test_pred, alpha=0.8, label='Predicted', cmap='viridis', marker='s')
            axes[0, 1].set_xlabel(self.feature_names[0])
            axes[0, 1].set_ylabel(self.feature_names[1])
            axes[0, 1].set_title('Test Predictions')
        else:
            axes[0, 1].scatter(self.X_test_scaled[:, 0], self.y_test, alpha=0.6, label='Actual', color='red')
            axes[0, 1].scatter(self.X_test_scaled[:, 0], y_test_pred, alpha=0.8, label='Predicted', color='green')
            axes[0, 1].set_xlabel(self.feature_names[0])
            axes[0, 1].set_ylabel(self.target_name)
            axes[0, 1].set_title('Test Predictions vs Actual')
            axes[0, 1].legend()
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            axes[1, 0].bar(range(len(importances)), importances[indices])
            axes[1, 0].set_xticks(range(len(importances)))
            axes[1, 0].set_xticklabels([self.feature_names[i] for i in indices], rotation=45)
            axes[1, 0].set_ylabel('Feature Importance')
            axes[1, 0].set_title('Feature Importance')
        
        # Metrics comparison
        if task == 'classification':
            metrics = ['Training Accuracy', 'Test Accuracy']
            values = [
                self.models['decision_tree']['train_accuracy'],
                self.models['decision_tree']['test_accuracy']
            ]
        else:
            metrics = ['Training MSE', 'Test MSE', 'Training R²', 'Test R²']
            values = [
                self.models['decision_tree']['train_mse'],
                self.models['decision_tree']['test_mse'],
                self.models['decision_tree']['train_r2'],
                self.models['decision_tree']['test_r2']
            ]
        
        bars = axes[1, 1].bar(metrics, values, color=['blue', 'red', 'green', 'orange'][:len(metrics)])
        axes[1, 1].set_title('Model Performance Metrics')
        axes[1, 1].set_ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show() 
        
    def support_vector_machine(self, kernel='rbf', real_time_viz=True, task='classification'):
        """
        Train Support Vector Machine model with real-time visualization.
        
        Parameters:
        -----------
        kernel : str
            Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
        real_time_viz : bool
            Whether to show real-time training visualization
        task : str
            'classification' or 'regression'
        """
        print(f"Training SVM model ({task}) with {kernel} kernel...")
        
        if real_time_viz:
            self._svm_realtime(kernel, task)
        else:
            self._svm_static(kernel, task)
    
    def _svm_static(self, kernel, task):
        """Train SVM with static visualization."""
        if task == 'classification':
            model = SVC(kernel=kernel, random_state=42, probability=True)
        else:
            model = SVR(kernel=kernel)
        
        model.fit(self.X_train_scaled, self.y_train)
        
        # Make predictions
        y_train_pred = model.predict(self.X_train_scaled)
        y_test_pred = model.predict(self.X_test_scaled)
        
        # Calculate metrics
        if task == 'classification':
            train_accuracy = accuracy_score(self.y_train, y_train_pred)
            test_accuracy = accuracy_score(self.y_test, y_test_pred)
            y_test_proba = model.predict_proba(self.X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            self.models['svm'] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'y_test_proba': y_test_proba,
                'task': task,
                'kernel': kernel
            }
            
            print(f"Training Accuracy: {train_accuracy:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_test_pred))
        else:
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            
            self.models['svm'] = {
                'model': model,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'task': task,
                'kernel': kernel
            }
            
            print(f"Training MSE: {train_mse:.4f}")
            print(f"Test MSE: {test_mse:.4f}")
            print(f"Training R²: {train_r2:.4f}")
            print(f"Test R²: {test_r2:.4f}")
        
        # Visualize results
        self._visualize_svm_results(y_train_pred, y_test_pred, task)
    
    def _svm_realtime(self, kernel, task):
        """Train SVM with real-time visualization."""
        if task == 'classification':
            model = SVC(kernel=kernel, random_state=42, probability=True)
        else:
            model = SVR(kernel=kernel)
        
        # Create figure for real-time visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'SVM - Real-time Training ({task}, {kernel} kernel)', fontsize=16)
        
        # Initialize plots
        if task == 'classification':
            scatter = axes[0, 0].scatter(self.X_train_scaled[:, 0], self.X_train_scaled[:, 1], 
                                       c=self.y_train, alpha=0.6, cmap='viridis')
            axes[0, 0].set_xlabel(self.feature_names[0])
            axes[0, 0].set_ylabel(self.feature_names[1])
            axes[0, 0].set_title('Training Data')
            plt.colorbar(scatter, ax=axes[0, 0])
        else:
            axes[0, 0].scatter(self.X_train_scaled[:, 0], self.y_train, alpha=0.6, label='Training Data')
            axes[0, 0].set_xlabel(self.feature_names[0])
            axes[0, 0].set_ylabel(self.target_name)
            axes[0, 0].set_title('Training Data')
            axes[0, 0].legend()
        
        # Train model
        model.fit(self.X_train_scaled, self.y_train)
        
        # Make predictions
        y_train_pred = model.predict(self.X_train_scaled)
        y_test_pred = model.predict(self.X_test_scaled)
        
        # Calculate metrics
        if task == 'classification':
            train_accuracy = accuracy_score(self.y_train, y_train_pred)
            test_accuracy = accuracy_score(self.y_test, y_test_pred)
            y_test_proba = model.predict_proba(self.X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            self.models['svm'] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'y_test_proba': y_test_proba,
                'task': task,
                'kernel': kernel
            }
            
            print(f"Training Accuracy: {train_accuracy:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_test_pred))
        else:
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            
            self.models['svm'] = {
                'model': model,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'task': task,
                'kernel': kernel
            }
            
            print(f"Training MSE: {train_mse:.4f}")
            print(f"Test MSE: {test_mse:.4f}")
            print(f"Training R²: {train_r2:.4f}")
            print(f"Test R²: {test_r2:.4f}")
        
        # Update decision boundary plot
        if task == 'classification':
            axes[0, 0].clear()
            scatter = axes[0, 0].scatter(self.X_train_scaled[:, 0], self.X_train_scaled[:, 1], 
                                       c=self.y_train, alpha=0.6, cmap='viridis')
            
            # Plot decision boundary
            x_min, x_max = self.X_train_scaled[:, 0].min() - 0.5, self.X_train_scaled[:, 0].max() + 0.5
            y_min, y_max = self.X_train_scaled[:, 1].min() - 0.5, self.X_train_scaled[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                np.arange(y_min, y_max, 0.02))
            
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            axes[0, 0].contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
            axes[0, 0].scatter(self.X_train_scaled[:, 0], self.X_train_scaled[:, 1], 
                              c=self.y_train, alpha=0.8, cmap='viridis', edgecolors='black')
            axes[0, 0].set_xlabel(self.feature_names[0])
            axes[0, 0].set_ylabel(self.feature_names[1])
            axes[0, 0].set_title(f'SVM Decision Boundary ({kernel} kernel)')
        else:
            axes[0, 0].clear()
            axes[0, 0].scatter(self.X_train_scaled[:, 0], self.y_train, alpha=0.6, label='Training Data')
            
            # Sort for plotting
            sort_idx = np.argsort(self.X_train_scaled[:, 0])
            X_sorted = self.X_train_scaled[sort_idx, 0]
            y_pred_sorted = y_train_pred[sort_idx]
            
            axes[0, 0].plot(X_sorted, y_pred_sorted, 'r-', linewidth=2, label=f'SVM Prediction ({kernel} kernel)')
            axes[0, 0].set_xlabel(self.feature_names[0])
            axes[0, 0].set_ylabel(self.target_name)
            axes[0, 0].set_title(f'SVM Regression ({kernel} kernel)')
            axes[0, 0].legend()
        
        # Update test predictions
        axes[0, 1].clear()
        if task == 'classification':
            axes[0, 1].scatter(self.X_test_scaled[:, 0], self.X_test_scaled[:, 1], 
                              c=self.y_test, alpha=0.6, label='Actual', cmap='viridis')
            axes[0, 1].scatter(self.X_test_scaled[:, 0], self.X_test_scaled[:, 1], 
                              c=y_test_pred, alpha=0.8, label='Predicted', cmap='viridis', marker='s')
            axes[0, 1].set_xlabel(self.feature_names[0])
            axes[0, 1].set_ylabel(self.feature_names[1])
            axes[0, 1].set_title('Test Predictions')
        else:
            axes[0, 1].scatter(self.X_test_scaled[:, 0], self.y_test, alpha=0.6, label='Actual', color='red')
            axes[0, 1].scatter(self.X_test_scaled[:, 0], y_test_pred, alpha=0.8, label='Predicted', color='green')
            axes[0, 1].set_xlabel(self.feature_names[0])
            axes[0, 1].set_ylabel(self.target_name)
            axes[0, 1].set_title('Test Predictions vs Actual')
            axes[0, 1].legend()
        
        # Support vectors
        if hasattr(model, 'support_vectors_'):
            axes[1, 0].scatter(self.X_train_scaled[:, 0], self.X_train_scaled[:, 1], 
                              alpha=0.3, c='lightgray', label='All Points')
            axes[1, 0].scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], 
                              c='red', s=100, label='Support Vectors', edgecolors='black')
            axes[1, 0].set_xlabel(self.feature_names[0])
            axes[1, 0].set_ylabel(self.feature_names[1])
            axes[1, 0].set_title('Support Vectors')
            axes[1, 0].legend()
        
        # Metrics
        if task == 'classification':
            metrics = ['Training Accuracy', 'Test Accuracy']
            values = [train_accuracy, test_accuracy]
        else:
            metrics = ['Training MSE', 'Test MSE', 'Training R²', 'Test R²']
            values = [train_mse, test_mse, train_r2, test_r2]
        
        bars = axes[1, 1].bar(metrics, values, color=['blue', 'red', 'green', 'orange'][:len(metrics)])
        axes[1, 1].set_title('Model Performance Metrics')
        axes[1, 1].set_ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def _visualize_svm_results(self, y_train_pred, y_test_pred, task):
        """Visualize SVM results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'SVM Results ({task}, {self.models["svm"]["kernel"]} kernel)', fontsize=16)
        
        model = self.models['svm']['model']
        
        # Decision boundary or regression line
        if task == 'classification':
            x_min, x_max = self.X_train_scaled[:, 0].min() - 0.5, self.X_train_scaled[:, 0].max() + 0.5
            y_min, y_max = self.X_train_scaled[:, 1].min() - 0.5, self.X_train_scaled[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                np.arange(y_min, y_max, 0.02))
            
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            axes[0, 0].contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
            axes[0, 0].scatter(self.X_train_scaled[:, 0], self.X_train_scaled[:, 1], 
                              c=self.y_train, alpha=0.8, cmap='viridis', edgecolors='black')
            axes[0, 0].set_xlabel(self.feature_names[0])
            axes[0, 0].set_ylabel(self.feature_names[1])
            axes[0, 0].set_title('Decision Boundary')
        else:
            axes[0, 0].scatter(self.X_train_scaled[:, 0], self.y_train, alpha=0.6, label='Training Data')
            
            # Sort for plotting
            sort_idx = np.argsort(self.X_train_scaled[:, 0])
            X_sorted = self.X_train_scaled[sort_idx, 0]
            y_pred_sorted = y_train_pred[sort_idx]
            
            axes[0, 0].plot(X_sorted, y_pred_sorted, 'r-', linewidth=2, label='SVM Prediction')
            axes[0, 0].set_xlabel(self.feature_names[0])
            axes[0, 0].set_ylabel(self.target_name)
            axes[0, 0].set_title('SVM Regression')
            axes[0, 0].legend()
        
        # Test predictions
        if task == 'classification':
            axes[0, 1].scatter(self.X_test_scaled[:, 0], self.X_test_scaled[:, 1], 
                              c=self.y_test, alpha=0.6, label='Actual', cmap='viridis')
            axes[0, 1].scatter(self.X_test_scaled[:, 0], self.X_test_scaled[:, 1], 
                              c=y_test_pred, alpha=0.8, label='Predicted', cmap='viridis', marker='s')
            axes[0, 1].set_xlabel(self.feature_names[0])
            axes[0, 1].set_ylabel(self.feature_names[1])
            axes[0, 1].set_title('Test Predictions')
        else:
            axes[0, 1].scatter(self.X_test_scaled[:, 0], self.y_test, alpha=0.6, label='Actual', color='red')
            axes[0, 1].scatter(self.X_test_scaled[:, 0], y_test_pred, alpha=0.8, label='Predicted', color='green')
            axes[0, 1].set_xlabel(self.feature_names[0])
            axes[0, 1].set_ylabel(self.target_name)
            axes[0, 1].set_title('Test Predictions vs Actual')
            axes[0, 1].legend()
        
        # Support vectors
        if hasattr(model, 'support_vectors_'):
            axes[1, 0].scatter(self.X_train_scaled[:, 0], self.X_train_scaled[:, 1], 
                              alpha=0.3, c='lightgray', label='All Points')
            axes[1, 0].scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], 
                              c='red', s=100, label='Support Vectors', edgecolors='black')
            axes[1, 0].set_xlabel(self.feature_names[0])
            axes[1, 0].set_ylabel(self.feature_names[1])
            axes[1, 0].set_title('Support Vectors')
            axes[1, 0].legend()
        
        # Metrics comparison
        if task == 'classification':
            metrics = ['Training Accuracy', 'Test Accuracy']
            values = [
                self.models['svm']['train_accuracy'],
                self.models['svm']['test_accuracy']
            ]
        else:
            metrics = ['Training MSE', 'Test MSE', 'Training R²', 'Test R²']
            values = [
                self.models['svm']['train_mse'],
                self.models['svm']['test_mse'],
                self.models['svm']['train_r2'],
                self.models['svm']['test_r2']
            ]
        
        bars = axes[1, 1].bar(metrics, values, color=['blue', 'red', 'green', 'orange'][:len(metrics)])
        axes[1, 1].set_title('Model Performance Metrics')
        axes[1, 1].set_ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def k_nearest_neighbors(self, n_neighbors=5, real_time_viz=True, task='classification'):
        """
        Train K-Nearest Neighbors model with real-time visualization.
        
        Parameters:
        -----------
        n_neighbors : int
            Number of neighbors to consider
        real_time_viz : bool
            Whether to show real-time training visualization
        task : str
            'classification' or 'regression'
        """
        print(f"Training KNN model ({task}) with {n_neighbors} neighbors...")
        
        if real_time_viz:
            self._knn_realtime(n_neighbors, task)
        else:
            self._knn_static(n_neighbors, task)
    
    def _knn_static(self, n_neighbors, task):
        """Train KNN with static visualization."""
        if task == 'classification':
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
        else:
            model = KNeighborsRegressor(n_neighbors=n_neighbors)
        
        model.fit(self.X_train_scaled, self.y_train)
        
        # Make predictions
        y_train_pred = model.predict(self.X_train_scaled)
        y_test_pred = model.predict(self.X_test_scaled)
        
        # Calculate metrics
        if task == 'classification':
            train_accuracy = accuracy_score(self.y_train, y_train_pred)
            test_accuracy = accuracy_score(self.y_test, y_test_pred)
            
            self.models['knn'] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'task': task,
                'n_neighbors': n_neighbors
            }
            
            print(f"Training Accuracy: {train_accuracy:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_test_pred))
        else:
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            
            self.models['knn'] = {
                'model': model,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'task': task,
                'n_neighbors': n_neighbors
            }
            
            print(f"Training MSE: {train_mse:.4f}")
            print(f"Test MSE: {test_mse:.4f}")
            print(f"Training R²: {train_r2:.4f}")
            print(f"Test R²: {test_r2:.4f}")
        
        # Visualize results
        self._visualize_knn_results(y_train_pred, y_test_pred, task)
    
    def _knn_realtime(self, n_neighbors, task):
        """Train KNN with real-time visualization."""
        if task == 'classification':
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
        else:
            model = KNeighborsRegressor(n_neighbors=n_neighbors)
        
        # Create figure for real-time visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'KNN - Real-time Training ({task}, k={n_neighbors})', fontsize=16)
        
        # Initialize plots
        if task == 'classification':
            scatter = axes[0, 0].scatter(self.X_train_scaled[:, 0], self.X_train_scaled[:, 1], 
                                       c=self.y_train, alpha=0.6, cmap='viridis')
            axes[0, 0].set_xlabel(self.feature_names[0])
            axes[0, 0].set_ylabel(self.feature_names[1])
            axes[0, 0].set_title('Training Data')
            plt.colorbar(scatter, ax=axes[0, 0])
        else:
            axes[0, 0].scatter(self.X_train_scaled[:, 0], self.y_train, alpha=0.6, label='Training Data')
            axes[0, 0].set_xlabel(self.feature_names[0])
            axes[0, 0].set_ylabel(self.target_name)
            axes[0, 0].set_title('Training Data')
            axes[0, 0].legend()
        
        # Train model
        model.fit(self.X_train_scaled, self.y_train)
        
        # Make predictions
        y_train_pred = model.predict(self.X_train_scaled)
        y_test_pred = model.predict(self.X_test_scaled)
        
        # Calculate metrics
        if task == 'classification':
            train_accuracy = accuracy_score(self.y_train, y_train_pred)
            test_accuracy = accuracy_score(self.y_test, y_test_pred)
            
            self.models['knn'] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'task': task,
                'n_neighbors': n_neighbors
            }
            
            print(f"Training Accuracy: {train_accuracy:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_test_pred))
        else:
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            
            self.models['knn'] = {
                'model': model,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'task': task,
                'n_neighbors': n_neighbors
            }
            
            print(f"Training MSE: {train_mse:.4f}")
            print(f"Test MSE: {test_mse:.4f}")
            print(f"Training R²: {train_r2:.4f}")
            print(f"Test R²: {test_r2:.4f}")
        
        # Update decision boundary plot
        if task == 'classification':
            axes[0, 0].clear()
            scatter = axes[0, 0].scatter(self.X_train_scaled[:, 0], self.X_train_scaled[:, 1], 
                                       c=self.y_train, alpha=0.6, cmap='viridis')
            
            # Plot decision boundary
            x_min, x_max = self.X_train_scaled[:, 0].min() - 0.5, self.X_train_scaled[:, 0].max() + 0.5
            y_min, y_max = self.X_train_scaled[:, 1].min() - 0.5, self.X_train_scaled[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                np.arange(y_min, y_max, 0.02))
            
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            axes[0, 0].contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
            axes[0, 0].scatter(self.X_train_scaled[:, 0], self.X_train_scaled[:, 1], 
                              c=self.y_train, alpha=0.8, cmap='viridis', edgecolors='black')
            axes[0, 0].set_xlabel(self.feature_names[0])
            axes[0, 0].set_ylabel(self.feature_names[1])
            axes[0, 0].set_title(f'KNN Decision Boundary (k={n_neighbors})')
        else:
            axes[0, 0].clear()
            axes[0, 0].scatter(self.X_train_scaled[:, 0], self.y_train, alpha=0.6, label='Training Data')
            
            # Sort for plotting
            sort_idx = np.argsort(self.X_train_scaled[:, 0])
            X_sorted = self.X_train_scaled[sort_idx, 0]
            y_pred_sorted = y_train_pred[sort_idx]
            
            axes[0, 0].plot(X_sorted, y_pred_sorted, 'r-', linewidth=2, label=f'KNN Prediction (k={n_neighbors})')
            axes[0, 0].set_xlabel(self.feature_names[0])
            axes[0, 0].set_ylabel(self.target_name)
            axes[0, 0].set_title(f'KNN Regression (k={n_neighbors})')
            axes[0, 0].legend()
        
        # Update test predictions
        axes[0, 1].clear()
        if task == 'classification':
            axes[0, 1].scatter(self.X_test_scaled[:, 0], self.X_test_scaled[:, 1], 
                              c=self.y_test, alpha=0.6, label='Actual', cmap='viridis')
            axes[0, 1].scatter(self.X_test_scaled[:, 0], self.X_test_scaled[:, 1], 
                              c=y_test_pred, alpha=0.8, label='Predicted', cmap='viridis', marker='s')
            axes[0, 1].set_xlabel(self.feature_names[0])
            axes[0, 1].set_ylabel(self.feature_names[1])
            axes[0, 1].set_title('Test Predictions')
        else:
            axes[0, 1].scatter(self.X_test_scaled[:, 0], self.y_test, alpha=0.6, label='Actual', color='red')
            axes[0, 1].scatter(self.X_test_scaled[:, 0], y_test_pred, alpha=0.8, label='Predicted', color='green')
            axes[0, 1].set_xlabel(self.feature_names[0])
            axes[0, 1].set_ylabel(self.target_name)
            axes[0, 1].set_title('Test Predictions vs Actual')
            axes[0, 1].legend()
        
        # K value analysis
        k_values = list(range(1, min(21, len(self.X_train_scaled) // 2)))
        k_accuracies = []
        
        for k in k_values:
            if task == 'classification':
                knn_temp = KNeighborsClassifier(n_neighbors=k)
            else:
                knn_temp = KNeighborsRegressor(n_neighbors=k)
            
            knn_temp.fit(self.X_train_scaled, self.y_train)
            y_pred_temp = knn_temp.predict(self.X_test_scaled)
            
            if task == 'classification':
                k_accuracies.append(accuracy_score(self.y_test, y_pred_temp))
            else:
                k_accuracies.append(r2_score(self.y_test, y_pred_temp))
        
        axes[1, 0].plot(k_values, k_accuracies, 'b-o')
        axes[1, 0].axvline(x=n_neighbors, color='red', linestyle='--', label=f'Selected k={n_neighbors}')
        axes[1, 0].set_xlabel('Number of Neighbors (k)')
        axes[1, 0].set_ylabel('Accuracy' if task == 'classification' else 'R² Score')
        axes[1, 0].set_title('Performance vs k Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Metrics
        if task == 'classification':
            metrics = ['Training Accuracy', 'Test Accuracy']
            values = [train_accuracy, test_accuracy]
        else:
            metrics = ['Training MSE', 'Test MSE', 'Training R²', 'Test R²']
            values = [train_mse, test_mse, train_r2, test_r2]
        
        bars = axes[1, 1].bar(metrics, values, color=['blue', 'red', 'green', 'orange'][:len(metrics)])
        axes[1, 1].set_title('Model Performance Metrics')
        axes[1, 1].set_ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def _visualize_knn_results(self, y_train_pred, y_test_pred, task):
        """Visualize KNN results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'KNN Results ({task}, k={self.models["knn"]["n_neighbors"]})', fontsize=16)
        
        model = self.models['knn']['model']
        
        # Decision boundary or regression line
        if task == 'classification':
            x_min, x_max = self.X_train_scaled[:, 0].min() - 0.5, self.X_train_scaled[:, 0].max() + 0.5
            y_min, y_max = self.X_train_scaled[:, 1].min() - 0.5, self.X_train_scaled[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                np.arange(y_min, y_max, 0.02))
            
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            axes[0, 0].contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
            axes[0, 0].scatter(self.X_train_scaled[:, 0], self.X_train_scaled[:, 1], 
                              c=self.y_train, alpha=0.8, cmap='viridis', edgecolors='black')
            axes[0, 0].set_xlabel(self.feature_names[0])
            axes[0, 0].set_ylabel(self.feature_names[1])
            axes[0, 0].set_title('Decision Boundary')
        else:
            axes[0, 0].scatter(self.X_train_scaled[:, 0], self.y_train, alpha=0.6, label='Training Data')
            
            # Sort for plotting
            sort_idx = np.argsort(self.X_train_scaled[:, 0])
            X_sorted = self.X_train_scaled[sort_idx, 0]
            y_pred_sorted = y_train_pred[sort_idx]
            
            axes[0, 0].plot(X_sorted, y_pred_sorted, 'r-', linewidth=2, label='KNN Prediction')
            axes[0, 0].set_xlabel(self.feature_names[0])
            axes[0, 0].set_ylabel(self.target_name)
            axes[0, 0].set_title('KNN Regression')
            axes[0, 0].legend()
        
        # Test predictions
        if task == 'classification':
            axes[0, 1].scatter(self.X_test_scaled[:, 0], self.X_test_scaled[:, 1], 
                              c=self.y_test, alpha=0.6, label='Actual', cmap='viridis')
            axes[0, 1].scatter(self.X_test_scaled[:, 0], self.X_test_scaled[:, 1], 
                              c=y_test_pred, alpha=0.8, label='Predicted', cmap='viridis', marker='s')
            axes[0, 1].set_xlabel(self.feature_names[0])
            axes[0, 1].set_ylabel(self.feature_names[1])
            axes[0, 1].set_title('Test Predictions')
        else:
            axes[0, 1].scatter(self.X_test_scaled[:, 0], self.y_test, alpha=0.6, label='Actual', color='red')
            axes[0, 1].scatter(self.X_test_scaled[:, 0], y_test_pred, alpha=0.8, label='Predicted', color='green')
            axes[0, 1].set_xlabel(self.feature_names[0])
            axes[0, 1].set_ylabel(self.target_name)
            axes[0, 1].set_title('Test Predictions vs Actual')
            axes[0, 1].legend()
        
        # K value analysis
        k_values = list(range(1, min(21, len(self.X_train_scaled) // 2)))
        k_accuracies = []
        
        for k in k_values:
            if task == 'classification':
                knn_temp = KNeighborsClassifier(n_neighbors=k)
            else:
                knn_temp = KNeighborsRegressor(n_neighbors=k)
            
            knn_temp.fit(self.X_train_scaled, self.y_train)
            y_pred_temp = knn_temp.predict(self.X_test_scaled)
            
            if task == 'classification':
                k_accuracies.append(accuracy_score(self.y_test, y_pred_temp))
            else:
                k_accuracies.append(r2_score(self.y_test, y_pred_temp))
        
        axes[1, 0].plot(k_values, k_accuracies, 'b-o')
        axes[1, 0].axvline(x=self.models['knn']['n_neighbors'], color='red', linestyle='--', 
                          label=f'Selected k={self.models["knn"]["n_neighbors"]}')
        axes[1, 0].set_xlabel('Number of Neighbors (k)')
        axes[1, 0].set_ylabel('Accuracy' if task == 'classification' else 'R² Score')
        axes[1, 0].set_title('Performance vs k Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Metrics comparison
        if task == 'classification':
            metrics = ['Training Accuracy', 'Test Accuracy']
            values = [
                self.models['knn']['train_accuracy'],
                self.models['knn']['test_accuracy']
            ]
        else:
            metrics = ['Training MSE', 'Test MSE', 'Training R²', 'Test R²']
            values = [
                self.models['knn']['train_mse'],
                self.models['knn']['test_mse'],
                self.models['knn']['train_r2'],
                self.models['knn']['test_r2']
            ]
        
        bars = axes[1, 1].bar(metrics, values, color=['blue', 'red', 'green', 'orange'][:len(metrics)])
        axes[1, 1].set_title('Model Performance Metrics')
        axes[1, 1].set_ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def compare_models(self):
        """Compare all trained models."""
        if not self.models:
            print("No models trained yet. Please train at least one model first.")
            return
        
        print("Model Comparison:")
        print("=" * 50)
        
        for model_name, model_info in self.models.items():
            print(f"\n{model_name.upper().replace('_', ' ')}:")
            print("-" * 30)
            
            if 'task' in model_info:
                task = model_info['task']
                if task == 'classification':
                    train_acc = model_info.get('train_accuracy', 'N/A')
                    test_acc = model_info.get('test_accuracy', 'N/A')
                    print(f"Training Accuracy: {train_acc:.4f}" if train_acc != 'N/A' else f"Training Accuracy: {train_acc}")
                    print(f"Test Accuracy: {test_acc:.4f}" if test_acc != 'N/A' else f"Test Accuracy: {test_acc}")
                else:
                    train_mse = model_info.get('train_mse', 'N/A')
                    test_mse = model_info.get('test_mse', 'N/A')
                    train_r2 = model_info.get('train_r2', 'N/A')
                    test_r2 = model_info.get('test_r2', 'N/A')
                    print(f"Training MSE: {train_mse:.4f}" if train_mse != 'N/A' else f"Training MSE: {train_mse}")
                    print(f"Test MSE: {test_mse:.4f}" if test_mse != 'N/A' else f"Test MSE: {test_mse}")
                    print(f"Training R²: {train_r2:.4f}" if train_r2 != 'N/A' else f"Training R²: {train_r2}")
                    print(f"Test R²: {test_r2:.4f}" if test_r2 != 'N/A' else f"Test R²: {test_r2}")
            else:
                # Linear regression
                train_mse = model_info.get('train_mse', 'N/A')
                test_mse = model_info.get('test_mse', 'N/A')
                train_r2 = model_info.get('train_r2', 'N/A')
                test_r2 = model_info.get('test_r2', 'N/A')
                print(f"Training MSE: {train_mse:.4f}" if train_mse != 'N/A' else f"Training MSE: {train_mse}")
                print(f"Test MSE: {test_mse:.4f}" if test_mse != 'N/A' else f"Test MSE: {test_mse}")
                print(f"Training R²: {train_r2:.4f}" if train_r2 != 'N/A' else f"Training R²: {train_r2}")
                print(f"Test R²: {test_r2:.4f}" if test_r2 != 'N/A' else f"Test R²: {test_r2}")
        
        # Create comparison plot
        self._plot_model_comparison()
    
    def _plot_model_comparison(self):
        """Plot comparison of all trained models."""
        if not self.models:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Model Comparison', fontsize=16)
        
        model_names = []
        train_scores = []
        test_scores = []
        
        for model_name, model_info in self.models.items():
            model_names.append(model_name.replace('_', ' ').title())
            
            if 'task' in model_info:
                task = model_info['task']
                if task == 'classification':
                    train_scores.append(model_info.get('train_accuracy', 0))
                    test_scores.append(model_info.get('test_accuracy', 0))
                else:
                    train_scores.append(model_info.get('train_r2', 0))
                    test_scores.append(model_info.get('test_r2', 0))
            else:
                # Linear regression
                train_scores.append(model_info.get('train_r2', 0))
                test_scores.append(model_info.get('test_r2', 0))
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = axes[0].bar(x - width/2, train_scores, width, label='Training', color='blue', alpha=0.7)
        bars2 = axes[0].bar(x + width/2, test_scores, width, label='Test', color='red', alpha=0.7)
        
        axes[0].set_xlabel('Models')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Model Performance Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(model_names, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
        
        # Error comparison (MSE for regression, 1-accuracy for classification)
        error_scores = []
        for model_name, model_info in self.models.items():
            if 'task' in model_info:
                task = model_info['task']
                if task == 'classification':
                    error_scores.append(1 - model_info.get('test_accuracy', 0))
                else:
                    error_scores.append(model_info.get('test_mse', 0))
            else:
                # Linear regression
                error_scores.append(model_info.get('test_mse', 0))
        
        bars3 = axes[1].bar(model_names, error_scores, color='orange', alpha=0.7)
        axes[1].set_xlabel('Models')
        axes[1].set_ylabel('Error Score')
        axes[1].set_title('Model Error Comparison')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars3:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show() 