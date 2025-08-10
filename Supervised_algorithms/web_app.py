from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
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

app = Flask(__name__)

class WebSupervisedLearning:
    def __init__(self):
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """Prepare data with train-test split and scaling"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        return {
            'X_train': self.X_train.tolist(),
            'X_test': self.X_test.tolist(),
            'y_train': self.y_train.tolist(),
            'y_test': self.y_test.tolist()
        }
    
    def linear_regression_step_by_step(self):
        """Linear Regression with step-by-step training data"""
        model = LinearRegression()
        
        # Simulate training with mini-batches
        n_samples = len(self.X_train)
        batch_size = max(1, n_samples // 20)
        
        training_steps = []
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = self.X_train[i:end_idx]
            y_batch = self.y_train[i:end_idx]
            
            # Fit model on batch
            model.fit(X_batch, y_batch)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # Store step data
            step_data = {
                'step': len(training_steps),
                'progress': (i + batch_size) / n_samples,
                'mse': float(mse),
                'r2': float(r2),
                'predictions': y_pred.tolist(),
                'coefficients': model.coef_.tolist(),
                'intercept': float(model.intercept_)
            }
            training_steps.append(step_data)
        
        return training_steps
    
    def logistic_regression_step_by_step(self):
        """Logistic Regression with step-by-step training data"""
        model = LogisticRegression(max_iter=1000, random_state=42)
        
        # Simulate training with mini-batches
        n_samples = len(self.X_train)
        batch_size = max(1, n_samples // 20)
        
        training_steps = []
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = self.X_train[i:end_idx]
            y_batch = self.y_train[i:end_idx]
            
            # Fit model on batch
            model.fit(X_batch, y_batch)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Store step data
            step_data = {
                'step': len(training_steps),
                'progress': (i + batch_size) / n_samples,
                'accuracy': float(accuracy),
                'predictions': y_pred.tolist(),
                'probabilities': y_proba.tolist(),
                'coefficients': model.coef_.tolist(),
                'intercept': float(model.intercept_)
            }
            training_steps.append(step_data)
        
        return training_steps
    
    def decision_tree_step_by_step(self, max_depth=5, task='classification'):
        """Decision Tree with step-by-step training data"""
        if task == 'classification':
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        else:
            model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        
        # Simulate training with mini-batches
        n_samples = len(self.X_train)
        batch_size = max(1, n_samples // 20)
        
        training_steps = []
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = self.X_train[i:end_idx]
            y_batch = self.y_train[i:end_idx]
            
            # Fit model on batch
            model.fit(X_batch, y_batch)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            if task == 'classification':
                accuracy = accuracy_score(self.y_test, y_pred)
                step_data = {
                    'step': len(training_steps),
                    'progress': (i + batch_size) / n_samples,
                    'accuracy': float(accuracy),
                    'predictions': y_pred.tolist(),
                    'feature_importance': model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else []
                }
            else:
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                step_data = {
                    'step': len(training_steps),
                    'progress': (i + batch_size) / n_samples,
                    'mse': float(mse),
                    'r2': float(r2),
                    'predictions': y_pred.tolist(),
                    'feature_importance': model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else []
                }
            
            training_steps.append(step_data)
        
        return training_steps
    
    def svm_step_by_step(self, task='classification'):
        """SVM with step-by-step training data"""
        if task == 'classification':
            model = SVC(kernel='rbf', probability=True, random_state=42)
        else:
            model = SVR(kernel='rbf')
        
        # Simulate training with mini-batches
        n_samples = len(self.X_train)
        batch_size = max(1, n_samples // 20)
        
        training_steps = []
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = self.X_train[i:end_idx]
            y_batch = self.y_train[i:end_idx]
            
            # Fit model on batch
            model.fit(X_batch, y_batch)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            if task == 'classification':
                accuracy = accuracy_score(self.y_test, y_pred)
                step_data = {
                    'step': len(training_steps),
                    'progress': (i + batch_size) / n_samples,
                    'accuracy': float(accuracy),
                    'predictions': y_pred.tolist(),
                    'support_vectors': model.support_vectors_.tolist() if hasattr(model, 'support_vectors_') else []
                }
            else:
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                step_data = {
                    'step': len(training_steps),
                    'progress': (i + batch_size) / n_samples,
                    'mse': float(mse),
                    'r2': float(r2),
                    'predictions': y_pred.tolist(),
                    'support_vectors': model.support_vectors_.tolist() if hasattr(model, 'support_vectors_') else []
                }
            
            training_steps.append(step_data)
        
        return training_steps
    
    def knn_step_by_step(self, task='classification', n_neighbors=5):
        """KNN with step-by-step training data"""
        if task == 'classification':
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
        else:
            model = KNeighborsRegressor(n_neighbors=n_neighbors)
        
        # Simulate training with mini-batches
        n_samples = len(self.X_train)
        batch_size = max(1, n_samples // 20)
        
        training_steps = []
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = self.X_train[i:end_idx]
            y_batch = self.y_train[i:end_idx]
            
            # Fit model on batch
            model.fit(X_batch, y_batch)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            if task == 'classification':
                accuracy = accuracy_score(self.y_test, y_pred)
                step_data = {
                    'step': len(training_steps),
                    'progress': (i + batch_size) / n_samples,
                    'accuracy': float(accuracy),
                    'predictions': y_pred.tolist(),
                    'n_neighbors': n_neighbors
                }
            else:
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                step_data = {
                    'step': len(training_steps),
                    'progress': (i + batch_size) / n_samples,
                    'mse': float(mse),
                    'r2': float(r2),
                    'predictions': y_pred.tolist(),
                    'n_neighbors': n_neighbors
                }
            
            training_steps.append(step_data)
        
        return training_steps

# Initialize the learning module
slm = WebSupervisedLearning()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate_data', methods=['POST'])
def generate_data():
    data = request.json
    dataset_type = data.get('dataset_type')
    
    if dataset_type == 'synthetic_classification':
        n_samples = data.get('n_samples', 500)
        n_features = data.get('n_features', 2)
        n_classes = data.get('n_classes', 2)
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_clusters_per_class=1,
            n_redundant=0,
            random_state=42
        )
        
    elif dataset_type == 'synthetic_regression':
        n_samples = data.get('n_samples', 500)
        n_features = data.get('n_features', 1)
        
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=0.1,
            random_state=42
        )
        
    elif dataset_type == 'real_dataset':
        dataset_name = data.get('dataset_name')
        
        if dataset_name == 'iris':
            data_obj = load_iris()
            X, y = data_obj.data, data_obj.target
        elif dataset_name == 'california':  # california housing
            data_obj = fetch_california_housing()
            X, y = data_obj.data, data_obj.target
    
    # Prepare data
    prepared_data = slm.prepare_data(X, y)
    
    return jsonify({
        'success': True,
        'data': prepared_data,
        'dataset_info': {
            'total_samples': len(X),
            'features': X.shape[1],
            'training_samples': len(slm.X_train),
            'test_samples': len(slm.X_test)
        }
    })

@app.route('/api/train_model', methods=['POST'])
def train_model():
    data = request.json
    algorithm = data.get('algorithm')
    dataset_type = data.get('dataset_type', 'synthetic_classification')
    
    if algorithm == 'linear_regression':
        training_steps = slm.linear_regression_step_by_step()
    elif algorithm == 'logistic_regression':
        training_steps = slm.logistic_regression_step_by_step()
    elif algorithm == 'decision_tree':
        max_depth = data.get('max_depth', 5)
        # Determine task type based on dataset
        if dataset_type in ['synthetic_classification', 'real_dataset']:
            task = 'classification'
        else:
            task = 'regression'
        training_steps = slm.decision_tree_step_by_step(max_depth, task)
    elif algorithm == 'svm':
        # Determine task type based on dataset
        if dataset_type in ['synthetic_classification', 'real_dataset']:
            task = 'classification'
        else:
            task = 'regression'
        training_steps = slm.svm_step_by_step(task)
    elif algorithm == 'knn':
        n_neighbors = data.get('n_neighbors', 5)
        # Determine task type based on dataset
        if dataset_type in ['synthetic_classification', 'real_dataset']:
            task = 'classification'
        else:
            task = 'regression'
        training_steps = slm.knn_step_by_step(task, n_neighbors)
    else:
        return jsonify({'success': False, 'error': 'Algorithm not implemented'})
    
    return jsonify({
        'success': True,
        'training_steps': training_steps
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
