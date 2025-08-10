"""
Demo script for the Supervised Learning Module
==============================================

This script demonstrates how to use the supervised learning module
with real-time visualizations for various algorithms.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression, load_iris, fetch_california_housing
try:
    from supervised_module import SupervisedLearningModule
    MODULE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import supervised_module: {e}")
    print("This demo requires the supervised_module.py to be fully implemented.")
    MODULE_AVAILABLE = False
import matplotlib.pyplot as plt

def demo_classification():
    """Demo for classification algorithms."""
    print("=" * 60)
    print("CLASSIFICATION DEMO")
    print("=" * 60)
    
    # Create synthetic classification dataset
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                             n_informative=2, n_clusters_per_class=1, 
                             random_state=42)
    
    # Initialize the module
    slm = SupervisedLearningModule()
    
    # Prepare data
    slm.prepare_data(X, y, feature_names=['Feature 1', 'Feature 2'], 
                    target_name='Class')
    
    # Train different classification models
    print("\n1. Training Logistic Regression...")
    slm.logistic_regression(real_time_viz=True)
    
    print("\n2. Training Decision Tree...")
    slm.decision_tree(max_depth=5, real_time_viz=True, task='classification')
    
    print("\n3. Training SVM...")
    slm.support_vector_machine(kernel='rbf', real_time_viz=True, task='classification')
    
    print("\n4. Training KNN...")
    slm.k_nearest_neighbors(n_neighbors=5, real_time_viz=True, task='classification')
    
    # Compare all models
    print("\n5. Comparing all models...")
    slm.compare_models()

def demo_regression():
    """Demo for regression algorithms."""
    print("=" * 60)
    print("REGRESSION DEMO")
    print("=" * 60)
    
    # Create synthetic regression dataset
    X, y = make_regression(n_samples=1000, n_features=2, n_informative=2, 
                          noise=0.1, random_state=42)
    
    # Initialize the module
    slm = SupervisedLearningModule()
    
    # Prepare data
    slm.prepare_data(X, y, feature_names=['Feature 1', 'Feature 2'], 
                    target_name='Target')
    
    # Train different regression models
    print("\n1. Training Linear Regression...")
    slm.linear_regression(real_time_viz=True)
    
    print("\n2. Training Decision Tree...")
    slm.decision_tree(max_depth=5, real_time_viz=True, task='regression')
    
    print("\n3. Training SVM...")
    slm.support_vector_machine(kernel='rbf', real_time_viz=True, task='regression')
    
    print("\n4. Training KNN...")
    slm.k_nearest_neighbors(n_neighbors=5, real_time_viz=True, task='regression')
    
    # Compare all models
    print("\n5. Comparing all models...")
    slm.compare_models()

def demo_real_dataset():
    """Demo with real datasets."""
    print("=" * 60)
    print("REAL DATASET DEMO")
    print("=" * 60)
    
    # Load Iris dataset for classification
    print("\nIris Dataset Classification:")
    iris = load_iris()
    X_iris = iris.data[:, :2]  # Use only first 2 features for visualization
    y_iris = iris.target
    
    slm_iris = SupervisedLearningModule()
    slm_iris.prepare_data(X_iris, y_iris, 
                         feature_names=iris.feature_names[:2], 
                         target_name='Species')
    
    print("Training Decision Tree on Iris dataset...")
    slm_iris.decision_tree(max_depth=4, real_time_viz=True, task='classification')
    
    # Load California Housing dataset for regression
    print("\nCalifornia Housing Dataset Regression:")
    california = fetch_california_housing()
    X_california = california.data[:, :2]  # Use only first 2 features
    y_california = california.target
    
    slm_california = SupervisedLearningModule()
    slm_california.prepare_data(X_california, y_california, 
                               feature_names=california.feature_names[:2], 
                               target_name='Price')
    
    print("Training Linear Regression on California Housing dataset...")
    slm_california.linear_regression(real_time_viz=True)

def demo_custom_dataset():
    """Demo with custom dataset upload functionality."""
    print("=" * 60)
    print("CUSTOM DATASET DEMO")
    print("=" * 60)
    
    # Simulate custom dataset upload
    print("Simulating dataset upload...")
    
    # Create a more complex dataset
    np.random.seed(42)
    n_samples = 1500
    
    # Create features with some correlation
    feature1 = np.random.normal(0, 1, n_samples)
    feature2 = 0.7 * feature1 + np.random.normal(0, 0.5, n_samples)
    feature3 = np.random.normal(0, 1, n_samples)
    
    X_custom = np.column_stack([feature1, feature2, feature3])
    
    # Create target with non-linear relationship
    y_custom = 2 * feature1**2 + 3 * feature2 + 0.5 * feature3 + np.random.normal(0, 0.1, n_samples)
    
    # Initialize module
    slm_custom = SupervisedLearningModule()
    
    # Prepare data
    slm_custom.prepare_data(X_custom, y_custom, 
                           feature_names=['Feature 1', 'Feature 2', 'Feature 3'], 
                           target_name='Target')
    
    # Train models with real-time visualization
    print("\nTraining models on custom dataset...")
    
    print("1. Linear Regression...")
    slm_custom.linear_regression(real_time_viz=True)
    
    print("2. Decision Tree...")
    slm_custom.decision_tree(max_depth=6, real_time_viz=True, task='regression')
    
    print("3. SVM...")
    slm_custom.support_vector_machine(kernel='rbf', real_time_viz=True, task='regression')
    
    print("4. KNN...")
    slm_custom.k_nearest_neighbors(n_neighbors=7, real_time_viz=True, task='regression')
    
    # Compare models
    print("\n5. Model comparison...")
    slm_custom.compare_models()

def main():
    """Main demo function."""
    print("SUPERVISED LEARNING MODULE DEMO")
    print("=" * 60)
    print("This demo showcases the supervised learning algorithms with")
    print("real-time visualizations for better understanding.")
    print("=" * 60)
    
    if not MODULE_AVAILABLE:
        print("\n❌ Cannot run demo: supervised_module is not available.")
        print("Please ensure supervised_module.py is properly implemented.")
        return
    
    # Run different demos
    demo_classification()
    demo_regression()
    demo_real_dataset()
    demo_custom_dataset()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED!")
    print("=" * 60)
    print("The supervised learning module provides:")
    print("✓ Real-time training visualizations")
    print("✓ Step-by-step model training progress")
    print("✓ Decision boundary plots")
    print("✓ Performance metrics comparison")
    print("✓ Support for both classification and regression")
    print("✓ Easy integration with custom datasets")

if __name__ == "__main__":
    main()
