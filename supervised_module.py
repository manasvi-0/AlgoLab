# Import essential libraries which will be needed ...

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Creating and running and visualizing supervised ML models...

class SupervisedML:
    # linear regression function..
    def linear_regression(self, X, y, test_size=0.2, random_state=42):
        """" 
        Args---- 
            X (np.array): Features 
            y (np.array): Target 
            test_size (float): Split ratio for test data 
            random_state (int): Seed for reproducibility

        Returns: model, X_train, X_test, y_train, y_test 
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        model = LinearRegression()
        model.fit(X_train, y_train)

        return model, X_train, X_test, y_train, y_test
    

     # Plot actual vs. predicted values using the trained model...
    def plot_linear_regression(self, model, X_test, y_test, feature_name='Feature'):
        """ 
        Args---- 
            model: Trained LinearRegression model 
            X_test (np.array): Test features 
            y_test (np.array): Actual test targets 
            feature_name (str): Label for x-axis 
        """
        y_pred = model.predict(X_test)

        plt.figure(figsize=(10, 6))
        plt.scatter(X_test, y_test, color='blue', label='Actual', alpha=0.7)
        plt.plot(X_test, y_pred, color='red', linewidth=2, label='Prediction')

        # Showing errors with dashed lines...
        for i in range(len(X_test)):
            plt.plot([X_test[i], X_test[i]], [y_test[i], y_pred[i]], color='gray', linestyle='--', linewidth=0.5)

        plt.title('Linear Regression: Actual vs. Predicted')
        plt.xlabel(feature_name)
        plt.ylabel('Target')
        plt.legend()
        plt.show()

        # Print performance metrics...
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print("<--- Model Performance --->")
        print(f"MSE: {mse:.2f}")
        print(f"R-squared: {r2:.2f}")

     # Plot training vs testing data...
    def plot_train_test_split(self, X_train, X_test, y_train, y_test, feature_name='Feature'):
        """ 
        Args---- 
            X_train (np.array): Train features 
            X_test (np.array): Test features 
            y_train (np.array): Train targets 
            y_test (np.array): Test targets 
            feature_name (str): Label for x-axis 
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(X_train, y_train, color='blue', label='Train', alpha=0.7)
        plt.scatter(X_test, y_test, color='green', label='Test', alpha=0.7)
        plt.title('Train vs. Test Split')
        plt.xlabel(feature_name)
        plt.ylabel('Target')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    print("<--- Running Linear Regression Example --->")

    # Generating simple sample data..
    np.random.seed(0)
    X_linear = 2 * np.random.rand(100, 1)
    y_linear = 4 + 3 * X_linear + np.random.randn(100, 1)

    # Initialize our ML class.
    ml_module = SupervisedML()

    # Training the model..
    linear_model, X_train_lin, X_test_lin, y_train_lin, y_test_lin = ml_module.linear_regression(X_linear, y_linear)

    # Visualizing the data split ...
    ml_module.plot_train_test_split(X_train_lin, X_test_lin, y_train_lin, y_test_lin, feature_name='X')

    # Ploting model predictions ..
    ml_module.plot_linear_regression(linear_model, X_test_lin, y_test_lin, feature_name='X')
