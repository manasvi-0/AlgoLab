import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


#to import from csv file
def load_csv(path):
    df = pd.read_csv(path)
    print(df.head())
    print(df.describe())
    print(f"Missing values:\n{df.isnull().sum()}")
    return df

#normalization
class LinearRegressionSupervisedML():

    def normalization(self,X):
        """
        ARG:
        :param X: X is the input data
        :return: normalized X
        """
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)
        return X_normalized


    def LinearRegression(self,X,y,test=0.2):
        """
        Arguments:
        :param X(ndarray(m,)): input data
        :param y: output data
        :param test: size of test set
        :return: model, x_train, x_test,y_train, y_test
        """
        X_normalized = self.normalization(X)
        X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=test, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model, X_train, X_test, y_train, y_test

    def PlotlinearRegression(self, model, X_test, y_test, feature_name='Feature'):
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
        #print report
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R-squared Score: {r2:.4f}")

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

class LogisticRegressionSupervisedML():
    def normalization(self,X):
        """
        ARG:
        :param X: X is the input data
        :return: normalized X
        """
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)
        return X_normalized

    def LogisticRegression(self,X,y,test=0.2):
        """
        Arguments:
        :param X(ndarray(m,)): input data
        :param y: output data
        :param test: size of test set
        :return: model, x_train, x_test,y_train, y_test
        """
        X_normalized = self.normalization(X)
        X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=test, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train.ravel())
        return model, X_train, X_test, y_train, y_test

    def PlotlogisticRegression(self, model, X_test, y_test, feature_name='Feature'):
        """
        Args----
            model: Trained LinearRegression model
            X_test (np.array): Test features
            y_test (np.array): Actual test targets
            feature_name (str): Label for x-axis
        """
        y_test = y_test.ravel()  # Ensure it's 1D
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Sort for smoother logistic curve
        sorted_idx = X_test.flatten().argsort()
        X_sorted = X_test.flatten()[sorted_idx]
        y_prob_sorted = y_prob[sorted_idx]

        plt.figure(figsize=(10, 6))
        plt.scatter(X_test, y_test, color='blue', label='Actual Labels', alpha=0.6)
        plt.plot(X_sorted, y_prob_sorted, color='red', label='Logistic Curve', linewidth=2)

        for i in range(len(X_test)):
            x_val = float(X_test[i][0])  # or np.squeeze(X_test[i])
            plt.plot([x_val, x_val], [y_test[i], y_prob[i]], color="gray", linestyle='--', linewidth=0.5)

        plt.title('Logistic Regression: Actual vs. Predicted Probability')
        plt.xlabel(feature_name)
        plt.ylabel('Probability / Label')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Print classification report
        report = classification_report(y_test, y_pred)
        print(report)

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
    X_linear = 4 * np.random.rand(100, 1)
    y_linear = 5 + 2 * X_linear + np.random.randn(100, 1)

    # Initialize our ML class.
    ml_module = LinearRegressionSupervisedML()


    # Training the model..
    linear_model, X_train_lin, X_test_lin, y_train_lin, y_test_lin = ml_module.LinearRegression(X_linear, y_linear)

    # Visualizing the data split ...
    ml_module.plot_train_test_split(X_train_lin, X_test_lin, y_train_lin, y_test_lin, feature_name='X')

    # Ploting model predictions ..
    ml_module.PlotlinearRegression(linear_model, X_test_lin, y_test_lin, feature_name='X')

if __name__ == '__main__':
    print("<--- Running Logistic Regression Example --->")

    # Generating simple sample data..
    np.random.seed(0)
    X_linear = 4 * np.random.rand(100, 1)
    y_bin = 5 + 2 * X_linear + np.random.randn(100, 1)
    y_linear = (y_bin > y_bin.mean()).astype(int)
    # Initialize our ML class.
    ml_module = LogisticRegressionSupervisedML()


    # Training the model..
    linear_model, X_train_lin, X_test_lin, y_train_lin, y_test_lin = ml_module.LogisticRegression(X_linear, y_linear)

    # Visualizing the data split ...
    ml_module.plot_train_test_split(X_train_lin, X_test_lin, y_train_lin, y_test_lin, feature_name='X')

    # Ploting model predictions ..
    ml_module.PlotlogisticRegression(linear_model, X_test_lin, y_test_lin, feature_name='X')


