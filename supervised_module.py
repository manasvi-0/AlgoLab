# Import essential libraries which will be needed ...

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Creating and running and visualizing supervised ML models...

class SupervisedML:
    # linear regression function.
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


