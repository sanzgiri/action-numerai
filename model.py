import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression


class NumeraiModel:
    """
    A simple linear regression model for Numerai predictions.
    Uses feature columns to predict the target variable.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.model = LinearRegression()
        self.feature_cols = None
        self.target_col = "target"

    def fit(self, train_data):
        """
        Train the model on the provided training data.

        Args:
            train_data: pandas DataFrame with features and target column
        """
        # Get feature columns (all columns except id, target, and era)
        self.feature_cols = [
            col for col in train_data.columns
            if col not in ["id", "target", "era"] and train_data[col].dtype in [np.float32, np.float64]
        ]

        if self.verbose:
            print(f"Training with {len(self.feature_cols)} features")

        # Prepare training data
        X = train_data[self.feature_cols].values
        y = train_data[self.target_col].values

        # Train the model
        self.model.fit(X, y)

        if self.verbose:
            print("Model training completed")

    def predict(self, live_data):
        """
        Generate predictions for the live tournament data.

        Args:
            live_data: pandas DataFrame with feature columns

        Returns:
            numpy array of predictions
        """
        if self.feature_cols is None:
            raise ValueError("Model must be trained before making predictions")

        # Prepare prediction data
        X = live_data[self.feature_cols].values

        # Generate predictions
        predictions = self.model.predict(X)

        # Clip predictions to valid range [0, 1]
        predictions = np.clip(predictions, 0, 1)

        if self.verbose:
            print(f"Generated {len(predictions)} predictions")

        return predictions

    def save(self, filename):
        """Save the model to a file."""
        joblib.dump(self, filename)
        if self.verbose:
            print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename):
        """Load a model from a file."""
        return joblib.load(filename)
