import os
import pandas as pd
from numerapi import NumerAPI
from model import NumeraiModel


def train():
    """
    Download Numerai training data and train a model.
    """
    # Initialize Numerai API
    public_id = os.environ.get("NUMERAI_PUBLIC_ID")
    secret_key = os.environ.get("NUMERAI_SECRET_KEY")

    napi = NumerAPI(public_id=public_id, secret_key=secret_key)

    # Download training dataset
    print("Downloading training dataset...")
    napi.download_dataset("v4.3/train.parquet")
    print("Dataset downloaded")

    # Load training data
    print("Loading training data...")
    train_data = pd.read_parquet("v4.3/train.parquet")
    print(f"Training data shape: {train_data.shape}")

    # Create and train model
    print("Training model...")
    model = NumeraiModel(verbose=True)
    model.fit(train_data)

    # Save the trained model
    model_path = "trained_model.joblib"
    model.save(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train()
