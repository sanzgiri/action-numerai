import os
import pandas as pd
from numerapi import NumerAPI
from model import NumeraiModel


def predict():
    """
    Download current Numerai dataset, generate predictions, and submit them.
    """
    # Initialize Numerai API
    public_id = os.environ.get("NUMERAI_PUBLIC_ID")
    secret_key = os.environ.get("NUMERAI_SECRET_KEY")

    if not public_id or not secret_key:
        raise ValueError("NUMERAI_PUBLIC_ID and NUMERAI_SECRET_KEY must be set as environment variables")

    napi = NumerAPI(public_id=public_id, secret_key=secret_key)

    # Get current round information
    current_round = napi.get_current_round()
    print(f"Current round: {current_round}")

    # Download the current dataset
    print("Downloading dataset...")
    napi.download_dataset("v5.2/live.parquet")
    print("Dataset downloaded")

    # Load the live data
    print("Loading live data...")
    live_data = pd.read_parquet("v5.2/live.parquet")
    print(f"Live data shape: {live_data.shape}")

    # Initialize or load model
    model_path = "trained_model.joblib"
    if os.path.exists(model_path):
        print(f"Loading saved model from {model_path}")
        model = NumeraiModel.load(model_path)
    else:
        print("No saved model found. Training new model...")
        # Download training data
        napi.download_dataset("v5.2/train.parquet")
        train_data = pd.read_parquet("v5.2/train.parquet")

        # Optional: Limit training rows to avoid timeout (set TRAIN_SAMPLE_SIZE env var)
        train_sample_size = os.environ.get("TRAIN_SAMPLE_SIZE")
        if train_sample_size:
            sample_size = int(train_sample_size)
            print(f"Using sample of {sample_size} rows for training (full dataset: {len(train_data)} rows)")
            train_data = train_data.sample(n=min(sample_size, len(train_data)), random_state=42)

        # Train model
        model = NumeraiModel(verbose=True)
        model.fit(train_data)

        # Save model for future use
        model.save(model_path)
        print(f"Model saved to {model_path}")

    # Generate predictions
    print("Generating predictions...")
    predictions = model.predict(live_data)

    # Create submission dataframe using index as id
    # In v5.2, the id is in the dataframe index, not as a column
    submission = pd.Series(predictions, index=live_data.index).to_frame("prediction")

    # Save predictions to file (index will be saved as id column)
    prediction_file = f"predictions_round_{current_round}.csv"
    submission.to_csv(prediction_file)
    print(f"Predictions saved to {prediction_file}")

    # Get model ID for submission
    models = napi.get_models()
    print(f"Available models: {models}")

    # Submit predictions for the first model (you may want to customize this)
    if models:
        model_id = list(models.keys())[0]
        print(f"Submitting predictions for model: {model_id}")

        submission_id = napi.upload_predictions(prediction_file, model_id=model_id)
        print(f"Submission successful! Submission ID: {submission_id}")
    else:
        print("No models found. Please create a model at https://numer.ai/models")
        print(f"Predictions saved to {prediction_file} for manual upload")


if __name__ == "__main__":
    predict()
