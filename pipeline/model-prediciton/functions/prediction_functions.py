from joblib import load
import dill as pickle
import pandas as pd

def load_models(project_root):
    """
    Load the LabelEncoder and Pipeline objects from files.
    
    Args:
        project_root (Path): The root path of the project.
        
    Returns:
        label_encoder (LabelEncoder): The loaded LabelEncoder.
        pipeline (Pipeline): The loaded Pipeline.
    """

    # Load the LabelEncoder
    label_encoder = load(project_root / "models" / 'label_encoder.pkl')
    
    # Load the pipeline
    with open(project_root / "models" / 'footy_tipper.pkl', 'rb') as f:
        pipeline = pickle.load(f)
        
    return label_encoder, pipeline

def model_predictions(pipeline, inference_data, label_encoder):
    """
    Make predictions using the trained model.

    Args:
        tuned_model (Pipeline): The trained model.
        inference_data (DataFrame): The data to make predictions on.
        label_encoder (LabelEncoder): The LabelEncoder object.

    Returns:
        results (DataFrame): The predictions and probability estimates.
    """
    
    # Make predictions
    encoded_predictions = pipeline.predict(inference_data)

    # Get probability estimates
    probability_estimates = pipeline.predict_proba(inference_data)

    # Reverse transform the predictions to get the original labels
    predictions = label_encoder.inverse_transform(encoded_predictions)

    # Put everything into a DataFrame
    results = pd.DataFrame({
        'game_id': inference_data["game_id"],
        'home_team_result': predictions,
        'home_team_win_prob': probability_estimates[:, 1],
        'home_team_lose_prob': probability_estimates[:, 0]
    })
    
    # Return the DataFrame
    return results
