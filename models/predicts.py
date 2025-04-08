from typing import Tuple, List, Any

import csv
import pickle
import copy

import numpy as np
import pandas as pd
from tensorflow.keras.models import model_from_json


def _load_model(trained_model_path: str, 
                trained_weight_path: str) -> Any:
    """
    Load the model architecture and weights from files

    Parameters:
        trained_model_path: File path for the JSON model architecture
        trained_weight_path: File path for the model weights
    
    Returns:
        The loaded Keras model
    """
    with open(trained_model_path, "r") as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)
    model.load_weights(trained_weight_path)
    return model


def _predict_with_mc(model: Any, 
                     X: np.ndarray, 
                     iter_num: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Monte Carlo predictions and return the mean and standard deviation
    
    Parameters:
        model: The Keras model used for prediction
        X: Input features as a NumPy array
        iter_num: The number of iterations for Monte Carlo sampling
    
    Returns:
        A tuple containing the mean predictions and standard deviations
    """
    mc_predictions = [model.predict(X) for _ in range(iter_num)]
    predictions_array = np.asarray(mc_predictions)
    mean_predictions = np.mean(predictions_array, axis=0)
    std_predictions = np.std(predictions_array, axis=0)
    return mean_predictions, std_predictions


def _write_prediction_results(output_file: str,
                              ddi_pairs: List[str], 
                              predicted_results: List[List[int]],
                              original_predicted: np.ndarray,
                              original_std: np.ndarray) -> None:
    """
    Write the prediction results to the output file using f-string formatting

    Parameters:
        output_file: The path to the output file
        ddi_pairs: List of drug pair identifiers
        predicted_results: List of predicted labels for each drug pair
        original_predicted: The original predicted scores (before thresholding)
        original_std: The original prediction standard deviations
    """
    header = ['Drug pair', 'Predicted class', 'Score', 'STD']
    with open(output_file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for i, pair in enumerate(ddi_pairs):
            predicted_ddi = predicted_results[i]
            scores = original_predicted[i]
            stds = original_std[i]
            for predicted_class in predicted_ddi:
                writer.writerow([pair,
                                 predicted_class,
                                 scores[predicted_class - 1],
                                 stds[predicted_class - 1]])


def predict_DDI(output_file: str,
                pca_df: pd.DataFrame,
                trained_model: str,
                trained_weight: str,
                binarizer_file: str,
                threshold: float) -> None:
    """
    Predict drug-drug interactions (DDI) and write the results to the output file

    Parameters:
        output_file: Path to the output file for predictions
        pca_df: DataFrame containing input features. The index should represent drug pairs
        trained_model: File path to the JSON file containing the trained model architecture
        trained_weight: File path to the file containing model weights
        binarizer_file: File path to the pickle file with the label binarizer
        threshold: The threshold used for classifying the predictions
    """
    # Load the label binarizer
    with open(binarizer_file, 'rb') as fid:
        label_binarizer = pickle.load(fid)

    ddi_pairs = list(pca_df.index)
    X = pca_df.values

    # Load the model and its weights
    model = _load_model(trained_model, trained_weight)

    # Perform Monte Carlo predictions
    mean_preds, std_preds = _predict_with_mc(model, X, iter_num=10)
    # Deep copy the original scores and standard deviations for output
    original_predicted = copy.deepcopy(mean_preds)
    original_std = copy.deepcopy(std_preds)

    # Apply the threshold to classify predictions
    mean_preds = np.where(mean_preds >= threshold, 1, 0)
    # Get the original labels by performing inverse transformation
    predicted_labels = label_binarizer.inverse_transform(mean_preds)

    _write_prediction_results(output_file, ddi_pairs, predicted_labels, original_predicted, original_std)


def predict_severity(output_file: str,
                     pca_df: pd.DataFrame,
                     trained_model: str,
                     trained_weight: str,
                     binarizer_file: str,
                     threshold: float) -> None:
    """
    Predict the severity of drug-drug interactions and write the results to the output file

    Parameters:
        output_file: Path to the output file for predictions
        pca_df: DataFrame containing input features. The index should represent drug pairs
        trained_model: File path to the JSON file containing the trained model architecture
        trained_weight: File path to the file containing model weights
        binarizer_file: File path to the pickle file with the label binarizer
        threshold: The threshold used for classifying the predictions
    """
    # Load the label binarizer
    with open(binarizer_file, 'rb') as fid:
        label_binarizer = pickle.load(fid)

    ddi_pairs = list(pca_df.index)
    X = pca_df.values

    # Load the model and its weights
    model = _load_model(trained_model, trained_weight)

    # Perform Monte Carlo predictions
    mean_preds, std_preds = _predict_with_mc(model, X, iter_num=10)
    original_predicted = copy.deepcopy(mean_preds)
    original_std = copy.deepcopy(std_preds)

    # Apply the threshold to classify predictions
    mean_preds = np.where(mean_preds >= threshold, 1, 0)
    # Get the original labels by performing inverse transformation
    predicted_labels = label_binarizer.inverse_transform(mean_preds)

    _write_prediction_results(output_file, ddi_pairs, predicted_labels, original_predicted, original_std)


if __name__ == "__main__":
    # Example usage: create a dummy DataFrame with sample data
    df = pd.DataFrame({
        'feature1': [0.1, 0.2],
        'feature2': [0.3, 0.4]
    }, index=["DrugA-DrugB", "DrugC-DrugD"])

    output_file_ddi = "ddi_predictions.csv"
    output_file_severity = "severity_predictions.csv"
    trained_model_file = "ddi_model.json"
    trained_weight_file = "ddi_model.h5"
    binarizer_file = "labels_binarizer.pkl"
    threshold_value = 0.5

    # predict_DDI(output_file_ddi, df, trained_model_file, trained_weight_file, binarizer_file, threshold_value)
    # predict_severity(output_file_severity, df, trained_model_file, trained_weight_file, binarizer_file, threshold_value)
