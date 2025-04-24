import sys
sys.setrecursionlimit(10000)
from typing import Tuple, List, Any

import csv
import pickle
import copy

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, BatchNormalization

def _initialize_model():
    model = Sequential(name="my_model")

    # Note: Input shape must be redefined (100 here, based on previous JSON model structure).
    model.add(Input(shape=(100,), name="input_layer"))

    # Layer 1: Dense with 1024 units and linear activation, followed by relu Activation, Dropout, and BatchNormalization
    model.add(Dense(1024, activation="linear", name="dense_198"))
    model.add(Activation("relu", name="activation_198"))
    model.add(Dropout(0.3, name="dropout_164"))
    model.add(BatchNormalization(name="batch_normalization_164"))

    # Layer 2
    model.add(Dense(1024, activation="linear", name="dense_199"))
    model.add(Activation("relu", name="activation_199"))
    model.add(Dropout(0.3, name="dropout_165"))
    model.add(BatchNormalization(name="batch_normalization_165"))

    # Layer 3
    model.add(Dense(1024, activation="linear", name="dense_200"))
    model.add(Activation("relu", name="activation_200"))
    model.add(Dropout(0.3, name="dropout_166"))
    model.add(BatchNormalization(name="batch_normalization_166"))

    # Layer 4
    model.add(Dense(1024, activation="linear", name="dense_201"))
    model.add(Activation("relu", name="activation_201"))
    model.add(Dropout(0.3, name="dropout_167"))
    model.add(BatchNormalization(name="batch_normalization_167"))

    # Output Layer: 113 units with linear activation, followed by sigmoid
    model.add(Dense(113, activation="linear", name="dense_202"))
    model.add(Activation("sigmoid", name="activation_202"))

    model.summary()

    # Load model weights from .h5 file (ensure "ddi_model.h5" is in the correct directory)
    model.load_weights("data/models/ddi_model.h5")
    return model


def _predict_with_mc(model: Any,
                     X: np.ndarray,
                     iter_num: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform predictions using Monte Carlo method and return the mean
    and standard deviation of the predictions.

    Parameters:
        model: Keras model used for prediction.
        X: Input feature matrix.
        iter_num: Number of Monte Carlo iterations.

    Returns:
        A tuple containing:
            - Mean of predictions.
            - Standard deviation of predictions.
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
    Write prediction results to an output CSV file.

    Parameters:
        output_file: Path to the output CSV file.
        ddi_pairs: List of drug-drug pairs.
        predicted_results: List of predicted label indices for each pair.
        original_predicted: Raw prediction scores (before applying thresholds).
        original_std: Standard deviations of the predictions.
    """
    header = ["Drug pair", "Predicted class", "Score", "STD"]
    with open(output_file, mode="w", newline="") as csvfile:
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
                binarizer_file: str,
                threshold: float) -> None:
    """
    Predict drug-drug interactions (DDI) and write the results to an output file.

    Parameters:
        output_file: Path to the output file where predictions will be saved.
        pca_df: DataFrame containing input features (index should be drug pairs).
        binarizer_file: Path to the PKL file containing the label binarizer.
        threshold: Threshold used to classify the prediction scores.
    """
    # Load label binarizer
    with open(binarizer_file, "rb") as fid:
        label_binarizer = pickle.load(fid)

    ddi_pairs = list(pca_df.index)
    X = pca_df.values

    # Load model and weights
    model = _initialize_model()

    # Perform Monte Carlo predictions
    mean_preds, std_preds = _predict_with_mc(model, X, iter_num=10)

    # Deep copy the original scores and stds for file output
    original_predicted = copy.deepcopy(mean_preds)
    original_std = copy.deepcopy(std_preds)

    # Apply threshold to generate binary predictions
    mean_preds = np.where(mean_preds >= threshold, 1, 0)

    # Convert binary predictions back to label format
    predicted_labels = label_binarizer.inverse_transform(mean_preds)

    _write_prediction_results(output_file, ddi_pairs, predicted_labels, original_predicted, original_std)