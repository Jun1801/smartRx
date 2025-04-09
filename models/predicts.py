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
    Tải kiến trúc và trọng số của mô hình từ các file.

    Tham số:
        trained_model_path: Đường dẫn tới file JSON chứa kiến trúc mô hình.
        trained_weight_path: Đường dẫn tới file H5 chứa trọng số của mô hình.
    
    Trả về:
        Mô hình Keras đã được tải.
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
    Thực hiện dự đoán theo phương pháp Monte Carlo và trả về giá trị trung bình 
    và độ lệch chuẩn của các dự đoán.

    Tham số:
        model: Mô hình Keras dùng để dự đoán.
        X: Ma trận đặc trưng đầu vào.
        iter_num: Số lần lặp Monte Carlo.

    Trả về:
        Một tuple gồm:
            Trung bình các dự đoán.
            Độ lệch chuẩn của các dự đoán.
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
    Công dụng:
        Ghi kết quả dự đoán vào file CSV đầu ra.

    Tham số:
        output_file: Đường dẫn đến file đầu ra.
        ddi_pairs: Danh sách các cặp thuốc.
        predicted_results: Danh sách các nhãn dự đoán cho từng cặp thuốc.
        original_predicted: Các điểm số dự đoán gốc (trước khi áp dụng ngưỡng).
        original_std: Độ lệch chuẩn của các dự đoán.
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
    Công dụng:
        Dự đoán tương tác thuốc (drug-drug interactions - DDI) và ghi kết quả vào file đầu ra.

    Tham số:
        output_file: Đường dẫn tới file đầu ra chứa kết quả dự đoán.
        pca_df: DataFrame chứa đặc trưng đầu vào (index là cặp thuốc).
        trained_model: Đường dẫn tới file JSON chứa kiến trúc mô hình đã huấn luyện.
        trained_weight: Đường dẫn tới file H5 chứa trọng số của mô hình.
        binarizer_file: Đường dẫn tới file PKL chứa đối tượng label binarizer.
        threshold: Ngưỡng dùng để phân loại kết quả dự đoán.
    """
    # Tải label binarizer
    with open(binarizer_file, 'rb') as fid:
        label_binarizer = pickle.load(fid)

    ddi_pairs = list(pca_df.index)
    X = pca_df.values

    # Tải mô hình và trọng số
    model = _load_model(trained_model, trained_weight)

    # Thực hiện dự đoán Monte Carlo
    mean_preds, std_preds = _predict_with_mc(model, X, iter_num=10)
    # Sao chép sâu các điểm số và độ lệch chuẩn ban đầu để ghi ra file
    original_predicted = copy.deepcopy(mean_preds)
    original_std = copy.deepcopy(std_preds)

    # Áp dụng ngưỡng để phân loại dự đoán
    mean_preds = np.where(mean_preds >= threshold, 1, 0)
    # Lấy các nhãn dự đoán ban đầu thông qua inverse transformation
    predicted_labels = label_binarizer.inverse_transform(mean_preds)

    _write_prediction_results(output_file, ddi_pairs, predicted_labels, original_predicted, original_std)


def predict_severity(output_file: str,
                     pca_df: pd.DataFrame,
                     trained_model: str,
                     trained_weight: str,
                     binarizer_file: str,
                     threshold: float) -> None:
    """
    Công dụng:
        Dự đoán mức độ nghiêm trọng của tương tác thuốc và ghi kết quả vào file đầu ra.

    Tham số:
        output_file: Đường dẫn tới file đầu ra chứa kết quả dự đoán.
        pca_df: DataFrame chứa đặc trưng đầu vào (index là cặp thuốc).
        trained_model: Đường dẫn tới file JSON chứa kiến trúc mô hình đã huấn luyện.
        trained_weight: Đường dẫn tới file H5 chứa trọng số của mô hình.
        binarizer_file: Đường dẫn tới file PKL chứa đối tượng label binarizer.
        threshold: Ngưỡng dùng để phân loại kết quả dự đoán.
    
    Trả về:
        Không trả về; kết quả dự đoán được ghi vào file đầu ra.
    """
    # Tải label binarizer
    with open(binarizer_file, 'rb') as fid:
        label_binarizer = pickle.load(fid)

    ddi_pairs = list(pca_df.index)
    X = pca_df.values

    # Tải mô hình và trọng số
    model = _load_model(trained_model, trained_weight)

    # Thực hiện dự đoán Monte Carlo
    mean_preds, std_preds = _predict_with_mc(model, X, iter_num=10)
    original_predicted = copy.deepcopy(mean_preds)
    original_std = copy.deepcopy(std_preds)

    # Áp dụng ngưỡng để phân loại dự đoán
    mean_preds = np.where(mean_preds >= threshold, 1, 0)
    # Lấy các nhãn dự đoán ban đầu thông qua inverse transformation
    predicted_labels = label_binarizer.inverse_transform(mean_preds)

    _write_prediction_results(output_file, ddi_pairs, predicted_labels, original_predicted, original_std)


if __name__ == "__main__":
    # Ví dụ sử dụng: tạo một DataFrame mẫu với dữ liệu giả định
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
