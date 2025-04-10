import sys
sys.setrecursionlimit(10000)
from typing import Tuple, List, Any

import csv
import pickle
import copy

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, BatchNormalization
#tensorflow==2.19

def _initialize_model():
    # Xây dựng cấu trúc mô hình mới sử dụng Sequential.
    model = Sequential(name="my_model")

    # Lưu ý: Bạn cần xác định lại kích thước input (ở đây là 100, dựa theo thông tin có trong file .json cũ)
    model.add(Input(shape=(100,), name="input_layer"))

    # Lớp 1: Dense với 1024 đơn vị và linear activation, sau đó áp dụng Activation relu, Dropout và BatchNormalization
    model.add(Dense(1024, activation="linear", name="dense_198"))
    model.add(Activation("relu", name="activation_198"))
    model.add(Dropout(0.3, name="dropout_164"))
    model.add(BatchNormalization(name="batch_normalization_164"))

    # Lớp 2: Dense với 1024 đơn vị và linear activation, sau đó Activation relu, Dropout và BatchNormalization
    model.add(Dense(1024, activation="linear", name="dense_199"))
    model.add(Activation("relu", name="activation_199"))
    model.add(Dropout(0.3, name="dropout_165"))
    model.add(BatchNormalization(name="batch_normalization_165"))

    # Lớp 3: Dense với 1024 đơn vị và linear activation, sau đó Activation relu, Dropout và BatchNormalization
    model.add(Dense(1024, activation="linear", name="dense_200"))
    model.add(Activation("relu", name="activation_200"))
    model.add(Dropout(0.3, name="dropout_166"))
    model.add(BatchNormalization(name="batch_normalization_166"))

    # Lớp 4: Dense với 1024 đơn vị và linear activation, sau đó Activation relu, Dropout và BatchNormalization
    model.add(Dense(1024, activation="linear", name="dense_201"))
    model.add(Activation("relu", name="activation_201"))
    model.add(Dropout(0.3, name="dropout_167"))
    model.add(BatchNormalization(name="batch_normalization_167"))

    # Lớp cuối: Dense với 113 đơn vị và linear activation, sau đó Activation sigmoid
    model.add(Dense(113, activation="linear", name="dense_202"))
    model.add(Activation("sigmoid", name="activation_202"))

    # Tóm tắt mô hình nếu cần
    model.summary()

    # Sau khi đã xây dựng cấu trúc mới, nạp trọng số từ file .h5 (đảm bảo file "ddi_model.h5" nằm trong cùng thư mục)
    model.load_weights("data/models/ddi_model.h5")
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
    model = _initialize_model()

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

