import numpy as np
import pandas as pd
import math
import os
from matplotlib import pyplot as plt


def normalize_vector(vector):
    return vector / np.linalg.norm(vector)


def compute_vectors_angular_deviation(v1, v2):
    v1_u = normalize_vector(v1)
    v2_u = normalize_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def load_and_process_data():
    work_dir = './data/task2'
    predictions_file_names = [file_name for file_name in os.listdir(work_dir) if (not ('por_gt' in file_name) and not('lost_frames' in file_name))]

    predictions_list = []

    for idx, prediction_file_name in enumerate(predictions_file_names):
        prediction_file_path = os.path.join(work_dir, prediction_file_name)
        prediction = pd.read_csv(prediction_file_path, header=None, sep=" ")
        predictions_list.append(prediction)
        if prediction_file_name != 'gaze_vectors.txt':
            prediction = prediction.iloc[:, 1:]
        prediction = np.array(prediction)
        predictions_list[idx] = prediction

    gt_file_path = os.path.join(work_dir, 'por_gt.txt')
    gt = pd.read_csv(gt_file_path, header=None, sep=" ")
    gt = gt.iloc[:, 1:]
    gt = np.array(gt)

    return predictions_list, gt


def compute_model_angular_deviations(predictions, gt):
    angular_deviations = np.empty((predictions.shape[0],))
    for idx, (pred_v, gt_v) in enumerate(zip(predictions, gt)):
        angular_deviations[idx] = compute_vectors_angular_deviation(pred_v, gt_v) * 180 / math.pi
    return angular_deviations


def compute_models_angular_deviations(predictions_list, gt):
    models_angular_deviations = []
    for predictions in predictions_list:
        models_angular_deviations.append(compute_model_angular_deviations(predictions, gt))
    return models_angular_deviations


def compute_models_errors_vs_ratio(models_angular_deviations):
    models_error_vs_ratio = []
    gaze_error_thresholds = range(45)
    for model_angular_deviations in models_angular_deviations:
        model_error_vs_ratio = []
        for gaze_error_threshold in gaze_error_thresholds:
            below_average_ratio = sum(angular_deviation < gaze_error_threshold for angular_deviation in model_angular_deviations) / 2000
            model_error_vs_ratio.append(below_average_ratio)
        models_error_vs_ratio.append(model_error_vs_ratio)
    return np.array(models_error_vs_ratio)


def plot_error_vs_ratio(models_errors_vs_ratio):
    for models_errors_vs_ratio in models_errors_vs_ratio:
        plt.plot(range(45), models_errors_vs_ratio)
    plt.savefig('error_vs_ratio.png')


def main():
    predictions_list, gt = load_and_process_data()
    models_angular_deviations = compute_models_angular_deviations(predictions_list, gt)
    models_errors_vs_ratio = compute_models_errors_vs_ratio(models_angular_deviations)
    plot_error_vs_ratio(models_errors_vs_ratio)


if __name__ == "__main__":
    main()
