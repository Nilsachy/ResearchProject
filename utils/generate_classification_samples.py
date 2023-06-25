
import numpy as np

from data_exploration.find_non_and_overlapping_segments import find_non_overlapping_segments_for_pid
from data_loading.load_annotations import load_realized_annotations, load_unrealized_annotations
from utils.AccelExtractor import AccelExtractor
from utils.generate_negative_intentions_intervals import generate_negative_intentions_intervals


def generate_classification_samples(pids, segment_length):
    X_windows = []
    X_segments = []
    y = []
    accel_ds_path = "../data/accel/subj_accel_interp.pkl"
    extractor = AccelExtractor(accel_ds_path)
    # Loop over every person id
    for pid in pids:
        # Load the annotated data for realized intentions
        time_list_of_realized_intentions = load_realized_annotations(pid, segment_length)
        time_list_of_unrealized_intentions = load_unrealized_annotations(pid, segment_length)
        time_list = time_list_of_realized_intentions + time_list_of_unrealized_intentions
        time_list_of_negative_samples = generate_negative_intentions_intervals(time_list, segment_length)
        concatenated_list = time_list_of_realized_intentions + time_list_of_negative_samples
        for t in concatenated_list:
            # Extract features (accelerometer readings) and label vector for pid
            current_accelerometer_data_window = extractor.__call__(pid, t[0], t[2])
            current_accelerometer_data_segment = extractor.__call__(pid, t[1], t[2])
            current_label = extract_label(t)
            # Add the new samples to the list
            X_windows.append(current_accelerometer_data_window)
            X_segments.append(current_accelerometer_data_segment)
            y.append(current_label)
        # Find the maximum row size across all sub-matrices
    max_row_size = max(len(row) for sub_matrix in X_segments for row in sub_matrix)
    X_segments = pad_matrix(X_segments, max_row_size)
    return np.array(X_windows), np.array(X_segments), np.array(y)


def generate_unrealized_classification_samples(pids, segment_length, max_row_size):
    X_windows = []
    X_segments = []
    y = []
    accel_ds_path = "../data/accel/subj_accel_interp.pkl"
    extractor = AccelExtractor(accel_ds_path)
    # Loop over every person id
    for pid in pids:
        # Load the annotated data for realized intentions
        time_list_of_unrealized_intentions = find_non_overlapping_segments_for_pid(pid, segment_length)
        for t in time_list_of_unrealized_intentions:
            # Extract features (accelerometer readings) and label vector for pid
            current_accelerometer_data_window = extractor.__call__(pid, t[0], t[2])
            current_accelerometer_data_segment = extractor.__call__(pid, t[1], t[2])
            current_label = extract_label(t)
            X_windows.append(current_accelerometer_data_window)
            X_segments.append(current_accelerometer_data_segment)
            y.append(current_label)
    X_segments = pad_matrix(X_segments, max_row_size)
    return np.array(X_windows), np.array(X_segments), np.array(y)


def pad_matrix(matrix, max_row_size):
    # Pad each row across the sub-matrices to the front with zeros
    padded_matrix = np.zeros((len(matrix), len(matrix[0]), max_row_size))

    for i, sub_matrix in enumerate(matrix):
        for j, row in enumerate(sub_matrix):
            padded_row = np.pad(row, (max_row_size - len(row), 0), 'constant')
            padded_matrix[i, j, :] = padded_row
    return padded_matrix


def extract_label(t):
    if t[1] == t[2]:
        label = 0
    else:
        label = 1
    # Append the tuple (features, label) to the list of samples
    return label


if __name__ == '__main__':
    person_ids = [2, 3, 4, 5, 7, 10, 11, 17, 22, 23, 27, 34, 35]
    res = generate_classification_samples(person_ids, 2)
    print(res)
