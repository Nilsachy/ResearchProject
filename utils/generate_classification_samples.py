import math

import numpy as np

from data_exploration.find_non_and_overlapping_segments import find_non_overlapping_segments_for_pid
from data_loading.load_annotations import load_realized_annotations, load_unrealized_annotations
from utils.AccelExtractor import AccelExtractor
from utils.generate_negative_intentions_intervals import generate_negative_intentions_intervals


def generate_classification_samples(pids, segment_length):
    X = []
    y = []
    accel_ds_path = "../data/accel/subj_accel_interp.pkl"
    extractor = AccelExtractor(accel_ds_path)
    # Loop over every person id
    for pid in pids:
        # Load the annotated data for realized intentions
        time_list_of_realized_intentions = load_realized_annotations(pid, segment_length)
        time_list_of_negative_samples = generate_negative_intentions_intervals(pid, segment_length)
        concatenated_list = time_list_of_realized_intentions + time_list_of_negative_samples
        for t in concatenated_list:
            # Extract features (accelerometer readings) and label vector for pid
            current_accelerometer_data = extractor.__call__(pid, t[0], t[2])
            current_label = extract_label(t)
            # Add the new samples to the list
            X.append(current_accelerometer_data)
            y.append(current_label)
    return np.array(X), np.array(y)


def generate_unrealized_classification_samples(pids, segment_length):
    X = []
    y = []
    accel_ds_path = "../data/accel/subj_accel_interp.pkl"
    extractor = AccelExtractor(accel_ds_path)
    # Loop over every person id
    for pid in pids:
        # Load the annotated data for realized intentions
        time_list_of_unrealized_intentions = find_non_overlapping_segments_for_pid(pid, segment_length)
        for t in time_list_of_unrealized_intentions:
            # Extract features (accelerometer readings) and label vector for pid
            current_accelerometer_data = extractor.__call__(pid, t[0], t[2])
            current_label = extract_label(t)
            # Add the new samples to the list
            X.append(current_accelerometer_data)
            y.append(current_label)
    print('Num of unrealized intentions: ', len(X))
    return np.array(X), np.array(y)


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
