import math

import numpy as np

from data_loading.load_annotations import load_annotations
from utils.AccelExtractor import AccelExtractor


def generate_samples(pids, segment_length):
    samples = []
    accel_ds_path = "../data/accel/subj_accel_interp.pkl"
    extractor = AccelExtractor(accel_ds_path)
    # Loop over every person id
    for pid in pids:
        # Load the annotated data for realized intentions
        time_list_of_realized_intentions = load_annotations(pid, segment_length)
        for t in time_list_of_realized_intentions:
            # Extract features (accelerometer readings) and label vector for pid
            current_sample = extract_features_and_label_vector(pid, t, segment_length, extractor)
            # Add the new samples to the list
            samples.append(current_sample)
    return samples


def extract_features_and_label_vector(pid, t, segment_length, extractor):
    # Call the extractor class to retrieve accelerometer readings from segment_start to end_time
    accelerometer_data = extractor.__call__(pid, t[0], t[2])
    # Generate a target vector of defined segment length
    label_vector = generate_target_vector(segment_length, t[0], t[1], t[2])
    # Append the tuple (features, label) to the list of samples
    return accelerometer_data, label_vector


def generate_target_vector(segment_length, segment_start, start_time, end_time):
    # Fill in a numpy array with only zeros with size segment_length
    segment_length_scaled = segment_length * 100
    sampled_target = np.zeros(segment_length_scaled)  # Create an array of zeros with specified length
    # Find the ceiling start index of the intention to speak based on the segment info
    start_index = math.ceil((start_time - segment_start) / (end_time - segment_start) * segment_length_scaled)
    # If start index is smaller than segment length assign 1s
    if start_index < segment_length_scaled:
        sampled_target[start_index:] = 1  # Assign 1 to positions after the specified index
    return sampled_target


if __name__ == '__main__':
    person_ids = [2, 3, 4, 5, 7, 10, 11, 17, 22, 23, 27, 34, 35]
    res = generate_samples(person_ids, 2)
    print(res)
