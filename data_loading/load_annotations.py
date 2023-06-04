import numpy as np

from utils.AccelExtractor import AccelExtractor


def load_annotations(pid, segment_length):
    file_path = "../data/annotations/realized/Person" + str(pid) + "-realized-intentions.txt"

    time_list = []

    # Read the file line by line and process each line
    with open(file_path, "r") as file:
        for line in file:
            # Remove leading/trailing whitespaces and split the line into start and end times
            start_time, end_time = map(str.strip, line.split("-"))
            # Convert time string from hh::mm::ss.sss to seconds as a single float value
            start_time = convert_to_seconds(start_time)
            end_time = convert_to_seconds(end_time)
            # Define the start of the segment as the end time minus the length of the segment
            segment_start = end_time - segment_length
            # Start intention time should be higher or equal to the start of the segment
            start_time = max(segment_start, start_time)
            # Create a 3-tuple of segment start, intention start and end times and append it to the list
            time_list.append((segment_start, start_time, end_time))
    return time_list


def convert_to_seconds(time_string):
    # Split the time string into hours, minutes, seconds, and milliseconds
    hours, minutes, seconds = map(float, time_string.split(':'))
    # Calculate the total seconds + 1 hour to account for the start_time of the dataset (starts at 01:00:00)
    total_seconds = (1 + hours) * 3600 + minutes * 60 + seconds
    return total_seconds


if __name__ == '__main__':
    realized_intentions = load_annotations(pid=23, segment_length=2)
    time_list = realized_intentions[0]
    accel_ds_path = "../data/accel/subj_accel_interp.pkl"
    extractor = AccelExtractor(accel_ds_path)
    accelerometer_data = extractor.__call__(2, time_list[0], time_list[2])
    print(accelerometer_data)
