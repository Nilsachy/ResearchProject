import random

from data_loading.load_annotations import load_realized_annotations


def generate_non_overlapping_times(current_times, segment_length, start_range, end_range):
    start_time = random.uniform(start_range, end_range - segment_length)
    end_time = start_time + segment_length

    for time in current_times:
        if start_time <= time[2] and end_time >= time[0]:
            # Overlaps with existing event, recursively call itself
            return generate_non_overlapping_times(current_times, segment_length, start_range, end_range)

    start_time = round(start_time, 3)
    end_time = round(end_time, 3)
    start_of_intention = end_time
    return start_time, start_of_intention, end_time


def generate_negative_intentions_intervals(pid, segment_length):
    current_times = load_realized_annotations(pid, segment_length)

    start_range = current_times[0][0]
    end_range = current_times[-1][2]

    new_events = []
    for _ in current_times:
        new_event = generate_non_overlapping_times(current_times, segment_length, start_range, end_range)
        new_events.append(new_event)

    return new_events


if __name__ == '__main__':
    generate_negative_intentions_intervals(23, 2)
