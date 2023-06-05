from data_loading.load_annotations import load_realized_annotations


def find_segment_info(pids, segment_length):
    total_sum = 0
    min_segment = 1000
    max_segment = 0
    cumulative_size = 0
    # loop over all person ids
    for pid in pids:
        # load the time list for that person id
        time_list = load_realized_annotations(pid, segment_length)
        for time in time_list:
            # find the current segment length and add to the total sum
            current_segment_length = (time[1] - time[0])
            total_sum += current_segment_length
            # if current segment length smaller than min, assign min to current segment length
            if current_segment_length < min_segment:
                min_segment = current_segment_length
            # if current segment length bigger than max, assign max to current segment length
            if current_segment_length > max_segment:
                max_segment = current_segment_length
        # add the size of the time list to the cumulative size (to retrieve the total over all people)
        cumulative_size += len(time_list)
    # calculate the average
    average_size = total_sum / cumulative_size
    return average_size, min_segment, max_segment


if __name__ == '__main__':
    person_ids = [2, 3, 4, 5, 7, 10, 11, 17, 22, 23, 27, 34, 35]
    segment_info = find_segment_info(person_ids, 2)
    print('Average of all segments: ', segment_info[0], ', Minimum size of segment:', segment_info[1], 'Maximum size of segment:', segment_info[2])
