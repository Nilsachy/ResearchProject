from data_loading.load_annotations import load_realized_annotations, load_unrealized_annotations


def find_non_overlapping_segments_for_pid(pid, segment_length):
    non_overlapping_segments = []
    overlaps = []
    realized_intentions = load_realized_annotations(pid, segment_length)
    unrealized_intentions = load_unrealized_annotations(pid, segment_length)
    for t_unrealized in unrealized_intentions:
        segment_start = t_unrealized[0]
        start_time = t_unrealized[1]
        end_time = t_unrealized[2]
        overlap = False
        for t_realized in realized_intentions:
            if segment_start < t_realized[0] < end_time:
                overlap = True
        if overlap:
            overlaps.append((pid, t_unrealized))
        else:
            non_overlapping_segments.append(t_unrealized)
    return non_overlapping_segments


def find_overlapping_segments(pids, segment_length):
    overlaps = []
    non_overlapping_segments = []
    for pid in pids:
        realized_intentions = load_realized_annotations(pid, segment_length)
        unrealized_intentions = load_unrealized_annotations(pid, segment_length)
        for t_unrealized in unrealized_intentions:
            segment_start = t_unrealized[0]
            start_time = t_unrealized[1]
            end_time = t_unrealized[2]
            overlap = False
            for t_realized in realized_intentions:
                if segment_start < t_realized[0] < end_time:
                    overlap = True
            if overlap:
                overlaps.append((pid, t_unrealized))
            else:
                non_overlapping_segments.append(t_unrealized)
    return overlaps


if __name__ == '__main__':
    pids = [2, 3, 4, 5, 7, 10, 11, 17, 22, 23, 27, 34]
    print(find_overlapping_segments(pids, 2))