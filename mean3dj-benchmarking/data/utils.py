from collections import defaultdict
import numpy as np


def check_anomalies(data):
    for sequence in data:
        for frame in sequence:
            for joint in frame:
                if joint[0] >= 1000 or joint[1] >= 1000 or joint[2] >= 1000:
                    raise ValueError

def remove_anomalies(data, labels, subjects):
    frames_to_drop = defaultdict(list)
    dropped = 0
    for sequence_ind, sequence in enumerate(data):
        for frame_ind in range(sequence.shape[0]):
            for joint in sequence[frame_ind]:
                assert joint.shape == (3,)
                if joint[0] >= 1000 or joint[1] >= 1000 or joint[2] >= 1000:
                    frames_to_drop[sequence_ind].append(frame_ind)
                    dropped += 1
    for k, v in frames_to_drop.items():
        dropped -= len(v)
    assert dropped == 0
    print("Dropping frames:", frames_to_drop)
    for sequence_ind in range(len(data)):
        data[sequence_ind] = np.delete(
            data[sequence_ind], frames_to_drop[sequence_ind], axis=0
        )
        
    check_anomalies(data)
    return data, labels, subjects

