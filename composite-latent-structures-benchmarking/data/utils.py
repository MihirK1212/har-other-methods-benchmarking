from collections import defaultdict
import numpy as np


tssi_order = [
    3, 2, 19, 2, 1, 8, 10, 12, 10, 8, 1, 2,
    0, 7, 9, 11, 9, 7, 0, 2, 3, 6, 4, 14, 16,
    18, 16, 14, 4, 6, 5, 13, 15, 17, 15, 13,
    5, 6, 3, 2
]


def get_tssi_dim_sequence(dim_sequence):
    return dim_sequence[:, tssi_order]

def get_tssi_angle_sequence(sequence):
    joint_indices = np.column_stack((tssi_order, (np.array(tssi_order) + 1) % sequence.shape[1]))
    vectors1 = sequence[:, joint_indices[:, 0], :]
    vectors2 = sequence[:, joint_indices[:, 1], :]
    dot_products = np.einsum('ijk,ijk->ij', vectors1, vectors2)
    magnitudes1 = np.linalg.norm(vectors1, axis=2)
    magnitudes2 = np.linalg.norm(vectors2, axis=2)
    cosine_angles = dot_products / (magnitudes1 * magnitudes2)
    cosine_angles = np.clip(cosine_angles, -1.0, 1.0)
    angles = np.arccos(cosine_angles)
    angles_in_degrees = np.degrees(angles)
    angles_normalized = angles_in_degrees / 360.0
    angles_normalized = np.nan_to_num(angles_normalized, nan=0)
    return angles_normalized


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

