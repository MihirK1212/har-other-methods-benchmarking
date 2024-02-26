import numpy as np

import config 
import data.statistical_moments_aggregation as statistical_moments_aggregation

def get_sequence_windows(sequence, required_windows = config.NUM_WINDOWS, min_stride = 2):
    num_frames = sequence.shape[0]
    stride = max(min_stride, num_frames//required_windows)
    window_size = num_frames - (required_windows - 1)*stride
    assert window_size >= stride
    lb, ub = 0, window_size - 1
    windows = []
    while (lb<=ub and ub<num_frames):
      window = sequence[lb:(ub+1)].copy()
      windows.append(window)
      lb+=stride
      ub+=stride
    assert ub == (num_frames - 1 + stride)
    assert len(windows) == required_windows
    return windows

def get_window_division_aggregation(sequence):
    windows = get_sequence_windows(sequence=sequence)
    res = []
    for window in windows:
      window_mean_feature_vector = statistical_moments_aggregation.get_statistical_moments_aggregation(sequence=window)
      res.append(window_mean_feature_vector)
    return np.concatenate(res)