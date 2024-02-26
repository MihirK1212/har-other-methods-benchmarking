import numpy as np
from scipy.stats import skew, kurtosis

import utils.utils as utils
import data.utils as data_utils
import config

def aggregate_dim_sequence_statistical_moments(dim_sequence):
    return utils.remove_nan(np.concatenate((
      utils.remove_nan(np.mean(dim_sequence, axis=0)), 
      utils.remove_nan(np.var(dim_sequence, axis=0)),
      utils.remove_nan(skew(dim_sequence, axis=0)),
      utils.remove_nan(kurtosis(dim_sequence, axis=0))
    ), axis=0))
    


def aggregate_angle_sequence_statistical_moments(angle_sequence):
    return utils.remove_nan(np.concatenate((
      utils.remove_nan(np.mean(angle_sequence, axis=0)), 
      utils.remove_nan(np.var(angle_sequence, axis=0)),
      utils.remove_nan(skew(angle_sequence, axis=0)),
      utils.remove_nan(kurtosis(angle_sequence, axis=0))
    ), axis=0))


def get_statistical_moments_aggregation(sequence):

    angle_sequence = data_utils.get_tssi_angle_sequence(sequence)

    sequence -= sequence[:, config.ROOT_JOINT_INDEX, :][:, np.newaxis, :]

    dim_sequence_x, dim_sequence_y, dim_sequence_z = sequence[:, :, 0], sequence[:, :, 1], sequence[:, :, 2]
    dim_sequence_x, dim_sequence_y, dim_sequence_z = data_utils.get_tssi_dim_sequence(dim_sequence_x), data_utils.get_tssi_dim_sequence(dim_sequence_y), data_utils.get_tssi_dim_sequence(dim_sequence_z)

    mean_feature_vector = []

    x_rep, y_rep, z_rep = aggregate_dim_sequence_statistical_moments(dim_sequence_x), aggregate_dim_sequence_statistical_moments(dim_sequence_y), aggregate_dim_sequence_statistical_moments(dim_sequence_z)
    for i in range(len(data_utils.tssi_order)):
      mean_feature_vector.extend([x_rep[i], x_rep[2*i], x_rep[3*i], x_rep[4*i]])
    for i in range(len(data_utils.tssi_order)):
      mean_feature_vector.extend([y_rep[i], y_rep[2*i], y_rep[3*i], y_rep[4*i]])
    for i in range(len(data_utils.tssi_order)):
      mean_feature_vector.extend([z_rep[i], z_rep[2*i], z_rep[3*i], z_rep[4*i]])

    angle_rep = aggregate_angle_sequence_statistical_moments(angle_sequence=angle_sequence)
    for i in range(len(data_utils.tssi_order)):
     mean_feature_vector.extend([angle_rep[i], angle_rep[2*i], angle_rep[3*i], angle_rep[4*i]])

    return np.array(mean_feature_vector)