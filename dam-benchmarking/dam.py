import numpy as np
from scipy.ndimage import gaussian_filter1d
import joblib

import constants
import utils
import config

class DAM:
    def __init__(self):
        pass

    def load_training_config(self):
        self.som = joblib.load(constants.SOM_PATH)
        self.class_cluster_probs = np.random.rand(constants.NUM_CLASSES, config.NUM_CLUSTERS)

    def smooth_joint_positions(self, joint_positions, sigma=1.0):
        smoothed_positions = np.zeros_like(joint_positions)
        for j in range(joint_positions.shape[1]):
            for c in range(joint_positions.shape[2]):
                smoothed_positions[:, j, c] = gaussian_filter1d(joint_positions[:, j, c], sigma=sigma)  
        return smoothed_positions


    def resample_joint_positions(self, joint_positions, target_frames):
        num_frames, num_dims = joint_positions.shape[0], joint_positions.shape[2]
        resampled_positions = np.zeros((target_frames, constants.NUM_JOINTS , num_dims))
        frame_x_values = np.arange(num_frames)
        for j in range(constants.NUM_JOINTS):
            for c in range(num_dims):
                resampled_positions[:, j, c] = utils.resample_along_arc_length(frame_x_values, joint_positions[:, j, c], target_frames)      
        return resampled_positions
        
    def get_prediction(self, sequence):

        smoothed_positions = self.smooth_joint_positions(sequence, sigma=1.0)
        assert sequence.shape == smoothed_positions.shape
        resampled_positions = self.resample_joint_positions(smoothed_positions, target_frames=config.NUM_RESAMPLED_FRAMES)
        assert resampled_positions.shape == (config.NUM_RESAMPLED_FRAMES, constants.NUM_JOINTS, 3)

        direction_frames = np.diff(resampled_positions, axis=0)
        f, W =  0, config.WINDOW_SIZE
        histogram_bins = np.zeros((config.NUM_CLUSTERS,), dtype=np.float64)
        count = 0
        while (f + W) <= direction_frames.shape[0]:
            wdf = direction_frames[f:(f+W)]
            wdf_flat = wdf.reshape(-1)
            winner = self.som.winner(wdf_flat)
            cluster_number = winner[0] * self.som._weights.shape[1] + winner[1]
            histogram_bins[cluster_number]+=1
            f+=1
            count+=1
        assert count == (config.NUM_RESAMPLED_FRAMES - config.WINDOW_SIZE)
        histogram_bins/=(config.NUM_RESAMPLED_FRAMES - config.WINDOW_SIZE)
        assert self.class_cluster_probs.shape[0] == constants.NUM_CLASSES and self.class_cluster_probs.shape[1] == histogram_bins.shape[0]
        class_probabilities = np.dot(self.class_cluster_probs, histogram_bins)
        return np.argmax(class_probabilities)
 
    def get_predictions(self, data):
        predictions = []
        for sequence in data:
            predictions.append(self.get_prediction(sequence))
        return predictions




