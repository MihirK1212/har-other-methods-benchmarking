import numpy as np
from scipy.ndimage import gaussian_filter1d
import joblib
from sklearn.cluster import SpectralClustering
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

import constants
import utils
import config
from elasticnet import ElasticNetSubspaceClustering
import tsc
from scipy.optimize import minimize

class CompLatent:
    def __init__(self):
        pass

    def load_training_config(self):
        self.omhega = np.random.rand(config.NUM_SUBCLASSES,)
    
    def get_transformed_sequence(self, sequence):
        return sequence.reshape(sequence.shape[0], -1)
    
    def get_optimal_values(self, x, y, omhega, num_intervals, num_subclasses, max_iter=5):
        num_frames = x.shape[0]
        u = init_u(num_intervals, num_subclasses)
        s = init_s(num_intervals, num_frames)
        for _ in range(max_iter):
            for i in range(num_intervals):
                u[i], s[i] = get_optimal(x, y, u, s, i, omhega, num_subclasses, num_frames)
            s[-1] = num_frames
        return u, s

    def get_prediction(self, sequence):
        x = self.get_transformed_sequence(sequence.copy())
        scores = []
        for y in range(constants.NUM_CLASSES):
            u, s = self.get_optimal_values(x, y, self.omhega, config.NUM_INTERVALS, config.NUM_SUBCLASSES)
            u = np.round(u).astype(np.int32)
            s = np.round(s).astype(np.int32)
            scores.append(score_function(x, y, u, s, self.omhega))
        return np.argmax(np.array(scores))
    
    def get_predictions(self, data):
        predictions = []
        for sequence in data:
            predictions.append(self.get_prediction(sequence))
        return predictions

def phi(frame, u_i, omhega):
    assert u_i>=0 and u_i<config.NUM_SUBCLASSES
    u_i = int(u_i)
    numerator = 1 / (1 + np.exp(-np.dot(omhega[u_i], frame)))
    denominator = np.sum([1 / (1 + np.exp(-np.dot(omhega[k], frame))) for k in range(len(omhega))])
    return np.log(numerator / denominator)[0]

def score_function(x, y, u, s, omhega):
    return np.random.rand()
    s = np.insert(s.copy(), 0, 0)
    num_intervals = u.shape[0]
    score = 0
    for i in range(num_intervals):
        start_idx = s[i]
        end_idx = s[i + 1]
        interval_sum = 0
        for j in range(int(start_idx), int(end_idx)):
            interval_sum += phi(x[j], u[i], omhega)
        score += interval_sum
    return score
    

def init_u(num_intervals, num_subclasses):
    u = np.zeros(num_intervals)
    for i in range(num_intervals):
        u[i] = i%num_subclasses
    return u

def objective_function(params, i, x, y, u, s, omhega):
    u[i] = round(params[0])
    s[i] = round(params[1])
    return -score_function(x, y, u, s, omhega)

def get_optimal(x, y, u, s, i, omhega, num_subclasses, num_frames):
    initial_guess = [u[i], s[i]]
    bounds = [(0, num_subclasses-1), (1, num_frames-1)]  # Bounds for u and s
    result = minimize(objective_function, initial_guess, args=(i, x, y, u, s, omhega), bounds=bounds)
    return result.x

def init_s(num_intervals, num_frames):
    s = np.zeros(num_intervals)
    delta = num_frames // num_intervals
    curr = delta
    for i in range(num_intervals):
        s[i] = curr
        curr+=delta
    s[-1] = num_frames
    return s
