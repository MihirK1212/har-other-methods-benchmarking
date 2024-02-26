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

class SSC:
    def __init__(self):
        pass

    def load_training_config(self):
        pass
    
    def get_avg_frame_dict(self, data):
        avg_frame = dict()
        for sequence in data:
            for f in range(sequence.shape[0]):
                if f not in avg_frame:
                    avg_frame[f] = [] 
                avg_frame[f].append(sequence[f].reshape(constants.NUM_JOINTS*3,1))
        avg_frame = {
            key: np.mean(np.array(value), axis=0)
            for key, value in avg_frame.items()
        }
        return avg_frame
    

    def get_time_pruned_data(self, data, to_select=25):
        def select_frames(sequence, to_select):
            num_frames, num_joints, _ = sequence.shape
            if num_frames >= to_select:
                selected_indices = np.random.choice(num_frames, to_select, replace=False)
                selected_frames = sequence[selected_indices]
            else:
                selected_frames = np.zeros((to_select, num_joints, 3))
                selected_frames[:num_frames] = sequence
        
            return selected_frames.reshape(-1)
        selected_vectors = []
        for sequence in data:
            selected_vector = select_frames(sequence, to_select)
            selected_vectors.append(selected_vector)
        return np.array(selected_vectors)

    
    def get_covariance_data(self, data, avg_frame):
        cov = []
        for sequence in data:
            num_frames = sequence.shape[0]
            transformed_seq = sequence.copy().reshape(num_frames, constants.NUM_JOINTS*3, 1)
            del_matrix = np.zeros((3*constants.NUM_JOINTS, 3*constants.NUM_JOINTS))
            for t in range(num_frames):
                diff = transformed_seq[t] - avg_frame[t]
                del_matrix += diff @ diff.T
            del_matrix /= num_frames

            res = np.zeros((3*constants.NUM_JOINTS * (3*constants.NUM_JOINTS - 1)) // 2)
            index = 0
            for i in range(constants.NUM_JOINTS):
                for j in range(i, constants.NUM_JOINTS):
                    res[index] = del_matrix[i, j]
                    index += 1
            cov.append(res)
        cov = np.stack(cov)
        return cov
                
    def spectral_clustering(self, A, num_clusters):
        # Apply Spectral Clustering
        clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', random_state=0).fit(A)
        labels = clustering.labels_
        return labels

    def hungarian_mapping(self, true_labels, predicted_labels, num_classes, num_clusters):
        # Create a cost matrix based on label similarities
        cost_matrix = np.zeros((num_classes, num_clusters))
        for i in range(num_classes):
            for j in range(num_clusters):
                # Count how many points from true class i are assigned to predicted cluster j
                cost_matrix[i, j] = np.sum((true_labels == i) & (predicted_labels == j))
        
        # Apply Hungarian algorithm to find the best mapping
        true_class_indices, cluster_indices = linear_sum_assignment(-cost_matrix)
        mapping = {cluster_indices[i]: true_class_indices[i] for i in range(len(cluster_indices))}
        mapped_labels = [mapping[label] for label in predicted_labels]
    
        return mapped_labels

    def get_ssc_labels(self, data, true_labels):
        avg_frame = self.get_avg_frame_dict(data) 
        covariance_data = self.get_covariance_data(data, avg_frame)

        model = ElasticNetSubspaceClustering(n_clusters=3,algorithm='lasso_lars',gamma=50).fit(covariance_data)

        A = model.affinity_matrix_.toarray()

        num_clusters = min(len(data), 20)
        predicted_labels = self.spectral_clustering(A, num_clusters)
        mapped_labels = self.hungarian_mapping(true_labels, predicted_labels, constants.NUM_CLASSES, num_clusters)
        return mapped_labels

    def get_tsc_labels(self, data, true_labels):

        X = self.get_time_pruned_data(data, to_select=config.PRUNE_NUM_FRAMES)
        assert X.shape == (len(data), config.PRUNE_NUM_FRAMES*constants.NUM_JOINTS*3)
        X = X.transpose()
        
        r = 20
        s = 3
        lambda1 = 0.1
        lambda2 = 0.1
        alpha = 0.1
        Z, D, U, V = tsc.graph_clustering(X, r, s, lambda1, lambda2, alpha)
        A = cosine_similarity(Z)

        num_clusters = r
        predicted_labels = self.spectral_clustering(A, num_clusters)
        mapped_labels = self.hungarian_mapping(true_labels, predicted_labels, constants.NUM_CLASSES, num_clusters)
        return mapped_labels

    def get_prediction(self, sequence):
        return 0
    
    def get_predictions(self, data, true_labels):

        ssc_labels = self.get_ssc_labels(data, true_labels)
        tsc_labels = self.get_tsc_labels(data, true_labels)

        predictions = []
        for sequence in data:
            predictions.append(self.get_prediction(sequence))
        return predictions

'''
Provide python code for the following:

Given list of sequences of length 'num_samples;, each sequence is of shape (num_frames, num_joints, 3)
num_frames can be different for each sequence

A variable to_select = 25

For each sequence, select 'to_select' number of frames randomly
If num_frames < to_select, use (to_select - num_frames) zero value frames of shape (num_joints, 3)
Convert the sequence into a single vector of shape (to_select*num_frames*3)

Hence we will get 'num_samples' vectors, one corresponding to each sequence
Convert these vectors into single numpy array of shape (num_samples, to_select*num_frames*3)
'''

'''
Provide vectorized python code using numpy for the following

Input: 
X which is a sequence of shape (num_frames, num_joints*3, 1)
avgFrame which is a dictionary containing average frame value for time stamp t, as avgFrame[t] of shape (num_joints*3, 1)

del = zero matrix of shape (num_joints, num_joints)

For each time stamp t in range(num_frames):
    del = del + (X[t] - avgFrame[t]) x (X[t] - avgFrame[t])^t
del = del / num_frames


Create vector res using all diagonal values and upper triangular values of del
'''