import numpy as np
import joblib

import constants
import utils
import config


class GraphBased:
    def __init__(self):
        pass

    def load_training_config(self):
        self.Gs = utils.generate_symmetric_binary_matrix(constants.NUM_JOINTS)
        self.Gt = utils.get_linear_temporal_graph()
        self.loaded_model = joblib.load('svm_model.pkl') 

    def get_aggregation(self, sequence):

        num_frames = sequence.shape[0]

        Gst = utils.cartesian_product(self.Gs, self.Gt)
        _, eigenvectors = np.linalg.eig(Gst)
        eigenvectors = eigenvectors.transpose(0, 1)

        F = np.zeros((num_frames - config.Nt + 1, constants.NUM_JOINTS, config.Nt, 3))
        for t in range(num_frames - config.Nt + 1):
            for i in range(constants.NUM_JOINTS):
                for s in range(config.Nt):
                    F[t][i][s] = sequence[t + s - 1][i]

        # Reshape the array
        F = F.reshape(F.shape[0], F.shape[1] * F.shape[2], F.shape[3])

        Fdash = []
        for t in range(F.shape[0]):
            fdash = np.matmul(eigenvectors, F[t]).reshape(-1)
            Fdash.append(fdash)
        Fdash = np.stack(Fdash)

        res = []

        for pyramid_level in range(config.M):
            k = pow(2, pyramid_level)
            m, n = Fdash.shape
            block_size = m // k
            remainder = m % k

            blocks = []
            start_idx = 0
            for i in range(k):
                if i < remainder:
                    end_idx = start_idx + block_size + 1
                else:
                    end_idx = start_idx + block_size
                curr_block = Fdash[start_idx:end_idx, :]
                curr_block = np.mean(curr_block, axis=0)
                blocks.append(curr_block)
                start_idx = end_idx

            assert len(blocks) == k
            curr_pyramid = np.concatenate(blocks)
            res.append(curr_pyramid)
        
        res = np.concatenate(res)
        return res

    def expand_array(self, arr, num_expanded_frames):
        new_shape = (num_expanded_frames, constants.NUM_JOINTS, 3)
        assert arr.shape[0] < num_expanded_frames
        padded_arr = np.zeros(new_shape)  # Create zero-filled array with the new shape
        padded_arr[:arr.shape[0], :, :] = arr  # Fill the beginning of the new array with original array content
        return padded_arr

    def get_predictions(self, data):
        X = []
        for sequence in data:
            t_sequence = sequence.copy()
            if t_sequence.shape[0] < config.Nt:
                t_sequence = self.expand_array(t_sequence, config.Nt)
            X.append(self.get_aggregation(t_sequence))
        X = np.stack(X)
        X = np.nan_to_num(X, nan=0)
        return self.loaded_model.predict(X).tolist()


"""
Input:
- Matrix A of shape (num_frames, num_joints, 3) where each frame contains 3d coordinates of 'num_joints' joints
- a parameter 'Nt'

Output:
- Matrix B of shape (num_frames, num_joints, Nt, 3) 
For frame t, joint i, and s in [1, Nt]:
B[t][i][s] = B[t+s-1][i][s]
"""
