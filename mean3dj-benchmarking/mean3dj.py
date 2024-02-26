import numpy as np
from scipy.ndimage import gaussian_filter1d
import joblib

import constants
import utils
import config

class Mean3DJ:
    def __init__(self):
        pass

    def load_training_config(self):
        self.loaded_model = joblib.load('rf_model.pkl') 
    
    def get_mean_feature_vector(self, sequence):
        def get_static_posture_feature(curr_joints):
            fp = []
            for i in range(constants.NUM_JOINTS):
                for j in range(i+1, constants.NUM_JOINTS):
                    diff = list(curr_joints[i] - curr_joints[j])
                    fp.extend(diff)
            return fp

        def get_overall_dynamic_feature(init_joints, curr_joints):
            fd = []
            for i in range(constants.NUM_JOINTS):
                for j in range(constants.NUM_JOINTS):
                    diff = list(curr_joints[i] - init_joints[j])
                    fd.extend(diff)
            return fd

        init_joints = sequence[0]
        Mc = []

        for frame in sequence:
            fp = get_static_posture_feature(curr_joints=frame)
            fd = get_overall_dynamic_feature(init_joints=init_joints, curr_joints=frame)
            f = fp + fd
            Mc.append(f)

        Mc = np.transpose(np.array(Mc))
        f = np.mean(Mc, axis=1)
        return f
 
    def get_predictions(self, data):
        X = []
        for sequence in data:
            f = self.get_mean_feature_vector(sequence=sequence)
            X.append(f)
        X = np.array(X)
        return self.loaded_model.predict(X).tolist()



