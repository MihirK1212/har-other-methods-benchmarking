import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt
import joblib

import config
import constants

np.random.seed(42)

data = np.random.rand(100, config.WINDOW_SIZE, constants.NUM_JOINTS, 3)  
data_flat = data.reshape((data.shape[0], -1))
map_size = (config.SOM_DIM, config.SOM_DIM)  
input_len = data_flat.shape[1]  
sigma = 1.0  
learning_rate = 0.5  
num_epochs = 100
som = MiniSom(map_size[0], map_size[1], input_len, sigma=sigma, learning_rate=learning_rate)
som.train_random(data_flat, num_epochs)
joblib.dump(som, constants.SOM_PATH)