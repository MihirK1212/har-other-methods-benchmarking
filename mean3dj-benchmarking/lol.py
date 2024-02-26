# import numpy as np
# from minisom import MiniSom
# import matplotlib.pyplot as plt
# import joblib

# np.random.seed(42)

# data = np.random.rand(100, 3, 20, 3)  
# data_flat = data.reshape((data.shape[0], -1))
# map_size = (25, 25)  
# input_len = data_flat.shape[1]  
# sigma = 1.0  
# learning_rate = 0.5  
# num_epochs = 100
# som = MiniSom(map_size[0], map_size[1], input_len, sigma=sigma, learning_rate=learning_rate)
# som.train_random(data_flat, num_epochs)
# joblib.dump(som, 'trained_som.joblib')

import numpy as np
import joblib

som = joblib.load('trained_som.joblib')

F = 16
J = 20
sequence = np.random.rand(F, J, 3)

direction_frames = np.diff(sequence, axis=0)

W = 3  
wdfs = []
f = 0
while (f + W) <= direction_frames.shape[0]:
    wdfs.append(direction_frames[f:(f+W)])
    f+=1

print(len(wdfs))
for wdf in wdfs:
    print('wdf shape:', wdf.shape, end=' ')
    wdf_flat = wdf.reshape(-1)
    winner = som.winner(wdf_flat)
    cluster_number = winner[0] * som._weights.shape[1] + winner[1]
    print('cluster:', cluster_number)


