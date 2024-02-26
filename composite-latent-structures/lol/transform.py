import numpy as np

# Define your sequence 's' with shape (num_frames, num_joints, 3)
# For demonstration purposes, let's assume some random values
num_frames = 10
num_joints = 5
s = np.random.rand(num_frames, num_joints, 3)

# Initialize result matrix
result = []

# Loop through each frame
for t in range(num_frames):
    x_t = []
    
    # Loop through each joint
    for i in range(num_joints):
        xmot_t_i = []
        xpos_t_i = []
        
        # Calculate xmot_t_i_j and xpos_t_i_j
        for j in range(num_joints):
            if t > 0:
                xmot_t_i_j = s[t, i] - s[t-1, j]
            else:
                xmot_t_i_j = np.zeros(3)  # If t is 0, set xmot_t_i_j to zeros
            
            xpos_t_i_j = s[t, i] - s[t, j]
            
            xmot_t_i.append(xmot_t_i_j)
            xpos_t_i.append(xpos_t_i_j)
        
        # Concatenate xmot_t_i_j and xpos_t_i_j for all j
        xmot_t_i = np.concatenate(xmot_t_i)
        xpos_t_i = np.concatenate(xpos_t_i)
        
        x_t.append(np.concatenate([xmot_t_i, xpos_t_i]))
    
    # Concatenate x_t_i for all i
    result.append(np.concatenate(x_t))

# Concatenate x_t for all t
result = np.concatenate(result)

# Print or use result as needed
print(result.shape)


'''
Provide python code using numpy for the following:

Input:
sequence 's' of shape (num_frames, num_joints, 3)


for t in range(num_frames):
    for i in range(num_joints):
        for j in range(num_joints):
            xmot_t_i_j = s_t_i - s_(t-1)_j
            xpos_t_i_j = s_t_i - s_t_j
    xmot_t_i = concat([xmot_t_i_j for all j in range(num_joints)])
    xpos_i = concat([xpos_t_i_j for all j in range(num_joints)])
        

for t in range(num_frames):
    for i in range(num_joints):
        x_t_i = concat(xmot_t_i, xpost_t_i)
    xt = concat([x_t_i for all i in range(num_joints)])

result is matrix containing rows x1, x2,...,xt,..
'''