import numpy as np

def phi(frame, u_i, omhega):
    numerator = 1 / (1 + np.exp(-np.dot(omhega[u_i], frame)))
    denominator = np.sum([1 / (1 + np.exp(-np.dot(omhega[k], frame))) for k in range(len(omhega))])
    return np.log(numerator / denominator)

def compute_score(x, y, omhega, u, s):
    s = np.insert(s.copy(), 0, 0)
    num_intervals = u.shape[0]
    score = 0
    for i in range(num_intervals):
        start_idx = s[i]
        end_idx = s[i + 1]
        interval_sum = 0
        for j in range(start_idx, end_idx):
            interval_sum += phi(x[j], u[i], omhega)
        score += interval_sum
    return score

# Example usage:
num_frames = 100
d = 10
num_subclasses = 5
num_intervals = 3
x = np.random.rand(num_frames, d)
y = np.random.randint(0, 20)
omhega = np.random.rand(num_subclasses, d)
u = np.random.randint(0, num_subclasses, size=num_intervals)
s = np.sort(np.random.randint(1, num_frames, size=num_intervals))
score = compute_score(x, y, omhega, u, s)
print("Score:", score)


'''
Provide python code for the following:

Input:
1) x of shape (num_frames, d)
2) y which is the class label in the range [0, 1, 2, .., 19]
3) omhega of shape (num_subclasses)
4) Paremeter u of size 'num_intervals', Paremeter s of size 'num_intervals + 1'
5) Values of u are in the range [0, 1, 2, ..., num_subclasses-1]
6) Values of s are in the range [1, 2, 3, ..., num_frames-1]

Output:

score = sum(i=0 to num_intervals-1)(phi(x_s[i], u_i, omhega) + phi(x_s[i+1], u_i, omhega))
phi(frame, u_i, omhega) = log ( (1 / (1 + e^(-ohmega_u_i^T * frame))) / sum(k=0 to num_subclasses-1)((1 / (1 + e^(-ohmega_k^T * frame)))) )
'''

