import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.feature_selection import SelectKBest, f_classif

num_states = 5

X_train, y_train = np.random.rand(100, 200), np.random.randint(2, size=100)
X_test, y_test = np.random.rand(20, 200), np.random.randint(2, size=20)

# Feature selection
selector = SelectKBest(score_func=f_classif, k=10)  # Example: selecting 10 best features
selected_features_train = selector.fit_transform(X_train, y_train)

# Train HMM with GID distribution as emission
model = GaussianHMM(n_components=num_states, covariance_type='full')
model.fit(selected_features_train)


selected_features_test = selector.transform(X_test)
predicted_labels = model.predict(selected_features_test)
print(predicted_labels)