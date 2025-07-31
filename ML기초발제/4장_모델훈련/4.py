import numpy as np
np.random.seed(42)
m=100
X=2*np.random.rand(m,1)
y=4+3*X+np.random.randn(m,1)

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import add_dummy_feature
X_b = add_dummy_feature(X)
theta_best=np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
theta_best