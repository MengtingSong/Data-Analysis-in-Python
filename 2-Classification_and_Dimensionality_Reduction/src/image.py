import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


X = np.load('../data/mnist_data.npy')

u, s, vt = np.linalg.svd(X, full_matrices=False)

# construct a rank-10 version of the images
scopy = s.copy()
scopy[10:] = 0
X_pca_10 = u.dot(np.diag(scopy)).dot(vt)

fig, axes = plt.subplots(10, 10, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_pca_10[i].reshape(28, 28))
fig.suptitle('Images with Top 10 Components')
plt.show()
