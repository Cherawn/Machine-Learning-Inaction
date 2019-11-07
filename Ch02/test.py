import kNN
import numpy as np
import matplotlib.pylab as plt

mat, labels = kNN.file_to_matrix('datingTestSet2.txt')

norm_mat, min_values, ranges = kNN.auto_norm(mat)

fig = plt.figure(figsize=(6, 12))
ax1 = fig.add_subplot(311)
ax1.scatter(norm_mat[:, 0], norm_mat[:, 1], 15*np.array(labels), np.array(labels))
ax2 = fig.add_subplot(312)
ax2.scatter(norm_mat[:, 0], norm_mat[:, 2], 15*np.array(labels), np.array(labels))
ax3 = fig.add_subplot(313)
ax3.scatter(norm_mat[:, 1], norm_mat[:, 2], 15*np.array(labels), np.array(labels))

plt.show()
