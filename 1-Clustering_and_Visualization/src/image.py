import numpy as np
import cv2
from sklearn.cluster import KMeans


img = cv2.imread('boston-1993606_1280.jpg')
shape = img.shape  # (850, 1280, 3)
img = img.reshape(-1, 3)  # convert 3d array to 2d

# implement KMeans
img_kmeans = KMeans(n_clusters=10).fit(img)
cluster_labels = img_kmeans.labels_
cluster_centers = img_kmeans.cluster_centers_

manipulated_img = np.copy(img)
# replace each pixel of original image with its cluster centers
for cluster in np.arange(10):
    index_list = np.where(cluster_labels == cluster)
    for index in index_list:
        manipulated_img[index] = cluster_centers[cluster]

# show original and manipulated images
cv2.imshow('Display Window 1', img.reshape(shape))
cv2.imshow('Display Window 2', manipulated_img.reshape(shape))
cv2.waitKey(0)
cv2.destroyAllWindows()
