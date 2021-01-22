from sklearn.cluster import KMeans


# Use the Elbow Method to determine the optimal number of clusters
# function returns Within-Cluster-Sum of Squared Errors (WSS) score for k values from 1 to kmax
def calculate_WSS(points, kmax):
    one_dim = False
    if points.ndim == 1:
        points = points.reshape(-1, 1)
        one_dim = True

    sse = []
    for k in range(1, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        if one_dim:
            for i in range(len(points)):
                curr_center = centroids[pred_clusters[i]]
                curr_sse += (points[i, 0] - curr_center[0]) ** 2
        else:
            for i in range(len(points)):
                curr_center = centroids[pred_clusters[i]]
                curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)
    return sse