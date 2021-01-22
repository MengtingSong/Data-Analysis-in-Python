import pandas as pd
from numpy import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import kmeans
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# read data from csv, select 3 features and eliminate listings with missing values
df = pd.read_csv("nyc_listings.csv")
df = df[['latitude', 'longitude', 'price']]
df.dropna()
print(df.head())

# scaling
scalar = StandardScaler()
df[df.columns] = scalar.fit_transform(df[df.columns].astype(float))
print(df.head())

numpy_loc = df[['latitude', 'longitude']].to_numpy()
numpy_price = df['price'].to_numpy()

# Cluster - Kmeans++
# Use Elbow Method to get the optimal number of clusters
sum_sqrt_error_loc = kmeans.calculate_WSS(numpy_loc, 20)
sum_sqrt_error_price = kmeans.calculate_WSS(numpy_price, 10)

plt.subplot(211)
plt.title('KMeans++ Elbow Method')
plt.ylabel('Number of Clusters')
plt.plot(range(20), sum_sqrt_error_loc)
plt.xticks(np.arange(1, 21, 1))

plt.subplot(212)
plt.plot(range(10), sum_sqrt_error_price)
plt.xticks(np.arange(1, 11, 1))
plt.xlabel('Sum of Squared Error')

plt.show()

# Set the optimal cluster number for closeness 3
# Set the optimal cluster number for expensiveness is 2
# Implement KMeans++ with the optimal cluster number
kmeans_loc_pred = KMeans(n_clusters=2).fit_predict(numpy_loc)
plt.subplot(121)
plt.scatter(numpy_loc[:, 0], numpy_loc[:, 1], c=kmeans_loc_pred)
plt.title("Closeness Cluster - KMeans++")

kmeans_price_pred = KMeans(n_clusters=2).fit_predict(numpy_price.reshape(-1, 1))
plt.subplot(122)
plt.scatter(numpy_price, len(numpy_price)*[1], c=kmeans_price_pred)
plt.title("Expensiveness Cluster - KMeans++")

plt.show()

# Cluster - Hierarchical
# show the dentrograms
plt.subplot(121)
plt.title('Dendrograms - Closeness')

dend_loc = shc.dendrogram(shc.linkage(numpy_loc, method='ward'))
plt.axhline(y=2, color='r', linestyle='--')

plt.subplot(122)
plt.title('Dendrograms - Expensiveness')
dend_price = shc.dendrogram(shc.linkage(numpy_price.reshape(-1, 1), method='ward'))
plt.axhline(y=2, color='r', linestyle='--')

plt.show()

# predict clusters
hc_loc = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
hc_loc_pred = hc_loc.fit_predict(numpy_loc)
plt.subplot(121)
plt.scatter(numpy_loc[:, 0], numpy_loc[:, 1], c=hc_loc_pred)
plt.title("Closeness Cluster - Hierarchical")

hc_price = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
hc_price_pred = hc_price.fit_predict(numpy_price.reshape(-1, 1))
plt.subplot(122)
plt.scatter(numpy_price, len(numpy_price)*[1], c=hc_price_pred)
plt.title("Expensiveness Cluster - Hierarchical")

plt.show()

# Cluster - GMM
# Use Silhouette Coefficient Score to get the optimal number of clusters
range_n_clusters = [2, 3, 4, 5, 6, 7]
silhouette_loc = []
silhouette_price = []
for n_clusters in range_n_clusters:
    gmm = GaussianMixture(n_components=n_clusters).fit(numpy_loc)
    loc_labels = gmm.predict(numpy_loc)
    silhouette_avg = silhouette_score(numpy_loc, loc_labels)
    silhouette_loc.append(silhouette_avg)

    numpy_price = numpy_price.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_clusters).fit(numpy_price)
    price_labels = gmm.predict(numpy_price)
    silhouette_avg = silhouette_score(numpy_price, price_labels)
    silhouette_price.append(silhouette_avg)

plt.subplot(211)
plt.title('GMM Silhouette Score - Closeness')
plt.ylabel('Number of Clusters')
plt.plot(range_n_clusters, silhouette_loc)
plt.subplot(212)
plt.title('GMM Silhouette Score - Expensiveness')
plt.ylabel('Number of Clusters')
plt.plot(range_n_clusters, silhouette_price)
plt.show()

# predict clusters
gmm = GaussianMixture(n_components=2).fit(numpy_loc)
labels = gmm.predict(numpy_loc)
plt.subplot(121)
plt.scatter(numpy_loc[:, 0], numpy_loc[:, 1], c=labels, s=40, cmap='viridis')
plt.title("Closeness Cluster - GMM")

numpy_price = numpy_price.reshape(-1, 1)
gmm = GaussianMixture(n_components=2).fit(numpy_price)
labels = gmm.predict(numpy_price)
plt.subplot(122)
plt.scatter(numpy_price, len(numpy_price)*[1], c=labels, s=40, cmap='viridis')
plt.title("Expensiveness Cluster - GMM")

plt.show()
