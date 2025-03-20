import numpy as np
import time
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score 

#load vectors
#document_vectors = np.loadtxt('vectors.txt', delimiter=' ')

#tsne
t0 = time.time()
#tsne = TSNE(n_components=2)
#X_tsne = tsne.fit_transform(document_vectors)
X_tsne = np.loadtxt('vectors_tsne.txt', delimiter=' ')

t1 = time.time()
total = t1-t0
print(f'vectors tsned!: {total} seconds')

#kmeans
kmeans = KMeans(n_clusters=7, n_init="auto")
X_kmeans = kmeans.fit(X_tsne)
t0 = time.time()



t1 = time.time()
total = t1-t0
print(f'vectors kmeaned!: {total} seconds')


#plot
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=X_kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], zorder=10,c='black')
plt.show()


t1 = time.time()
total = t1-t0
print(f'vectors saved!: {total} seconds')

#100_000/2-20
#vectors loaded!: 0.2694211006164551 seconds
#vectors tsned, kmeaned and plotted!: 1775.7663662433624 seconds

#100_000/20-40
#vectors loaded!: 0.2723226547241211 seconds
#vectors tsned!: 156.2063012123108 seconds
#vectors kmeaned!: 4.76837158203125e-07 seconds
#vectors saved!: 3.0994415283203125e-05 seconds