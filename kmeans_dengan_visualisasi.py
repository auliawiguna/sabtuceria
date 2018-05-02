__author__ = 'ahmadauliawiguna'

import numpy as np
import os #import library untuk akses fitur OS
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_argmin

cwd = os.getcwd()
raw_data = cwd + "/dataset/seeds_dataset.csv"
dataset = np.loadtxt(raw_data, delimiter=",",dtype=None)

jumlah_baris = dataset.shape[0]
jumlah_kolom= dataset.shape[1]
#
X = dataset[:,0:jumlah_kolom-1] #load semua baris, load kolom ke 0 sampai dengan kolom ke JUMLAH KOLOM - 1
y = dataset[: , jumlah_kolom-1] #load kolom paling kanan


contoh = [[14.16,14.4,0.8584,5.658,3.129,3.072,5.176]]

kmeans = KMeans(n_clusters=5, random_state=0).fit(X)

cluster_ke_berapa = kmeans.predict(contoh)[0]

# print(kmeans.cluster_centers_)
centers = kmeans.cluster_centers_
n_clusters = len(centers)

fig = plt.figure(figsize=(8, 3))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ['#4EACC5', '#FF9C34', '#4E9A06','#45FFAA']

k_means_cluster_centers = np.sort(kmeans.cluster_centers_, axis=0)
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

# KMeans
ax = fig.add_subplot(1, 3, 1)
urut = 0
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    print(cluster_center)
    #Tandai pusat cluster
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)


    if(cluster_ke_berapa == urut):
        #tandai contoh warna merah
        ax.plot(contoh[0][0], contoh[0][1], 'o', markerfacecolor="#ff0000",
                markeredgecolor='k', markersize=11,marker='x')

    urut == urut+1
ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())

plt.show()

