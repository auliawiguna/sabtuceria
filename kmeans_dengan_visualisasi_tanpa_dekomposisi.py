



# HAHA
__author__ = 'ahmadauliawiguna'

import numpy as np
import os #import library untuk akses fitur OS
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pylab as pl

cwd = os.getcwd()
raw_data = cwd + "/dataset/seeds_dataset.csv"
dataset = np.loadtxt(raw_data, delimiter=",",dtype=None)

jumlah_baris = dataset.shape[0]
jumlah_kolom= dataset.shape[1]
#
X = dataset[:,0:jumlah_kolom-1] #load semua baris, load kolom ke 0 sampai dengan kolom ke JUMLAH KOLOM - 1
y = dataset[: , jumlah_kolom-1] #load kolom paling kanan



# AWAL
pca = PCA(n_components=7).fit(X)
pca_2d = pca.transform(X)
kmeans = KMeans(n_clusters=4, random_state=111)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

contoh = [[14.16,14.4,0.8584,5.658,3.129,3.072,5.176]]
#contoh dikelompokkan di cluster ke berapa
cluster_ke_berapa = kmeans.predict(contoh)[0]
#tampilkan titik pusatnya
centers = kmeans.cluster_centers_
n_clusters = len(centers)


pl.figure('K-means dengan 4 clusters')
pl.scatter(X[:, 0], X[:, 1], c=y_kmeans)
pl.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
pl.scatter(contoh[0][0], contoh[0][1], c="blue",s=350,alpha=0.8,marker="*");
pl.show()



