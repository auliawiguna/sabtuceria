__author__ = 'ahmadauliawiguna'

import numpy as np
import os #import library untuk akses fitur OS
from sklearn.cluster import KMeans

cwd = os.getcwd()
raw_data = cwd + "/dataset/seeds_dataset.csv"
dataset = np.loadtxt(raw_data, delimiter=",",dtype=None)

jumlah_baris = dataset.shape[0]
jumlah_kolom= dataset.shape[1]
#
X = dataset[:,0:jumlah_kolom-1] #load semua baris, load kolom ke 0 sampai dengan kolom ke JUMLAH KOLOM - 1
y = dataset[: , jumlah_kolom-1] #load kolom paling kanan


contoh = [[14.16,14.4,0.8584,5.658,3.129,3.072,5.176]]

kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

cluster_ke_berapa = kmeans.predict(contoh)[0]
print(cluster_ke_berapa)
# print(kmeans.cluster_centers_)
centers = kmeans.cluster_centers_
print(centers)
