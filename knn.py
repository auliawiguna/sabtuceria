__author__ = 'ahmadauliawiguna'

import numpy as np
import os #import library untuk akses fitur OS
from sklearn.neighbors import KNeighborsClassifier

cwd = os.getcwd()
raw_data = cwd + "/dataset/heart.csv"
dataset = np.loadtxt(raw_data, delimiter=",",dtype=None)

jumlah_baris = dataset.shape[0]
jumlah_kolom= dataset.shape[1]
#
X = dataset[:,0:jumlah_kolom-1] #load semua baris, load kolom ke 0 sampai dengan kolom ke JUMLAH KOLOM - 1
y = dataset[: , jumlah_kolom-1] #load kolom paling kanan

#bisa entropy juga bisa gini
klasifier = KNeighborsClassifier(n_neighbors=1)
klasifier.fit(X,y)

contoh = [[70,1,4,130,322,0,2,109,0,2.4,2,3,3]]
hasil_prediksi = klasifier.predict(contoh) #coba prediksikan pakai array
print(hasil_prediksi)

