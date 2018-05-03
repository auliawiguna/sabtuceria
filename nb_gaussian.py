__author__ = 'ahmadauliawiguna'

#digunakan jika feature2 di dataset bervariasi, dan tidak continuos

import numpy as np
import os #import library untuk akses fitur OS
from sklearn.naive_bayes import GaussianNB #load naive bayes


cwd = os.getcwd()
raw_data = cwd + "/dataset/heart.csv"
dataset = np.loadtxt(raw_data, delimiter=",",dtype=None)


jumlah_kolom= dataset.shape[1]
X = dataset[:,0:jumlah_kolom-1] #load semua baris, load kolom ke 0 sampai dengan kolom ke JUMLAH KOLOM - 1
y = dataset[: , jumlah_kolom-1] #load kolom paling kanan di dataset asli

klasifier = GaussianNB()
klasifier.fit(X,y) #trainkan
hasil_prediksi = klasifier.predict([[52,1,2,120,325,0,0,172,0,0.2,1,0,3]]) #coba prediksikan pakai array
#
print(hasil_prediksi)