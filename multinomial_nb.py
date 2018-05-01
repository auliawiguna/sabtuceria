__author__ = 'ahmadauliawiguna'

import numpy as np
from sklearn import  preprocessing
import os #import library untuk akses fitur OS
from sklearn.naive_bayes import MultinomialNB #load naive bayes



cwd = os.getcwd()
raw_data = cwd + "/dataset/heart.csv"
dataset = np.loadtxt(raw_data, delimiter=",",dtype=None)

enc = preprocessing.LabelEncoder() #deklarasikan label encoder


jumlah_baris = dataset.shape[0]
jumlah_kolom= dataset.shape[1]
#
X = dataset[:,0:jumlah_kolom-1] #load semua baris, load kolom ke 0 sampai dengan kolom ke JUMLAH KOLOM - 1
y = dataset[: , jumlah_kolom-1] #load kolom paling kanan

klasifier = MultinomialNB()
klasifier.fit(X,y) #trainkan
hasil_prediksi = klasifier.predict([[70,1,4,130,322,0,2,109,0,2.4,2,3,3]]) #coba prediksikan pakai array

print(hasil_prediksi)