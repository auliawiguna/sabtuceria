__author__ = 'ahmadauliawiguna'

import numpy as np
from sklearn import  preprocessing
import os #import library untuk akses fitur OS
from sklearn.naive_bayes import GaussianNB #load naive bayes
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

cwd = os.getcwd()
raw_data = cwd + "/dataset/heart.csv"
dataset = np.loadtxt(raw_data, delimiter=",",dtype=None)

enc = preprocessing.LabelEncoder() #deklarasikan label encoder


jumlah_baris = dataset.shape[0]
jumlah_kolom= dataset.shape[1]
#
X = dataset[:,0:jumlah_kolom-1] #load semua baris, load kolom ke 0 sampai dengan kolom ke JUMLAH KOLOM - 1
y = dataset[: , jumlah_kolom-1] #load kolom paling kanan


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #data training 9, testing 1



klasifier = GaussianNB()
klasifier.fit(X_train,y_train) #trainkan
hasil_prediksi = klasifier.predict(X_test) #coba prediksikan pakai array
akurasi = accuracy_score(y_test, hasil_prediksi)
print(hasil_prediksi)
print "Akurasi = ","%0.2f" % (akurasi*100)