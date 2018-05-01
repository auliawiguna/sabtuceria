__author__ = 'ahmadauliawiguna'

import numpy as np
from sklearn import  preprocessing
import os #import library untuk akses fitur OS
from sklearn.naive_bayes import GaussianNB #load naive bayes
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

cwd = os.getcwd()
raw_data = cwd + "/dataset/heart.csv"
dataset = np.loadtxt(raw_data, delimiter=",",dtype=None)

enc = preprocessing.LabelEncoder() #deklarasikan label encoder


jumlah_baris = dataset.shape[0]
jumlah_kolom= dataset.shape[1]
#
X = dataset[:,0:jumlah_kolom-1] #load semua baris, load kolom ke 0 sampai dengan kolom ke JUMLAH KOLOM - 1
y = dataset[: , jumlah_kolom-1] #load kolom paling kanan

kf = KFold(n_splits=10) #10 Fold Cross

klasifier = GaussianNB()

urut = 1
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    klasifier.fit(X_train,y_train) #trainkan
    hasil_prediksi = klasifier.predict(X_test) #coba prediksikan pakai array
    akurasi = accuracy_score(y_test, hasil_prediksi)
    print urut,". Akurasi = ","%0.2f" % (akurasi*100)
    urut = urut +1
