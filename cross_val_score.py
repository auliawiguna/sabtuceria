__author__ = 'ahmadauliawiguna'

import numpy as np
from sklearn import  preprocessing
import os #import library untuk akses fitur OS
from sklearn.naive_bayes import GaussianNB #load naive bayes
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

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
klasifier.fit(X,y) #trainkan

scores = cross_val_score(klasifier, X, y, cv=10)

print(scores)
print "Akurasi rata-rata : %0.2f (+/- %0.2f)" % (scores.mean()*100, scores.std() * 2)