__author__ = 'ahmadauliawiguna'

#digunakan jika feature2 di dataset bervariasi, dan tidak continuos

import numpy as np
from sklearn import  preprocessing
import os #import library untuk akses fitur OS
from sklearn.naive_bayes import MultinomialNB #load naive bayes


cwd = os.getcwd()
raw_data = cwd + "/dataset/heart.csv"
dataset = np.loadtxt(raw_data, delimiter=",",dtype=None)

enc = preprocessing.LabelEncoder() #deklarasikan label encoder

#looping dataset
X=[] #menampung array hasil labelencoder

features = dataset[:,0:dataset.shape[1]-1] #pisahkan feature

for baris in features :
    X.append(enc.fit_transform(baris)) #convert per baris, diskretisasikan

X = np.array(X) #convert array biasa jadi numpy array

jumlah_kolom= dataset.shape[1]
y = dataset[: , jumlah_kolom-1] #load kolom paling kanan di dataset asli


klasifier = MultinomialNB()
klasifier.fit(X,y) #trainkan
hasil_prediksi = klasifier.predict([[6,1,4,7,9,1,4,8,3,2,3,3,5]]) #coba prediksikan pakai array
#
print(hasil_prediksi)