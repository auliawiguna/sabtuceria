__author__ = 'ahmadauliawiguna'

#digunakan jika feature2 di dataset bervariasi, dan tidak continuos

import numpy as np
import os #import library untuk akses fitur OS
from sklearn.naive_bayes import BernoulliNB #load naive bayes


cwd = os.getcwd()
raw_data = cwd + "/dataset/heart.csv"
dataset = np.loadtxt(raw_data, delimiter=",",dtype=None)


jumlah_kolom= dataset.shape[1]

X = np.random.randint(2, size=(100, 8))
y =  np.random.randint(2,4,100)

klasifier = BernoulliNB()
klasifier.fit(X,y) #trainkan
hasil_prediksi = klasifier.predict([[1,0,1,1,1,0,1,1]]) #coba prediksikan pakai array
# # #
print(hasil_prediksi)