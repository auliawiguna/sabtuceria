__author__ = 'ahmadauliawiguna'

import numpy as np
from sklearn import  preprocessing
import os #import library untuk akses fitur OS

cwd = os.getcwd()
raw_data = cwd + "/dataset/audiology.standardized.csv"
dataset = np.genfromtxt(raw_data, delimiter=",",dtype=None)

enc = preprocessing.LabelEncoder() #deklarasikan label encoder

#looping dataset
array_baru=[] #menampung array hasil labelencoder
for baris in dataset :
    array_baru.append(enc.fit_transform(baris)) #convert per baris

array_baru = np.array(array_baru) #convert array biasa jadi numpy array

jumlah_baris = array_baru.shape[0]
jumlah_kolom= array_baru.shape[1]
#
X = array_baru[:,0:jumlah_kolom-1] #load semua baris, load kolom ke 0 sampai dengan kolom ke JUMLAH KOLOM - 1
y = array_baru[: , jumlah_kolom-1] #load kolom paling kanan

print(X[0:3, : ]) #coba keluarkan baris 0 sebanyak 3 baris

