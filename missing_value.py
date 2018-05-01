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
    array_baru.append(enc.fit_transform(baris) ) #convert per baris

array_baru = np.array(array_baru) #convert array biasa jadi numpy array

jumlah_baris = array_baru.shape[0]
jumlah_kolom= array_baru.shape[1]
#
X = array_baru[:,0:jumlah_kolom-1] #load semua baris, load kolom ke 0 sampai dengan kolom ke JUMLAH KOLOM - 1
y = array_baru[: , jumlah_kolom-1] #load kolom paling kanan

print(X[0:3, : ]) #coba keluarkan baris 0 sebanyak 3 baris, bandingkan dengan dataset asli, yg missing di-coding jadi 0
X = X.astype('float') #convert X dari integer ke float karena NaN bernilai float
X[X == 0] = np.NaN #replace 0 jadi NaN
print(X[0:3, : ]) #coba keluarkan baris 0 sebanyak 3 baris, bandingkan dengan dataset asli, yg missing di-coding jadi 0

imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=1) #missing valuenya dianggap 0, pakai rata2 record,
# axis 0 = hitung by kolom (all field di record terpilih), axis 1 = hitung all baris (cari rata2 field tersebut)


X = imp.fit_transform(X) #ubah missing value (0) di dataset menggunakan rata-rata
print(X[0:3, : ]) #coba keluarkan baris 0 sebanyak 3 baris, bandingkan dengan dataset asli, yg missing di-coding jadi gak 0
#diganti rata-rata
