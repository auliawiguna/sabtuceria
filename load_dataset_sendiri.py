__author__ = 'ahmadauliawiguna'

import numpy as np
import os #import library untuk akses fitur OS
cwd = os.getcwd()

raw_data = cwd + "/dataset/heart.csv"
dataset = np.loadtxt(raw_data, delimiter=",")

try:
    #kalau datasetmu delimiternya koma
    dataset = np.loadtxt(raw_data, delimiter=",")
except:
    #kalau datasetmu bukan koma pasti error, coba titik koma deh
    dataset = np.loadtxt(raw_data, delimiter=";")

jumlah_baris = dataset.shape[0]
jumlah_kolom= dataset.shape[1]

X = dataset[:,0:jumlah_kolom-1] #load semua baris, load kolom ke 0 sampai dengan kolom ke JUMLAH KOLOM - 1
y = dataset[: , jumlah_kolom-1]

print(y)