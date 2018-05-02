__author__ = 'ahmadauliawiguna'

import numpy as np
import os #import library untuk akses fitur OS
import graphviz
from sklearn import tree

cwd = os.getcwd()
raw_data = cwd + "/dataset/heart.csv"
dataset = np.loadtxt(raw_data, delimiter=",",dtype=None)

jumlah_baris = dataset.shape[0]
jumlah_kolom= dataset.shape[1]
#
X = dataset[:,0:jumlah_kolom-1] #load semua baris, load kolom ke 0 sampai dengan kolom ke JUMLAH KOLOM - 1
y = dataset[: , jumlah_kolom-1] #load kolom paling kanan

#bisa entropy juga bisa gini
klasifier = tree.DecisionTreeClassifier(criterion="entropy")
klasifier = klasifier.fit(X,y)

dot_data = tree.export_graphviz(klasifier, out_file=None,
                         feature_names=["age","sex","cp","trestbps","kolesterol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"],
                         class_names=['1','-1'],
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph.format = 'dot'
graph.render()

#cari Source.gv, visualkan di https://dreampuf.github.io/GraphvizOnline/
#laptop saya kaga ada dot reader ternyata hahaha
