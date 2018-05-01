__author__ = 'ahmadauliawiguna'

from sklearn.preprocessing import Binarizer
from sklearn import  datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

print(iris.data) #tampilkan atribut dataset

#
binarizer = Binarizer(threshold=4).fit(iris.data)
hasil = binarizer.transform(iris.data)

print (iris.data)
print ('------------------------------------------------------------------------------------------');
print (hasil)