__author__ = 'ahmadauliawiguna'

from sklearn import datasets

iris = datasets.load_iris()
print(iris.DESCR) #tampilkan deskripsi dataset
print(iris.data) #tampilkan atribut dataset
print(iris.target) #tampilkan target label dari dataset




# load_boston([return_X_y])	Load and return the boston house-prices dataset (regression).
# load_iris([return_X_y])	Load and return the iris dataset (classification).
# load_diabetes([return_X_y])	Load and return the diabetes dataset (regression).
# load_digits([n_class, return_X_y])	Load and return the digits dataset (classification).
# load_linnerud([return_X_y])	Load and return the linnerud dataset ( regression).
# load_wine([return_X_y])	Load and return the wine dataset (classification).
# load_breast_cancer([return_X_y])	Load and return the breast cancer wisconsin dataset (classification).