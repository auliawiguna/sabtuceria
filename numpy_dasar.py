__author__ = 'ahmadauliawiguna'

import numpy as np

a = np.array([1, 2, 3])  #contoh bikin array
#print(a)
#print(a.shape) #dimensinya


b = np.array([[1,2,3],[4,5,6]])    #buat arrau berukuran 2 x 3
#print(b)


a = np.zeros((3,5)) #array 3 x 5 isinya nol semua
#print(a)

a = np.zeros((3,5)) #array 3 x 5 isinya satu semua
#print(a)


a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12],[13,14,15,16],[17,18,19,20]])
print(a)

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
#param [mulai dari baris berapa:ambil berapa baris, dari kolom berapa dari kiri mulai dari 0:sampai kolom ke berapa dari kiri mulai dari 1]
b = a[0:3, 0:3]
print(a.shape)
# print(a[: , a.shape[1]-1:])
print(a[3:5 , 1:3])

print(a[: , 0:a.shape[1]-1])

#arange(x) , menghasilkan numPy array 0 sampai x
# print(np.arange(10))
#ambil elemen ke 2,2,3,3,4 dari a
# print(a[np.arange(a.shape[0]),np.array([2,2,3,3,1])])
