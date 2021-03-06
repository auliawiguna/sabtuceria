__author__ = 'ahmadauliawiguna'
from sklearn.datasets import load_files
import os #import library untuk akses fitur OS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

cwd = os.getcwd()
data_train = load_files(cwd + "/dataset/movie_reviews" ,encoding='utf8', decode_error='ignore')

#Tokenizing text
count_vect = CountVectorizer() #menghitung kata2
X_train_counts = count_vect.fit_transform(data_train.data)
tf= TfidfTransformer(use_idf=False).fit(X_train_counts) #hanya ngitung TF tanpa IDF
X_train_tf = tf.transform(X_train_counts)

klasifier = MultinomialNB().fit(X_train_tf, data_train.target)

test = ['poor Steven siegel','actually the avenger is not worth for you to watch it',
'this movie make me sleepy','Liv Tyler deserve to get oscar','sick movie!',
        'i am victim of marketing, kind of shit movie'] #kalimat yg mau dites

X_new_counts = count_vect.transform(test)
X_new_tfidf = tf.transform(X_new_counts) #hitung TF kata2 di dataset, 1 + log(jumlah kata)

array_prediksi = klasifier.predict(X_new_tfidf)

print(array_prediksi)

for doc, indeks_kategori in zip(test, array_prediksi):
    print('%r =>  %s' % (doc, data_train.target_names[indeks_kategori]))