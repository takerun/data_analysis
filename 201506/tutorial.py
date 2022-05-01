#coding: UTF-8

import os
from sklearn.datasets import load_svmlight_file
DATA_TRAIN_PATH = './text-classification-002/data-train.dat'
N_FEATURES = 1881
X, y = load_svmlight_file(DATA_TRAIN_PATH, n_features=N_FEATURES, dtype=int, zero_based=True)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=1.0)
clf.fit(X_train, y_train)
from sklearn.svm import SVC
clf = SVC(C=1.0)
clf.fit(X,y)
DATA_TEST_PATH = './text-classification-002/data-test.dat'
X_test, y_test = load_svmlight_file(DATA_TEST_PATH, n_features=N_FEATURES, dtype=int, zero_based=True)
print(clf.predict(X_test))
