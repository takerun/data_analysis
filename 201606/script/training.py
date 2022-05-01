# coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Constants
# Train
TRAIN_TARGET = "../future-temparture-prediction/Temperature_Train_Target.dat.tsv"
# Result
TEST_TARGET = "../result/Temperature_Test_Target.dat"

### Function
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn import grid_search
parameters = {'alpha':[float(i)/100 for i in xrange(0,101)]}
clf = grid_search.GridSearchCV(Ridge(), parameters, cv=10, scoring='mean_squared_error')

clf.fit(X, y)

clf.predict(np.identity(11))
