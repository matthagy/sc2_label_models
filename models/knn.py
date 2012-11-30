
from __future__ import division

import cPickle as pickle

import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold

import autopath
from datasets import training_set, test_set
from util import convert_gray_scale, flatten

Xr,Yr = training_set
Xe,Ye = test_set

Xr = flatten(convert_gray_scale(Xr))
Xe = flatten(convert_gray_scale(Xe))

k_fold = 8

ks = [1, 2, 3, 5, 8, 12, 20, 25, 50]
xacc = []
for k in ks:
    acc = []
    for train_inx, valid_inx in StratifiedKFold(Yr, k_fold):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(Xr[train_inx], Yr[train_inx])
        score = np.mean(knn.predict(Xr[valid_inx]) == Yr[valid_inx])
        acc.append(score)
        print k, score
    xacc.append([np.mean(acc), np.std(acc)])

m,s = np.array(xacc).T

plt.figure(2)
plt.clf()
plt.errorbar(ks, m, s)
plt.xscale('log')
plt.draw()
plt.show()

knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(Xr, Yr)
Yp = knn.predict(Xe)
print np.mean(Yp == Ye)
