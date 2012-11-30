
from __future__ import division

import cPickle as pickle

import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble.forest import RandomForestClassifier

import autopath
from datasets import training_set, test_set
from util import convert_gray_scale, flatten


Xr,Yr = training_set
Xe,Ye = test_set

Xr = flatten(convert_gray_scale(Xr))
Xe = flatten(convert_gray_scale(Xe))

rf = RandomForestClassifier(n_estimators=100, verbose=3, oob_score=True, compute_importances=True)
rf.fit(Xr, Yr)

Yp = rf.predict(Xe)
print np.mean(Yp == Ye)

Ypp = rf.predict_proba(Xe).max(axis=1)

plt.figure(1)
plt.clf()
plt.hist(Ypp[Yp == Ye], 50, color='b', normed=True, alpha=0.4,
         label='classified')
plt.hist(Ypp[Yp != Ye], 50, color='r', normed=True, alpha=0.4,
         label='misclassified')
plt.legend(loc='upper left')
plt.draw()
plt.show()

plt.figure(3)
plt.clf()
cannot_classify = Ypp < 0.95

n = 0.01 * float(len(Yp))
correct = sum(Ye[~cannot_classify] == Yp[~cannot_classify]) / n
incorrect = sum(Ye[~cannot_classify] != Yp[~cannot_classify]) / n
not_classified = sum(cannot_classify) / n
x = [0,1,2]
plt.bar(np.array(x) + 0.1, [correct, incorrect, not_classified],
        color=['g', 'r', 'b'])
plt.gca().set_xticks(np.array(x) + 0.5)
plt.gca().set_xticklabels(['correct', 'incorrect', 'not classified'],
                          rotation=0)
plt.ylabel('Percentage')
plt.title('Random Forest Classification of Tab Labels')

ax = plt.gca()
for xi,yi in zip(x, [correct, incorrect, not_classified]):
    ax.text(xi + 0.5, yi + 2, '%.1f %%' % (yi,),
            horizontalalignment='center',
            verticalalignment='bottom')

plt.draw()
plt.show()
plt.savefig('random_forest.png')

with open('random_forest.p', 'w') as fp:
    pickle.dump(rf, fp, pickle.HIGHEST_PROTOCOL)
