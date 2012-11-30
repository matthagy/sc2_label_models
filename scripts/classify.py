'''Script to manually classify unclassified labels
'''

from __future__ import division

import sys
import os

import numpy as np
import matplotlib.pyplot as plt

import autopath
from paths import images_path, classification_path


def main():
    global classifications
    images = np.load(images_path)
    if not os.path.exists(classification_path):
        classifications = np.zeros(len(images), int) - 1
    else:
        classifications = np.load(classification_path)

    unclassified = classifications == -1


    print '%d unclassified' % unclassified.sum()

    if not unclassified.any():
        return

    fig = plt.figure(1)

    indices = unclassified.nonzero()[0]
    np.random.shuffle(indices)

    for i,index in enumerate(indices):
        fig.clf()
        axu = fig.add_subplot(211)
        ax = fig.add_subplot(212)
        axu.imshow((classifications != -1)[np.newaxis, ::],
                   aspect='auto', cmap=plt.cm.jet)
        axu.text(0.01, 0.95, '%05d' % (sum(classifications != -1),),
                 horizontalalignment='left',
                 verticalalignment='top',
                 transform = axu.transAxes,
                 backgroundcolor='white',
                 size=14,
                 color='k')

        ax.imshow(images[index], aspect='auto')
        fig.canvas.draw()
        plt.show()
        try:
            label = raw_input('%05d ? ' % (index,)).lower().strip()
        except EOFError:
            break
        if label == 'q' or not label:
            break
        assert label in ('p','u','i')
        classifications[index] = ord(label)

        if not i%5:
            print 'save'
            if os.path.exists(classification_path):
                os.rename(classification_path, classification_path + '.back')
            np.save(classification_path, classifications)


__name__ == '__main__' and main()
