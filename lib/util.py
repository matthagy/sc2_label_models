
from __future__ import division

import Image
import numpy as np


def convert_gray_scale(X):
    return np.array([np.asarray(Image.fromarray(xi).convert('L'))
                     for xi in X])

def flatten(X):
    return X.reshape(X.shape[0], X.shape[1]*X.shape[2])
