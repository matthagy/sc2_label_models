'''Script to manually classify unclassified labels
'''

from __future__ import division

import sys
import os

import numpy as np

import autopath
from paths import classification_path


classifications = np.load(classification_path)
for c in 'pui':
    mask = classifications == ord(c.upper())
    classifications[mask] = ord(c.lower())
np.save(classification_path, classifications)

