
import sys
import os.path as pth


libdir = pth.dirname(pth.abspath(__file__))
rootdir = pth.dirname(libdir)
datadir = pth.join(rootdir, 'data')

images_path = pth.join(datadir, 'images.npy')
classification_path = pth.join(datadir, 'classifications.npy')
