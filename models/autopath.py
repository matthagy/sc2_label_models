'''Install lib directory in python path
'''

import sys
import os.path as pth

modeldir = pth.dirname(pth.abspath(__file__))
rootdir = pth.dirname(modeldir)
libdir = pth.join(rootdir, 'lib')
assert pth.isdir(libdir)

sys.path.append(libdir)
