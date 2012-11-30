
import numpy as np

import autopath
from paths import images_path, classification_path

SHUFFLE_SEED = 0xC0FFEE
TRAIN_FRAC = 0.9

images = np.load(images_path)
classifications = np.load(classification_path)

classified = classifications != -1
images = images[classified, ...]
classifications = classifications[classified]

inx = np.arange(len(images))
rnd = np.random.RandomState(SHUFFLE_SEED)
rnd.shuffle(inx)
n_train = int(np.round(len(inx) * TRAIN_FRAC))

inx_train = inx[:n_train:]
training_images = images[inx_train, ...]
training_classifications = classifications[inx_train]

inx_test = inx[n_train::]
test_images= images[inx_test, ...]
test_classifications = classifications[inx_test]

training_set = training_images, training_classifications
test_set = test_images, test_classifications
