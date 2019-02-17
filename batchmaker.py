import numpy as np
from threading import Thread, Lock
import sys
if sys.version_info >= (3, 0):
    from queue import Queue
else:
    from Queue import Queue
from math import *
from keras.utils import Progbar
from PIL import Image

class BatchMaker(object):

    def __init__(self):
        super(BatchMaker, self).__init__()


    def get_batch(self, offset, batch_size):
        X = np.random.normal((batch_size, 64, 64, 3))
        Y = np.zeros((batch_size,100))
        return (X,Y)


    def make_batch(self, q, offset, size, w_index):
        x, y = self.get_batch(offset, size)
        if x is None or y is None:
            return 0
        q.put((x, y))
        return 1


