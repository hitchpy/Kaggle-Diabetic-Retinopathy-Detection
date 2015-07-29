import numpy as np
import pandas as pd
import os

from skimage import transform
from skimage import io
from random import shuffle 


def split_index(row, batch_size):
    # split row to 6 processes for parallel processing
    res = []
    chunk = row/batch_size
    for i in range(1, chunk):
        res.append([(i-1)*batch_size, i*batch_size])
    res.append([batch_size*(chunk -1), row])
    return res

def gen_test(batch_size, fdir = '../input/processed/run-normal/test/'):
    folder = os.listdir(fdir)
    splits = split_index(len(folder), batch_size)
    for idx in splits:
        X_test = np.zeros([idx[1] - idx[0], 3, 256, 256], dtype = 'float32')
        pic_id = []* (idx[1] - idx[0])
        i = 0
        for x in folder[idx[0]:idx[1]]:
            cur = io.imread(fdir + x)
            cur = np.swapaxes(cur, 0, 2)
            X_test[i] = cur
            pic_id.append(x[:-5])
            i += 1
        X_test /= 255
        yield pic_id, X_test