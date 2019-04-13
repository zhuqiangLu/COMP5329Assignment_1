import numpy as np
import h5py
import matplotlib.pyplot as pl

with h5py.File('Assignment-1-Dataset/train_128.h5','r') as H:
    data = np.copy(H['data'])


pic = data[0]
pl.imshow(pic.reshape((16,8)))
pl.show()
