import numpy as np

def one_hot(size, target):
    one_hot_vec = np.zeros(size)
    one_hot_vec[target] = 1
    return one_hot_vec