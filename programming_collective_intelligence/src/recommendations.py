import numpy as np

def sim_distance(v1, v2):
    return np.sqrt(np.square(v1 - v2).sum())