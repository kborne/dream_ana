import numpy as np

def repeat(arr, n_arr):
        #print(np.unique(arr))
        return np.repeat(arr*1e12, n_arr.astype(int))
