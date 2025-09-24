import numpy as np

def duck_goose_arr(arr, n_arr, ec, ec_01):
    ec_repeat = np.repeat(ec, n_arr.astype(int))
    inds = ec_repeat == ec_01
    return arr[inds]


def duck_goose_arr1(arr, ec, ec_01):
    inds = ec == ec_01
    return arr[inds]
    

def atm(line, xgmd, ec, xgmd_min, ec_01):
    if xgmd[-1]>xgmd_min and ec[-1]==ec_01:
        return line
