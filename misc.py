import numpy as np

def nparray_2_dict(A_array):
    A_dict = {int(row[0]) : row[1] for row in A_array}
    return A_dict

def dict_2_nparray(A_dict):
    num_data = len(A_dict)
    A_array = np.zeros([num_data, 2], dtype=float)
    for idx, key in enumerate(A_dict.keys()):
        A_array[idx, 0] = key
        A_array[idx, 1] = A_dict[key]

    return A_array

def load_array(path):
    B_array = np.loadtxt(path, delimiter=',')
    return B_array.reshape([-1, 2])

def save_array(path, A_array):
    np.savetxt(path, A_array, delimiter=',', fmt='%d, %.16e' )

