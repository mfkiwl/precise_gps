import numpy as np
import tensorflow as tf
import tensorflow_transform as tft


def sub_kernel(kernel, dim1, dim2):

    sub_kernel = kernel[dim1[0]:dim1[1],dim2[0]:dim2[1]]
    return sub_kernel

def pca(list_of_params, gradient):
    num_of_params = len(list_of_params[0])
    num_of_rows = len(list_of_params)

    M = np.zeros((num_of_rows, num_of_params))
    if gradient:
        for idx, param in enumerate(list_of_params[:-1]):
            M[idx] = param - list_of_params[idx+1]
    else:
        for idx, param in enumerate(list_of_params):
            M[idx] = param 
    
    return tft.pca(M, 2, dtype = tf.float64)

def eigen(M):
    values, vectors = np.linalg.eig(M)
    return values, vectors