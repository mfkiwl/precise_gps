import numpy as np 
import pickle 
import tensorflow_probability as tfp
import tensorflow_transform as tft
import tensorflow as tf 

def parse_trace(d, lassos, max_iter = 1500):
    """
    Used to parse marginal-log-likelihoods. all the iterations the same length.

    Args:   
        d (dict)       : dictionary of parameters (different lassos)
        lassos (list)  : lassos used during optimization
        max_iter (int) : max iteration during optimization

    Returns:
        dictionary
    """
    traces = {}
    for i in range(len(d)):
        tr = []
        current_lasso = lassos[i]
        for j in range(len(d[current_lasso])):
            current = np.array([x.numpy() for x in d[current_lasso][j]])
            last_term = current[-1]
            len_cur = len(current)
            concatted = np.concatenate((current, last_term*np.ones(max_iter-len_cur)), axis=0)
        tr.append(concatted)
        traces[i] = tr
    return traces

def parse_traceL(d, lassos, max_iter = 1500):
    """
    Used for parcing the paramaters this allows visualizing trace plots. 

    Args:   
        d (dict)       : dictionary of parameters (different lassos)
        lassos (list)  : lassos used during optimization
        max_iter (int) : max iteration during optimization

    Returns:
        dictionary
    """
    traces = {}
    for i in range(len(d)):
        traces[i] = {}
        current_lasso = lassos[i]
        for j in range(len(d[current_lasso])):
            current = np.array([x for x in d[current_lasso][j]])
            last_term = current[-1]
            len_cur = current.shape[0]
            if len_cur != max_iter:
                stack = [last_term for _ in range(max_iter-len_cur)]
                stack = np.stack(np.array(stack), axis = 0)
                concatted = np.concatenate((current, stack), axis=0)
            else:
                concatted = current
            traces[i][j] = concatted
    return traces

def parse_pickle(path):
    """
    Load a specific pickle file

    Args:
        path (string) : path to the pickle file 
    
    Returns:
        dictionary
    """
    f = open(path, "rb")
    data = pickle.load(f)
    f.close()
    return data 

def init_precision(dim):
    """
    Initializes full gaussian kernel with random precision

    Args:
        dim (int) : dimension of the precision matrix (dim x dim)

    Returns:
        Cholesky decomposition of the precision in vector format
    """
    full_L = np.random.uniform(-1,1,(dim,dim))
    P = full_L@np.transpose(full_L)

    lower_L = np.linalg.cholesky(P)
    return tfp.math.fill_triangular_inverse(lower_L)

def sub_kernel(kernel, dim1, dim2):

    sub_kernel = kernel[dim1[0]:dim1[1],dim2[0]:dim2[1]]
    return sub_kernel

def pca(list_of_params):
    num_of_params = len(list_of_params[0])
    num_of_rows = len(list_of_params)

    last_param = list_of_params[-1]

    M = np.zeros((num_of_rows, num_of_params))
    for idx, param in enumerate(list_of_params):
        M[idx] = param - last_param
    
    return tft.pca(M, 2, dtype = tf.float64)
    



