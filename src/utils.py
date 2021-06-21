import numpy as np 
import pickle 
import tensorflow_probability as tfp

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
