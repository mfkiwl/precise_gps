import numpy as np 
import json 

def parse_trace(d, lassos, max_iter = 1500):
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
        tr = np.array(tr).mean(axis = 0)
        traces[i] = tr
    return traces

def parse_traceL(d, lassos, max_iter = 1500):
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
