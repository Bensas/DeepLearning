import numpy as np
def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1.0 - np.tanh(x)**2

def sigmoide(x):
    return 1/(1+np.float128(-x))

def dsigmoide(x):
    s = 1/(1+np.float128(-x))
    return s * (1-s)