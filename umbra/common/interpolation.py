import numpy as np

class LinearFitInterp:
    def __init__(self, x, y):
        '''
        Parameters:
        - x : (M,) or (M,N) array.
        - y : (M,) or (M,K) array.
        '''
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if x.ndim == 1: # (M,) -> (M,1)
            x = x.reshape(-1, 1)
        self.m, _, _, _ = np.linalg.lstsq(x, y, rcond=None) # (N,) or (N,K)

    def __call__(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x, ndmin=1) # ensure singletons are read as (1,) and not (,)
        if x.ndim == 1: # (M,) -> (M,1)
            x = x.reshape(-1, 1)
        y = x @ self.m # (M,) or (M,K)
        if y.shape[0] == 1:
            return y[0]
        else:
            return y