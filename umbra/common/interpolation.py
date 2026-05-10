from collections.abc import Sequence

import numpy as np


class LinearFitInterp:
    def __init__(self, x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray):
        '''
        Parameters:
        - x : (M,) or (M,N) array.
        - y : (M,) or (M,K) array.
        '''
        x = np.asarray(x)
        y = np.asarray(y)
        if x.ndim == 1: # (M,) -> (M,1)
            x = x.reshape(-1, 1)
        self.m, _, _, _ = np.linalg.lstsq(x, y, rcond=None) # (N,) or (N,K)

    def __call__(self, x: float | Sequence[float] | np.ndarray) -> np.ndarray:
        x = np.array(x, ndmin=1) # ensure singletons are read as (1,) and not (,)
        if x.ndim == 1: # (M,) -> (M,1)
            x = x.reshape(-1, 1)
        y = x @ self.m # (M,) or (M,K)
        if y.shape[0] == 1:
            return y[0]
        else:
            return y
