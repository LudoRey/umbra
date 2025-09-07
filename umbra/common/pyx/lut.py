# cython: language_level=3
# distutils: extra_compile_args=/openmp

import numpy as np
import cython
from cython.parallel import prange #type: ignore

img_type = cython.fused_type(cython.float, cython.double)
values_type = cython.fused_type(cython.uchar, cython.ushort, cython.float) 

@cython.boundscheck(False)
@cython.wraparound(False)
def apply_lut_rgb(img: img_type[:,:,::1], lut: values_type[::1]) -> values_type: #type: ignore

    h: cython.Py_ssize_t = img.shape[0]
    w: cython.Py_ssize_t = img.shape[1]
    c: cython.Py_ssize_t = img.shape[2]
    lut_resolution: cython.float = len(lut)-1

    # Initialize values (nd.array) and get a view of it
    if values_type is cython.uchar:
        dtype = np.uint8
    elif values_type is cython.ushort:
        dtype = np.uint16
    elif values_type is cython.float:
        dtype = np.float32
    values = np.zeros((h,w,c), dtype=dtype) # same dtype as lut
    values_view: values_type[:,:,::1] = values # type: ignore

    # Apply the LUT
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    k: cython.Py_ssize_t
    key: cython.Py_ssize_t
    for i in prange(h, nogil=True):
        for j in range(w):
            for k in range(c):
                key = cython.cast(cython.Py_ssize_t, img[i,j,k]*lut_resolution)
                values_view[i,j,k] = lut[key]

    return values


@cython.boundscheck(False)
@cython.wraparound(False)
def apply_lut_grayscale(img: img_type[:,::1], lut: values_type[::1]) -> values_type: #type: ignore

    h: cython.Py_ssize_t = img.shape[0]
    w: cython.Py_ssize_t = img.shape[1]
    lut_resolution: cython.float = len(lut)-1

    # Initialize values (nd.array) and get a view of it
    if values_type is cython.uchar:
        dtype = np.uint8
    elif values_type is cython.ushort:
        dtype = np.uint16
    elif values_type is cython.float:
        dtype = np.float32
    values = np.zeros((h,w), dtype=dtype) # same dtype as lut
    values_view: values_type[:,::1] = values # type: ignore

    # Apply the LUT
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    key: cython.Py_ssize_t
    for i in prange(h, nogil=True):
        for j in range(w):
            key = cython.cast(cython.Py_ssize_t, img[i,j]*lut_resolution)
            values_view[i,j] = lut[key]

    return values