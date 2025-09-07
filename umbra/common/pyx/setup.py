from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize("lut.py"),
    include_dirs=[np.get_include()]
)

# To run this : python setup.py build_ext --inplace