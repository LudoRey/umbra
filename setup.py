import platform
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extra_compile_args = []
extra_link_args = []

if platform.system() == "Windows":
    extra_compile_args.append("/openmp")
elif platform.system() == "Linux":
    extra_compile_args.append("-fopenmp")
    extra_link_args.append("-fopenmp")
elif platform.system() == "Darwin":
    # macOS: use libomp from Homebrew if available, otherwise skip OpenMP
    try:
        import subprocess
        brew_prefix = subprocess.check_output(
            ["brew", "--prefix", "libomp"], text=True
        ).strip()
        extra_compile_args += ["-Xpreprocessor", "-fopenmp", f"-I{brew_prefix}/include"]
        extra_link_args += [f"-L{brew_prefix}/lib", "-lomp"]
    except Exception:
        pass  # OpenMP not available; compile without it

extensions = [
    Extension(
        "umbra.common.pyx.lut",
        sources=["umbra/common/pyx/lut.py"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    ext_modules=cythonize(extensions),
)
