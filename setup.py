import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if sys.platform == "win32":
    cxx_compile_args = ["/Zc:preprocessor", "/std:c++20", "/wd5249"]
    nvcc_compile_args = ["-std=c++20", "-Xcompiler=/Zc:preprocessor,/std:c++20"]
else:
    cxx_compile_args = ["-std=c++20", "-Wno-3189"]
    nvcc_compile_args = ["-std=c++20"]

setup(
    name="pycuda_extension",
    packages=[],
    py_modules=[],
    ext_modules=[
        CUDAExtension(
            "pycuda_extension",
            [
                "image_blur.cpp",
                "image_blur_kernel.cu",
            ],
            extra_compile_args={
                "cxx": cxx_compile_args,
                "nvcc": nvcc_compile_args,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
