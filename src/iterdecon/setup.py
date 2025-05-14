# from setuptools import setup, Extension
# from Cython.Build import cythonize
# import numpy as np

# # Define the extension with compiler optimizations
# # python setup.py build_ext --inplace
# extensions = [
#     Extension(
#         "iterdecon_cython",
#         sources=["iterdecon_cython.pyx"],  # Your Cython file name
#         include_dirs=[np.get_include()],  # NumPy include directory
# #         define_macros=[('HAVE_FFTW3', '1')],
#         libraries=['fftw3'],
#         extra_compile_args=["-O3", "-ffast-math", "-march=native"],  # Aggressive optimizations
#         language="c",
#     )
# ]

# setup(
#     name="iterdecon_cython",
#     version="0.1",
#     description="Cython-optimized iterative deconvolution for seismic data",
#     author="Your Name",
#     author_email="your.email@example.com",
#     ext_modules=cythonize(
#         extensions,
#         compiler_directives={
#             'language_level': "3",
#             'boundscheck': False,
#             'wraparound': False,
#             'initializedcheck': False,
#             'cdivision': True,
#             'nonecheck': False,
#         },
#     ),
#     zip_safe=False,
# )

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# 检查是否安装了 FFTW3 库
# def check_fftw():
#     try:
#         # 尝试链接 FFTW3 库（仅检查是否存在）
#         from subprocess import run
#         result = run(["pkg-config", "--exists", "fftw3"], capture_output=True)
#         return result.returncode == 0
#     except:
#         return False

# 定义扩展模块
extensions = [
    Extension(
        "iterdecon_cython",
        sources=["iterdecon_cython.pyx"],
        include_dirs=[
            np.get_include(),  # NumPy 头文件路径
#             "/usr/local/include",  # FFTW3 可能的头文件路径（Linux/macOS）
        ],
#         library_dirs=[
#             "/usr/local/lib",  # FFTW3 可能的库路径（Linux/macOS）
#         ],
        libraries=['fftw3_threads', 'fftw3', 'pthread', "m"],
        extra_compile_args=[
            '-DFFTW_THREADS',
            "-O3", 
            "-ffast-math", 
            "-march=native",  # CPU 特定优化
            "-fopenmp",       # 如果使用 OpenMP 并行
        ],
        extra_link_args=['-lfftw3_threads', '-lfftw3', '-lpthread', '-fopenmp'],
        language="c",
    )
]

# 如果 FFTW3 未安装，提示用户
# if not check_fftw():
#     print("警告: FFTW3 库未找到，编译可能失败！")
#     print("请安装 FFTW3:")
#     print("  - Linux: sudo apt-get install libfftw3-dev")
#     print("  - macOS: brew install fftw")

# 配置 setup
setup(
    name="iterdecon_cython",
    version="0.1",
    description="Cython-optimized iterative deconvolution for seismic data",
    author="Your Name",
    author_email="your.email@example.com",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",      # Python 3 兼容
            'boundscheck': False,      # 禁用边界检查（提升性能）
            'wraparound': False,      # 禁用负数索引检查
            'initializedcheck': False, # 禁用未初始化内存检查
            'cdivision': True,        # 使用 C 风格的除法
            'nonecheck': False,       # 禁用 None 检查
        },
    ),
    zip_safe=False,
    install_requires=[
        "numpy>=1.0",  # 确保 NumPy 已安装
        "cython>=0.29", # 确保 Cython 版本足够新
    ],
)