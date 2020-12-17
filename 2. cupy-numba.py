# CuPy and Numba on GPU
# Author: Shaurya

import numpy as np
import cupy as cp
from time import perf_counter

# similar to numpy arrays
# cupy was created by the people behind chainer for their GPU based DL
arr=cp.arange(10)
repr(arr)
arr.dtype
arr.shape
arr.strides
arr.device

# we can move data between CPU and the GPU
arr_cpu=np.arange(10)
arr_gpu=cp.asarray(arr_cpu)
arr_gpu.device
# when we print contents of gpu array, it loads array back into cpu to show results

# numpy to cupy
cp.asarray(arr_cpu)

# cupy to numpy
cp.asnumpy(arr_gpu)

# mathematical functions
arr_gpu*2
cp.exp(-0.5*arr_gpu)
cp.linalg.norm(arr_gpu)

#! Numba on the GPU
# ufuncs and gfuncs
from numba import vectorize

@vectorize(['int64(int64,int64)'],target='cuda')
def add_ufunc(x,y):
    return x+y

a=np.array([1,2,3,4,6],dtype='int64')
b=np.array([10,20,30,40,60],dtype='int64')
b_col=b[:,np.newaxis]
c=np.arange(4*4).reshape((4,4))

start=perf_counter()
print(f'a+b= {add_ufunc(a,b)}')
print(f'Time: {perf_counter()-start}s')

import ctypes
ctypes.CDLL('libnvvm.dll')
