# @title: Memory management on the device
# @author: Shaurya 

from numba import vectorize
import numpy as np
from time import perf_counter

# timeit
def timeit(f):
    def decorate(*args,**kwargs):
        start=perf_counter()
        result=f(*args,**kwargs)
        end=perf_counter()
        try:
            print(f'Time taken to execute {f.__name__}: {end-start}s')
        except AttributeError as e:
            print("GPU bound problem ")
            print(e)
            print('Time taken to execute: %.2fs'%(end-start))
        return result
    return decorate

@timeit
@vectorize(['float32(float32,float32)'],target='cuda')
def add_ufunc(x,y):
    return x+y


n=1000000
x=np.arange(n).astype(np.float32)
y=2*x
add_ufunc(x,y)

# Numba can create its own GPU array object but not as fully fledged as a Cupy array
from numba import cuda
x_device=cuda.to_device(x)
y_device=cuda.to_device(y)
print(x_device)
print(y_device)
print(x_device.shape)

add_ufunc(x_device,y_device)

# we are still allocating device array for the output and copying 
# it back to the CPU
#? define the output buffer
out_device=cuda.device_array(shape=(n,),dtype=np.float32)
out_device.shape
out_device.alloc_size
add_ufunc(x_device,y_device,out=out_device)

# if we want to get it back to the CPU, we can use the copy_to_host() method
out_host=out_device.copy_to_host()
out_host.shape

#! CuPy interoperability
import cupy as cp
x_cp=cp.asarray(x)
y_cp=cp.asarray(y)

out_cp=cp.empty_like(y_cp)
out_cp.shape
y_cp.shape
x_cp.shape

x_cp.__cuda_array_interface__

add_ufunc(x_cp,y_cp,out=out_cp)