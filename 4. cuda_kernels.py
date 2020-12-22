    """ @title: CUDA Kernels with Numba
        @author:Shaurya
        @date: 2020-12-22
    """

import numpy as np
from time import perf_counter
from functools import wraps
from numba import cuda

#? Performance timer for non-return CUDA functions
def timeit(f):
    @wraps(f)
    def get_time(*args,**kwargs):
        start=perf_counter()
        f(*args,**kwargs)
        end=perf_counter()
        print(f"Time taken to execute: "+str(end-start))
    return get_time

# GPU has a grid of multiple SMs(streaming multi-processors)
# each grid has blocks on threads
# Each thread is identified by a threadidx (index for threads)
# and blockidx(index for the block)
@cuda.jit
def add_kernel(x,y,out):
    tx=cuda.threadIdx.x
    ty=cuda.blockIdx.y
    block_size=cuda.blockDim.x # num threads per block
    grid_size=cuda.gridDim.x # num blocks in the grid
    start=tx+ty*block_size
    stride=block_size*grid_size
    for i in range(start,x.shape[0],stride):
        out[i]=x[i]+y[i]


n=1000000
x=np.arange(n).astype(np.float32)
y=2*x
out=np.empty_like(x)
threads_per_block=128
blocks_per_grid=30
add_kernel[blocks_per_grid,threads_per_block](x,y,out)
out[:10]

# Using Numba helper functions
@cuda.jit
def add_kernel(x,y,out):
    start=cuda.grid(1) # 1-d thread grid
    stride=cuda.gridsize(1) # 1-d thread grid
    for i in range(start,x.shape[0],stride):
        out[i]=x[i]+y[i]

x_device=cuda.to_device(x)
y_device=cuda.to_device(y)
out_device=cuda.device_array_like(x)

start=perf_counter()
add_kernel[threads_per_block,blocks_per_grid](x,y,out)
print("Time: "+repr(perf_counter()-start)+"s")

start=perf_counter()
add_kernel[threads_per_block,blocks_per_grid](x_device,y_device,out_device)
print("Time: "+repr(perf_counter()-start)+"s")
out_device.copy_to_host()

#? Kernel Synchronization
# always synchronize with the GPU
start=perf_counter()
add_kernel[threads_per_block,blocks_per_grid](x,y,out)
print("Time: "+repr(perf_counter()-start)+"s")

cuda.synchronize()
start=perf_counter()
add_kernel[blocks_per_grid,threads_per_block](x_device,y_device,out_device)
print("Time: "+repr(perf_counter()-start)+"s")
cuda.synchronize()

cuda.synchronize()
start=perf_counter()
add_kernel[blocks_per_grid,threads_per_block](x_device,y_device,out_device);cuda.synchronize()
print("Time: "+repr(perf_counter()-start)+"s")


#? Avoid race conditions(deadlocks):Thread counter

@cuda.jit
def thread_counter_race(global_counter):
    global_counter[0]+=1

@cuda.jit
def thread_counter_safe(global_counter):
    cuda.atomic.add(global_counter,0,1) # safely add 1 to offset 0 in the array

global_counter=cuda.to_device(np.array([0],dtype=np.int32))
thread_counter_race[64,64](global_counter)
global_counter.copy_to_host()

global_counter=cuda.to_device(np.array([0],dtype=np.int32))
thread_counter_safe[64,64](global_counter)
global_counter.copy_to_host()