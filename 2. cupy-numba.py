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

# with gpu cupy
start=perf_counter()
print(f'a+b= {add_ufunc(a,b)}')
print(f'Time: {perf_counter()-start}s')

# with CPU numpy
start=perf_counter()
print(f'a+b={a+b}')
print(f'Time: {perf_counter()-start}s')
# 10 times faster than GPU because there is no arithmetic intensity

import math
sqrt_2pi=np.float32((2*math.pi)**0.5)

# for gpu
@vectorize(['float32(float32,float32,float32)'],target='cuda')
def gaussian_pdf(x,mean,sigma):
    return math.exp(-0.5*((x-mean)/sigma)**2)/(sigma*sqrt_2pi)

# for CPU
import scipy.stats

# evaluate gaussian prob distribution function million times
x=np.random.uniform(-3,3,size=1000000).astype(np.float32)
mean=np.float32(0.0)
sigma=np.float32(1.0)

# calc time taken cpu
start=perf_counter()
scipy.stats.norm.pdf(x,mean,sigma)
print(repr(perf_counter()-start)+'s')
# 0.065 seconds

# calc time gpu
start=perf_counter()
gaussian_pdf(x,mean,sigma)
print(repr(perf_counter()-start)+'s')

# CUDA device functions
from numba import cuda

@cuda.jit(device=True)
def polar_to_cartesian(rho,theta):
    x=rho*math.cos(theta)
    y=rho*math.sin(theta)
    return x,y

@timeit
@vectorize(['float32(float32,float32,float32,float32)'],target='cuda')
def polar_distance(rho1,theta1,rho2,theta2):
    x1,y1=polar_to_cartesian(rho1,theta1)
    x2,y2=polar_to_cartesian(rho2,theta2)
    return ((x1-x2)**2+(y1-y2)**2)**0.5

n=1000000
rho1=np.random.uniform(0.5,1.5,size=n).astype(np.float32)
theta1=np.random.uniform(-np.pi,np.pi,size=n).astype(np.float32)
rho2=np.random.uniform(0.5,1.5,size=n).astype(np.float32)
theta2=np.random.uniform(-np.pi,np.pi,size=n).astype(np.float32)

start=perf_counter()
polar_distance(rho1,theta1,rho2,theta2)
print(repr(perf_counter()-start)+'s')


# CPU version of above
import numba
@numba.jit
def polar_to_cartesian_cpu(rho,theta):
    x=rho*math.cos(theta)
    y=rho*math.sin(theta)
    return x,y

# no device specified in vectorize; defaults to CPU
@timeit
@vectorize(['float32(float32,float32,float32,float32)'])
def polar_distance_cpu(rho1,theta1,rho2,theta2):
    x1,y1=polar_to_cartesian_cpu(rho1,theta1)
    x2,y2=polar_to_cartesian_cpu(rho2,theta2)
    return ((x1-x2)**2+(y1-y2)**2)**0.5

np.testing.assert_allclose(polar_distance(rho1,theta1,rho2,theta2),
                            polar_distance(rho1,theta1,rho2,theta2),
                            rtol=1e-7,atol=5e-7)

# timeit
def timeit(f):
    def decorate(*args,**kwargs):
        start=perf_counter()
        result=f(*args,**kwargs)
        end=perf_counter()
        try:
            print(f'Time taken to execute {f.__name__}: {end-start}s')
        except AttributeError as e:
            print(e)
            print('GPU bound probably...')
            print(f'Time taken to execute: {end-start}s')
        return result
    return decorate

# check timings for polar distance on cpu and gpu
polar_distance_cpu(rho1,theta1,rho2,theta2)
polar_distance(rho1,theta1,rho2,theta2)