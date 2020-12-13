# @title: CPU bound examples of NUMBA and numpy
# @author: Shaurya
# @date: 13/12/2020

import numpy as np
from numba import jit
import math
import time
import os
# mutlti-dimensional array type
x=np.zeros(shape=(2,3),dtype=np.float64)

# string representation
repr(x)

print(x.dtype)
print(x.shape)
print(x.strides) # strides: byte jumps to traverse along an axis (x,y,z..)
# strides are required as they reflect how numpy stores arrays in a data buffer


y=np.ones(shape=(2,3,4),dtype=np.int32)
print(y)
print(y.strides)


np.arange(20)
np.arange(20).strides

# Array operations (multi-dimensional)
a=np.array([1,2,3,5])
b=np.array([2,3,5,6])
np.add(a,b)

# newaxis
b[:,np.newaxis].shape
np.expand_dims(b,1).shape
# igual

# outside of numpy array primitives (which runs on c compiled code)
# the python operations on arrays become very slow

# numpy roll
# shift array
np.roll(b,2)
np.roll(b,-2)

#! NUMBA
# functional, just-in-time, type-specializing

@jit
def hypot(x,y):
    x=abs(x)
    y=abs(y)
    t=min(x,y)
    x=max(x,y)
    t=t/x
    return x*math.sqrt(1+t*t)

start=time.time()
print(hypot(3,5))
end=time.time()
print("Time: %.5f"%(end-start))
