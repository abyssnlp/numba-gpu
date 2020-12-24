"""
        @title: Python bindings, Calling C/C++ programs from python
        @author: Shaurya
        @date: 2020-12-24
"""

# ctypes
# Calling the function:
# Load library
# Wrap the input paramters
# specify return type of the function

# Loading library
import ctypes
import pathlib
import sys

if __name__ == "__main__":
    if sys.platform.startswith("win"):
        c_lib=ctypes.CDLL("cmult.dll")
    else:
        c_lib=ctypes.CDLL("libcmult.so")
    
    x,y=5,5.78
    c_lib.cmult.restype=ctypes.c_float
    answer=c_lib.cmult(x,ctypes.c_float(y))
    print(f"In python int: {x} float {y:.1f} returns {answer:.1f}")
    print()
