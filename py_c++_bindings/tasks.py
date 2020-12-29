"""
    @title: Tasks Invocation
    @author: Shaurya
    @date: 2020-12-24

"""

import invoke
import pathlib
import sys
import os
import shutil
import re
import glob
import cffi

on_win=sys.platform.startswith("win")

def print_banner(msg):
    print("==================================================")
    print("={}".format(msg))

@invoke.task()
def test_ctypes(c):
    print_banner("cmult test")
    if on_win:
        invoke.run("python c_python_bindings.py")
    else:
        pass

@invoke.task()
def build_cffi(c):
    print_banner("Building CFFI module")
    ffi=cffi.FFI()
    this_dir=pathlib.Path().resolve()
    # use the same header file for cmult
    with open('cmult.h') as f:
        # cffi does not support pre-processor directives
        lns=f.read().splitlines()
        filtered=filter(lambda line: not re.match(r' *#',line),lns)
        filtered=map(lambda ln:ln.replace("EXPORT_SYMBOL ",""),filtered)
        ffi.cdef(str('\n').join(filtered))
    # describe source file that CFFI will generate
    ffi.set_source(
        "cffi_test",
        '#include "cmult.h"',
        libraries=["cmult"],
        library_dirs=[this_dir.as_posix()],
        extra_link_args=["-Wl,-rpath,."],
    )
    ffi.compile()
    print("Done!")

# pybind11 cpp invoke
invoke.run(
    "g++ -O3 -Wall -Werror -shared -std=c++11 -fPIC cppmult.cpp"
    " -o libcppmult.so"
)
invoke.run(
    "g++ -O3 -Wall -Werror -shared -std=c++11 -fPIC"
    "`python -m pybind11 includes`"
    "-I /usr/include/python"
)