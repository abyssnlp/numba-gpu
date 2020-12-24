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
