import sys, os
filepath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(filepath)
from eq_opencl import clHistEq

clHistEq.getInstance()