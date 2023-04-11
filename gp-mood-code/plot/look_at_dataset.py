import h5py
import sys 
import numpy as np
import matplotlib.pyplot as plt

def sum_char(input):
    r=''
    for char in input:
        r=r+str(char)[-2]
    return r

file = sys.argv[1]
f = h5py.File(file)

print("problem", sum_char(f["problem"]))
print("method", sum_char(f["method"]))
print("CFL", sum_char(f["CFL"]))
print("lf", f["lf"][0])
print("nf", f["nf"][0])
print("n_overwrite", f["n_overwrite"][0])
print("size", f["inputs"].shape[0])
