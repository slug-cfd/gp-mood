import h5py
import sys 
import numpy as np
import matplotlib.pyplot as plt

file = sys.argv[1]
f = h5py.File(file)

for key in f.keys():
    print(f[key])
