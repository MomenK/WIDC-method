from scipy.signal import hilbert, chirp
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# from matplotlib.backend_bases import key_press_handler
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import numpy as np
from os import listdir
from os.path import isfile, join
from scipy import signal
import tkinter as tk
from scipy import interpolate
from scipy import stats
from scipy.ndimage.filters import uniform_filter1d
import math

def declutter(x,window):
    slowX = uniform_filter1d(x, size=window,axis=1)
    print("SlowX")
    print(slowX.shape)
    return x-slowX,slowX


X = np.zeros((10,100))
t = range(0,100)
for i in range(0,10):
    X[i] = np.sin(t)

f = 10

for i in range(0,10):
    X[i] = X[i]+ t

    
    # x[row] = x[row]*row

x,slowX = declutter(X,20)
print(X)
print("SlowX")
print(slowX)

plt.figure(figsize=(15,4))
plt.subplot(311)
plt.imshow(X,aspect='auto',cmap='gray')
plt.subplot(312)
plt.imshow(slowX,aspect='auto',cmap='gray')
plt.subplot(313)
plt.imshow(x,aspect='auto',cmap='gray')

S =  9

plt.figure(figsize=(15,4))
plt.plot(t,np.sin(t))
plt.plot(X[S])
plt.plot(slowX[S])

plt.show()