from os import listdir
from os.path import isfile, join
import time
import scipy.io
import numpy as np
from bf import bf 
from scipy.signal import hilbert, chirp
import matplotlib.pyplot as plt
from scipy import interpolate

from scipy import signal
from scipy import stats

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def clean(X):
    img = hilbert(X.T).T  #Mother fucker
    # print(np.amax(img))
    img= img/np.amax(img)
    img = np.abs(img)
    # print(img.shape)
    img  = 20*np.log10(img)
    return img

aspect = 0.1
file_name= 'N45_120'
Path = '../Clean/UserSessions/Unfocused_flow/'+ file_name +'/M_RFArrays/'
file = 'data.npy'
T_Path = '../Clean/UserSessions/Unfocused_flow/'+ file_name +'/M_Arrays/'
T_file = 'time.npy'
T = np.load(T_Path +T_file )
T_end = T[-1]-T[0]
# print(T[0],T[-1],T_end, 1/(T[1]-T[0]))

X = np.load(Path +file )

X = X[250:700,:]

extent = [0,T_end, X.shape[0] *1.540*0.5*(1/20),0]

X = butter_highpass_filter(X.T,1*1e6,20*1e6,order =5).T  # MUST BE ROW ARRAY 32*1000

fig2 = plt.subplot(311)
Image = plt.imshow(clean(X),cmap='gray',interpolation='none',extent = extent, aspect=aspect)
Image.set_clim(vmin= -40, vmax= 0)


u, s, vh = np.linalg.svd(X, full_matrices=False)

print(X.shape,u.shape,s.shape,vh.shape)


for element in np.round(s):
    print(element,end =",")
be = 0
ee = 2
Y =  np.dot(u[:,be:ee] * s[be:ee], vh[be:ee,:])

fig2 = plt.subplot(312)
Image = plt.imshow(clean(Y),cmap='gray',interpolation='none',extent = extent, aspect=aspect)
Image.set_clim(vmin= -10, vmax= 0)

print(Y.shape)


be = 50
ee = 300
Z =  np.dot(u[:,be:ee] * s[be:ee], vh[be:ee,:])

fig2 = plt.subplot(313)
Image = plt.imshow(clean(Z),cmap='gray',interpolation='none',extent = extent, aspect=aspect)
Image.set_clim(vmin= -40, vmax= 0)

print(Z.shape)

plt.show()


# ******************************** De-correlation
corr = []
M = clean(Y)
for i in range(0,M.shape[0]-2,2):
    print(i)
    v = stats.pearsonr(M[:,i]  ,M[:,i]   )[0]
    corr.append(v)

    v = stats.pearsonr(M[:,i]  ,M[:,i+1]   )[0]
    corr.append(v)

# print(corr)

plt.plot(corr)
plt.ylabel('some numbers')
plt.show()