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


# load file ****************************************************************************************
from os import listdir
from os.path import isfile, join
import time

fold_name= 'Mice_8_B'
Path = '../Clean/UserSessions/'+ fold_name +'/RFArrays/'
file = 'B_80,0_0,0.npy'

X = np.load(Path +file)
X = butter_highpass_filter(X.T,1*1e6,20*1e6,order =5).T  # MUST BE ROW ARRAY 32*1000

print(X.shape)
fig = plt.figure(figsize =(7,7) )# 

scale = 5

max_depth = 30
start_depth = 10
begin = round(start_depth*scale/0.05 )
aspect = 0.5
element = 16
# RF ****************************************************************************************

res = 5e-5
steps = round((max_depth/0.05))
print("hhhhhhhhhhh")
print(steps)
z_axis = np.arange(0,steps*scale)*res/scale

z_axis = z_axis[begin:-1]

fig1 = plt.subplot(133)

X_RF = X[:,element]
print(X_RF.shape)

# X_RF = X_RF[0: round(  max_depth/(1.540*0.5*(1/20)) )+1,:]
grid = np.arange(0,X_RF.shape[0])*1.540*0.5*(1/20)*1e-3
print(grid.shape,grid[0],grid[-1])

f = interpolate.interp1d(grid, X_RF,fill_value=0,bounds_error=False,kind='linear')
X_RF  = f(z_axis).reshape(-1,1)

print(z_axis,grid)

print(X_RF.shape)
Image = plt.imshow(clean(X_RF),cmap='gray',interpolation='none', extent=[-1,1,max_depth,start_depth],aspect=aspect)


# 32 BF ****************************************************************************************
pitch = 3e-04
f_resample = 20e6
c = 1540
tstart = 0
terror = 0
scale = 5
res = 5e-5
steps = round((max_depth/0.05))
print("hhhhhhhhhhh")
print(steps)
# z_axis = np.arange(0,steps*scale)*res/scale

ne = 32

x_channels = (np.arange(0,ne)*pitch) 
# print(x_channels[15:18])

x_axis = x_channels



mask =np.zeros((z_axis.shape[0],x_axis.shape[0],ne))
f_num = 1
ii = 0
for i in x_axis:
    a = z_axis/2*f_num
    start = np.floor((i - a)/pitch ).astype(int)
    end = np.ceil((i + a)/pitch ).astype(int)
    start[ start< 0] = 0
    end[ end > ne-1] = ne-1
    for j in range(0,len(z_axis)):
        mask[j,ii,start[j]:end[j] ] = 1
    ii = ii+1

X_32 = bf(X,x_axis,z_axis,x_channels,ne,0,c,terror,tstart,f_resample,mask)

print(X_32.shape)

M_X_32 = X_32[:,element].reshape(-1,1)

fig2 = plt.subplot(131)

Image = plt.imshow(clean(M_X_32),cmap='gray',interpolation='none', extent=[-1,1,max_depth,start_depth],aspect=aspect)


# 3 BF ****************************************************************************************


# frr = element - ( ne - 2)
# too = element + ( ne - 1)
half_ne = 1
frr = element - half_ne
too = element + half_ne + 1
ne = too-frr

print("*******************************")

print(frr,too,ne)

x_channels = (np.arange(frr,too)*pitch) 
x_axis = x_channels

mask = mask[:,frr:too,frr:too]

# mask =np.zeros((z_axis.shape[0],x_axis.shape[0],ne))
# f_num = 1
# ii = 0
# for i in x_axis:
#     a = z_axis/2*f_num
#     start = np.floor((i - a)/pitch ).astype(int)
#     end = np.ceil((i + a)/pitch ).astype(int)
#     start[ start< 15] = 0
#     end[ end > 15+ne-1] = 15+ne-1
#     for j in range(0,len(z_axis)):
#         mask[j,ii,start[j]:end[j] ] = 1
#     ii = ii+1

X_3 = bf(X[:,frr:too],x_axis,z_axis,x_channels,ne,0,c,terror,tstart,f_resample,mask)

print(X_3.shape)
print("*****************")
print(int(ne/2))
M_X_3 = X_3[:,element-frr].reshape(-1,1)

fig3 = plt.subplot(132)

Image = plt.imshow(clean(M_X_3),cmap='gray',interpolation='none', extent=[-1,1,max_depth,start_depth],aspect=aspect)


plt.show()

# *********************************** compare

def ethan_coffie(a,b):
    aa = a-np.mean(a)
    bb = b-np.mean(b)
    # print(aa,bb)
    # print(aa * bb)

    top = np.sum(aa * bb)
    bot = np.sqrt(np.sum(aa**2)) * np.sqrt(np.sum(bb**2))
  
    # print(top,bot)

    return (top)/(bot)


print("COMPARING")
print(X_RF.shape)
print(M_X_32.shape)
print(M_X_3.shape)

# print(ethan_coffie(M_X_32, M_X_32))
# print(ethan_coffie(M_X_32, M_X_3))
# print(ethan_coffie(M_X_32, X_RF))

# print(ethan_coffie(M_X_3, X_RF))

print(ethan_coffie(clean(M_X_32), clean(M_X_32)))
print(ethan_coffie(clean(M_X_32), clean(M_X_3)))
print(ethan_coffie(clean(M_X_32), clean(X_RF)))

# print(ethan_coffie(clean(M_X_3), clean(X_RF)))

# a = np.asarray([1,1])
# b = clean(M_X_3).reshape(-1)
# print(a.shape,b.shape)
# stats.pearsonr(b, b)

print(stats.pearsonr(clean(M_X_32).flatten(), clean(M_X_32).flatten()))
print(stats.pearsonr(clean(M_X_32).flatten(), clean(M_X_3).flatten()))
print(stats.pearsonr(clean(M_X_32).flatten(), clean(X_RF).flatten()))