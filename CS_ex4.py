from pylbfgs import owlqn

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from cosamp_fn import cosamp
import cvxpy as cvx

from os import listdir
from scipy.signal import hilbert, chirp
from scipy import signal
from scipy import interpolate

def dct2(x):
    return dct(dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    return idct(idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)



def evaluate(x, g, step):
    """An in-memory evaluation callback."""

    # we want to return two things: 
    # (1) the norm squared of the residuals, sum((Ax-b).^2), and
    # (2) the gradient 2*A'(Ax-b)

    # expand x columns-first
    x2 = x.reshape((nx, ny)).T

    # Ax is just the inverse 2D dct of x2
    Ax2 = idct2(x2)

    # stack columns and extract samples
    Ax = Ax2.T.flat[perm].reshape(y.shape)

    # calculate the residual Ax-b and its 2-norm squared
    Axb = Ax - y
    fx = np.sum(np.power(Axb, 2))

    # project residual vector (k x 1) onto blank image (ny x nx)
    Axb2 = np.zeros(x2.shape)
    Axb2.T.flat[perm] = Axb # fill columns-first

    # A'(Ax-b) is just the 2D dct of Axb2
    AtAxb2 = 2 * dct2(Axb2)
    
    AtAxb = AtAxb2.T.reshape(x.shape) # stack columns

    # copy over the gradient vector
    np.copyto(g, AtAxb)

    return fx

######################################################################################


######################################################################################
def capture(folder,index):
    RFPath = folder+'/M_RFArrays/'
    TimePath = folder+'/M_Arrays/'

    files = listdir(RFPath)
    timeFiles = ['T'+file for file in files]
    print(files,timeFiles)
    return RFPath + files[index],TimePath + timeFiles[index]

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
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def clean(X):
      X = butter_highpass_filter(X.T,1*1e6,20*1e6,order =5).T  
# X = butter_lowpass_filter( X.T,8*1e6,20*1e6,order =5).T 
      img = hilbert(X.T).T  
      # img = hilbert(X)
      img= img/np.amax(img)
      img = np.abs(img)
      img  = 20*np.log10(img)
      return img

def corrr(x,y,scale):
      kind = 'quadratic'
      xsize = x.shape[0]
      xaxis = np.linspace(0,xsize,xsize)

      upaxis = np.linspace(0,xsize,xsize*scale)
      fx = interpolate.interp1d(xaxis, x,kind= kind)
      fy = interpolate.interp1d(xaxis, y,kind= kind)

      x = fx(upaxis)
      y = fy(upaxis)

      return signal.correlate(x,y,mode='same')

def diameter(X):
    upsample = 10
    [nx, ny] = X.shape
    mx = int(nx/2)
    corrMapTop = np.zeros((upsample*mx,ny))
    corrMapBot = np.zeros((upsample*mx,ny))
    some_index = int(ny/2)
    sigTop = X[:,some_index][0:mx]
    sigBot = X[:,some_index][mx:]

    for i in range(ny):
        sigTop1 = X[:,i][0:mx] 
        corrMapTop[:,i] = corrr(sigTop1,sigTop,upsample)
        sigBot1 = X[:,i][mx:]       
        corrMapBot[:,i] = corrr(sigBot1,sigBot,upsample)
    
    TopInd = np.argmax(corrMapTop,axis=0)/upsample
    BotInd = mx + np.argmax(corrMapBot,axis=0)/upsample

    return BotInd  - TopInd 

    pass
######################################################################################
def repeat_random(k,ny,nx,C):
    permCD = np.zeros((k,C))
    perm = np.zeros((k,nx))

    for j in range(C):
        permCD[:,j] = np.random.choice(ny, k, replace=False)

    offset = 0
    for i in range(nx):
        ii = int(i%C)
        perm[:,i]  = permCD[:,ii] + offset
        offset = offset+ny

    return perm.flat[:].astype(int)
######################################################################################


## Read Image file
file_name= 'Rabbit_Full/'+'Aor_M_20'
index = 0
XF,TF = capture(file_name,index)
X = np.load(XF)
X = X[500:800,:3000:2]
Time = np.load(TF)

######################################################################################


## Randomly samples the signal
# [nx,ny] = X.shape
[ny,nx] = X.shape

# create random sampling index vector
s = 0.4
# k = round(nx * ny * s)
# perm = np.random.choice(nx * ny, k, replace=False) # random sample of indices

# This bits consider uniform sampling! consider have a random rpeat rate of beat cycle
C = 20
k = round(ny * s)
perm = repeat_random(k,ny,nx,C)



######################################################################################


# take random samples of image, store them in a vector b
y = b = X.T.flat[perm].astype(float)


# create images of mask (for visualization)
Xm = 0 * np.ones(X.shape)
# Xm.T.flat[perm] = X.T.flat[perm]
Xm.T.flat[perm] = 255 * np.ones(perm.shape)
Xm.reshape((nx, ny)).T 


# perform the L1 minimization in memory
Xat2 = owlqn(nx*ny, evaluate, None, 5)

# # transform the output back into the spatial domain
Xat = Xat2.reshape((nx, ny)).T # stack columns
Xrecon = idct2(Xat)

# print(X.shape)
# print(Xat.shape)
# print(Xrecon.shape)

######################################################################################
## Plot
fig,axes = plt.subplots(1,4)
axes = axes.reshape(-1)


image = axes[0].imshow(clean(X),aspect='auto',cmap='gray')
image.set_clim(vmin=-40, vmax= 0)
axes[0].title.set_text('Original Image (100 % sampling)')

image = axes[1].imshow(Xm,aspect='auto',cmap='gray')
# image.set_clim(vmin=-40, vmax= 0)
axes[1].title.set_text('Random sampling matrix (40 % sampling)')

image = axes[2].imshow(clean(Xrecon),aspect='auto',cmap='gray')
image.set_clim(vmin=-38, vmax= 0)
axes[2].title.set_text('Reconstructed Image (40 % sampling)')

axes[3].plot(diameter(X)[10:],'b')
axes[3].plot(diameter(Xrecon)[10:],'r')
axes[3].title.set_text('Diameter traces')

plt.show()

# # print(perm.shape)