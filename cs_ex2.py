import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from cosamp_fn import cosamp
import cvxpy as cvx

from os import listdir
from scipy.signal import hilbert, chirp
from scipy import signal


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
      img = hilbert(X.T).T  
      # img = hilbert(X)
      img= img/np.amax(img)
      img = np.abs(img)
      img  = 20*np.log10(img)
      return img
######################################################################################


## Read Image file
file_name= 'Rabbit_Full/'+'Aor_M_20'
index = 0
XF,TF = capture(file_name,index)
X = np.load(XF)
X = butter_highpass_filter(X.T,0.01*1e6,20*1e6,order =5).T  # MUST BE ROW ARRAY 32*1000
# X = butter_lowpass_filter( X.T,8*1e6,20*1e6,order =5).T  # MUST BE ROW ARRAY 32*1000
# X = X[550:750,200:400:2]
X = X[550:600,200:300:2]
Time = np.load(TF)
print(X.shape)



## Randomly samples the signal
[nx,ny] = X.shape
p = 0.5
k = round(nx * ny * p) # 50% sample
perm = np.random.choice(nx * ny, k, replace=False) # random sample of indices

y = X.T.flat[perm]
# y = np.expand_dims(y, axis=1)

## Plot
fig,axes = plt.subplots(1,2)
axes = axes.reshape(-1)
image = axes[0].imshow(clean(X),aspect='auto',cmap='gray')
image.set_clim(vmin=-40, vmax= 0)

Xm = 0 * np.ones(X.shape)
# Xm.T.flat[perm] = clean(X).T.flat[perm]
Xm.T.flat[perm] = clean(y)
# Xm.T.flat[perm] = 255
image = axes[1].imshow(Xm,aspect='auto',cmap='gray')
image.set_clim(vmin=-40, vmax= 0)


## Solve compressed sensing problem
print(nx,ny)
Psi = np.kron(
    dct(np.identity(nx), norm='ortho', axis=0),
    dct(np.identity(ny), norm='ortho', axis=0)
    )
Theta = Psi[perm,:] # same as phi times kron


vx = cvx.Variable(nx * ny)
objective = cvx.Minimize(cvx.norm(vx, 1))
constraints = [Theta*vx == y]
prob = cvx.Problem(objective, constraints)
result = prob.solve(verbose=True)
Xat2 = np.array(vx.value).squeeze()

Xat = Xat2.reshape(nx, ny).T # stack columns
Xa = idct(idct(Xat.T))
image = axes[1].imshow(Xa,aspect='auto',cmap='gray')
image.set_clim(vmin=-40, vmax= 0)



plt.show()
