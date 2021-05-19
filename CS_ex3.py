import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from cosamp_fn import cosamp
import cvxpy as cvx

from os import listdir
from scipy.signal import hilbert, chirp
from scipy import signal
from scipy import interpolate

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


## Read Image file
file_name= 'Rabbit_Full/'+'Aor_M_20'
index = 0
XF,TF = capture(file_name,index)
X = np.load(XF)
# X = X[::2,:]
# X = butter_highpass_filter(X.T,1*1e6,20*1e6,order =5).T  # MUST BE ROW ARRAY 32*1000
# X = butter_lowpass_filter( X.T,8*1e6,20*1e6,order =5).T  # MUST BE ROW ARRAY 32*1000
# X = X[550:750,::2]
# X = X[550:750,100:2000:2]
X = X[500:800,:3000:2]
Time = np.load(TF)
print(X.shape)
# X = clean(X)

## Randomly samples the signal
[nx,ny] = X.shape
per = 0.7
p = np.round(nx*per).astype(int)
print(p,nx)
perm = np.random.choice(nx, p, replace=False)
y = np.zeros((p,ny))
for i in range(ny):
    y[:,i] = X[perm,i]





# ## Solve compressed sensing problem
Xrecon = np.zeros((nx,ny))
Psi = dct(np.identity(nx))
Theta = Psi[perm,:]

# for i in range(ny):
#     print(i)
#     s = cosamp(Theta,y[:,i],100,epsilon=1.e-10,max_iter=10)
#     xrecon = idct(s)
#     Xrecon[:,i] = xrecon

## Solve with other methods
for i in range(ny):
    print(i)
    vx = cvx.Variable(nx)
    objective = cvx.Minimize(cvx.norm(vx, 1))
    constraints = [Theta*vx == y[:,i]]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=False)
    s = np.array(vx.value)
    s = np.squeeze(s)
    xrecon = idct(s)
    Xrecon[:,i] = xrecon




## Plot
fig,axes = plt.subplots(1,4)
axes = axes.reshape(-1)


image = axes[0].imshow(clean(X),aspect='auto',cmap='gray')
image.set_clim(vmin=-40, vmax= 0)

Xm = 0 * np.ones(X.shape)
for i in range(ny):
    Xm[perm,i]  = y[:,i] 
image = axes[1].imshow(Xm,aspect='auto',cmap='gray')
image.set_clim(vmin=-40, vmax= 0)

image = axes[2].imshow(clean(Xrecon),aspect='auto',cmap='gray')
image.set_clim(vmin=-40, vmax= 0)


# fig,axes = plt.subplots(1,2)
# axes = axes.reshape(-1)
axes[3].plot(diameter(X)[10:],'b')
axes[3].plot(diameter(Xrecon)[10:],'r')





plt.show()
