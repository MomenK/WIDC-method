import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from cosamp_fn import cosamp
import cvxpy as cvx


n = 4096 # High resolutions samples
t = np.linspace(0,1,n)
x =  np.cos(2 * 97 * np.pi * t) + np.cos(2 * 777 * np.pi * t) 

## Randomly samples the signal
p = 128
perm = np.floor(np.random.rand(p) * n).astype(int)
# perm = np.random.choice(n, p, replace=False)
# perm.sort()
y = x[perm]

## Plot
time_window = np.array([1024, 1280])/n

fig,axes = plt.subplots(1,2)
axes = axes.reshape(-1)

axes[0].plot(t,x)
axes[0].plot(t[perm],y,'x',ms = 12)
axes[0].set_xlim(time_window[0],time_window[1])
axes[0].set_ylim(-2,2)

# ## Solve compressed sensing problem
Psi = dct(np.identity(n))
print(Psi.shape)
Theta = Psi[perm,:]
s = cosamp(Theta,y,10,epsilon=1.e-50,max_iter=1000)
xrecon = idct(s)

## Solve with other methods
# vx = cvx.Variable(n)
# objective = cvx.Minimize(cvx.norm(vx, 1))
# constraints = [Theta*vx == y]
# prob = cvx.Problem(objective, constraints)
# result = prob.solve(verbose=True)
# s = np.array(vx.value)
# s = np.squeeze(s)
# xrecon = idct(s)


## Plot
axes[0].plot(t,xrecon,'r')
# axes[1].set_xlim(time_window[0],time_window[1])
# axes[1].set_ylim(-2,2)

axes[1].imshow(Theta)

plt.show()


print("Norm")
print(np.linalg.norm(x-xrecon, ord=2))
print(np.linalg.norm(xrecon, ord=1))