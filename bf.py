import numpy as np
import numpy.matlib
import math
from scipy import interpolate

def bf(data,x_axis,z_axis,x_channels,ne,theta,c,terror,tstart,f_resample,mask):
    # print(data.shape,theta)

    #  Vairables **************************************************************************************
    X = np.matlib.repmat(x_axis, len(z_axis), 1)
    Z = np.matlib.repmat(z_axis, len(x_axis),1).T

    delay_tx = np.zeros_like(X)
    delay_rcv = np.zeros((X.shape[0],X.shape[1],ne))
    apo = np.zeros((X.shape[0],X.shape[1],ne))

    #  calculate TX delays **************************************************************************************
    delay_tx = ( Z*math.cos(math.radians(theta)) + X*math.sin(math.radians(theta)) )/c
    delay_tx.tofile('delay_tx.csv',sep=',',format='%10.10f')
    print(np.amax(delay_tx),np.amin(delay_tx))

    #  Receive delays and apodization **************************************************************************************
    d = 3.0000e-04
    k = (math.pi*d*f_resample)/(4*c)
    print(d,k)

    #  calcuate RX delays and apodization **************************************************************************************

    for i in range(0,ne):
        delay_rcv[:,:,i] = np.sqrt((X - x_channels[i])**2 + Z**2 )/c
        apoTheta = (X - x_channels[i])/Z
        # print(apoTheta.shape)
        apoTheta = np.arctan(apoTheta)

        # apo[:,:,i] = (np.sin(k*np.sin(apoTheta))*np.cos(apoTheta))/(k*np.sin(apoTheta))

    # print(apo.shape,delay_rcv.shape)

    print(np.amax(delay_rcv),np.amin(delay_rcv))
    print(np.amax(apo),np.amin(apo))
    #  Beamforming **************************************************************************************
    interp_value = np.zeros((X.shape[0],X.shape[1],ne))
    
    for j in range(0,ne):
        t = np.arange( 0 , data.shape[0]/f_resample, 1/f_resample) + tstart-terror
        D = delay_tx + delay_rcv[:,:,j]
        # print(t.shape,D.shape,data.shape)
        Dshape = D.shape

        f = interpolate.interp1d(t, data[:,j],fill_value=0,bounds_error=False,kind='linear')
        intd = f(D.flatten()).reshape(Dshape)

        # interp_value[:,:,j] = apo[:,:,j]*intd
        # interp_value[:,:,j] = intd
        interp_value[:,:,j] = mask[:,:,j]*intd

    output = np.sum(interp_value,axis=2)
    
    

    print(output.shape)

    print(np.amax(output),np.amin(output))

    return output


