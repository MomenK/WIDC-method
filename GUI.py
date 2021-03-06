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
from scipy.signal import savgol_filter

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


class MainApp(tk.Frame):
     def __init__(self, parent,path,index,*args, **kwargs):
        print(path,index)
        super().__init__(parent,*args, **kwargs)
        
        # Read Data ******************************************************************************************
        XF,TF = self.capture(path,index)
        X = np.load(XF )
        Time = np.load(TF)
        Time = Time - Time[0]
        X = butter_highpass_filter(X.T,3*1e6,20*1e6,order =5).T  # MUST BE ROW ARRAY 32*1000

        X = butter_lowpass_filter( X.T,8*1e6,20*1e6,order =5).T  # MUST BE ROW ARRAY 32*1000
      #   data = self.clean(X)
        data = X

      #  set plot window
        self.PlotWindow = PlotWindow(parent,data,Time)
        self.PlotWindow.place(1,0)

        # self.PlotWindow1 = PlotWindow(parent,data)
        # self.PlotWindow1.place(5,0)

     def capture(self,folder,index):
        RFPath = folder+'/M_RFArrays/'
        TimePath = folder+'/M_Arrays/'

        files = listdir(RFPath)
        timeFiles = ['T'+file for file in files]
        print(files,timeFiles)
        return RFPath + files[index],TimePath + timeFiles[index]

 

class PlotWindow():
   def __init__(self, parent, data,Time):
      self.startIdx = 558
      self.range = 130
      self.DR = 30
      self.endIdx = self.startIdx + self.range
      self.data = data
      self.dataRF = self.data[self.startIdx :self.endIdx ,:]
      self.dataToPlot = self.clean(self.dataRF)

      self.wall = 0

      self.be = 0
      self.ee = self.range
      print(self.ee)

      self.Time = Time

      self.scale = 1.540*0.5*(1/20)
      print(self.scale)
      self.extent = [self.Time[0],self.Time[-1], self.endIdx *self.scale,self.startIdx *self.scale]


      print(self.Time.shape)
      y = self.Time.reshape(-1,2).T
      self.fs  = 1/(np.mean(y[1]-y[0]))
      print(self.Time.shape,y.shape,self.fs)


      
      # **********************************************  Scale
      self.VerOffVar =  tk.DoubleVar()
      self.VerOffScale = tk.Scale( parent, orient=tk.VERTICAL,from_=500, to=800, resolution=1, \
      length=300, label='Vertical offset ',variable = self.VerOffVar, command= self._updateVerOff )  
      self.VerOffScale.set(int(self.startIdx + self.range/2))

      self.RangeVar =  tk.DoubleVar()
      self.RangeScale = tk.Scale( parent, orient=tk.VERTICAL,from_=50, to=600, resolution=10, \
      length=300, label='Vertical offset ',variable = self.RangeVar, command= self._updateRange )  
      self.RangeVar.set(self.range)
      
      self.DRVar =  tk.DoubleVar()
      self.Scale = tk.Scale( parent, orient=tk.HORIZONTAL,from_=1, to=60, resolution=1, \
      length=300, label='Dynamic range (dB)',variable = self.DRVar, command= self._update ) 
      self.Scale.set(self.DR)

      self.MinVar =  tk.DoubleVar()
      self.MinScale = tk.Scale( parent, orient=tk.HORIZONTAL,from_=0, to=self.range, resolution=1, \
      length=300, label='Minimum Rank',variable = self.MinVar, command= self._SVDupdate ) 
      self.MinScale.set(0)

      self.MaxVar =  tk.DoubleVar()
      self.MaxScale = tk.Scale( parent, orient=tk.HORIZONTAL,from_=0, to=self.range, resolution=1, \
      length=300, label='Maximum Rank',variable = self.MaxVar, command= self._SVDupdate ) 
      self.MaxVar.set(self.range)
      
      # # ********************************************** Figure
      fig = plt.figure(figsize =(15,5) )#
      self.ax = plt.gca()
      self.image = plt.imshow(self.dataToPlot, cmap='gray',aspect='auto',extent=self.extent)

      self.TopInd = np.zeros_like(self.Time) + (self.startIdx +  self.range/4)
      self.BotInd = np.zeros_like(self.Time) + (self.endIdx - self.range/4)

      self.axpMid, = self.ax.plot(self.Time,np.zeros_like(self.Time) + (self.startIdx +  self.range/2)*self.scale,'w')
      self.axTop, = self.ax.plot(self.Time,self.TopInd *self.scale,'m')
      self.axBot, = self.ax.plot(self.Time,self.BotInd *self.scale,'r')

      plt.title('Image')
      plt.ylabel('Width (mm)')
      plt.xlabel('Time (Seconds)')

      self.canvas = FigureCanvasTkAgg(fig, master=parent)  # A tk.DrawingArea.
      self.canvas.draw()

      #  Button 
      self.button = tk.Button(master=parent,bg='whitesmoke', text="Wall", command=self.Wall)
      self.Flowbutton = tk.Button(master=parent,bg='whitesmoke', text="Flow", command=self.Flow)
      self.SpectrumButton = tk.Button(master=parent,bg='whitesmoke', text="Spectrum", command=self.Spectrum)



   def place(self,x,y):
      self.VerOffScale.grid(row=x, column=y+1, padx=5, pady=5, sticky='w'+'e'+'n'+'s')
      self.RangeScale.grid(row=x, column=y+2, padx=5, pady=5, sticky='w'+'e'+'n'+'s')
      self.button.grid(row=x+1, column=y+1, padx=5, pady=5, sticky='w'+'e'+'n'+'s')
      self.Flowbutton.grid(row=x+2, column=y+1, padx=5, pady=5, sticky='w'+'e'+'n'+'s')
      self.SpectrumButton.grid(row=x+3, column=y+1, padx=5, pady=5, sticky='w'+'e'+'n'+'s')

      self.canvas.get_tk_widget().grid(row=x, column=y, padx=5, pady=5, sticky='w'+'e'+'n'+'s')

      self.Scale.grid(row=x+1, column=y, padx=5, pady=5, sticky='w'+'e'+'n'+'s')
      self.MinScale.grid(row=x+2, column=y, padx=5, pady=5, sticky='w'+'e'+'n'+'s')
      self.MaxScale.grid(row=x+3, column=y, padx=5, pady=5, sticky='w'+'e'+'n'+'s')
      pass

   def updateImage(self):
      self.dataRF = self.data[self.startIdx :self.endIdx ,:]
      dataRF = self.dataRF.copy()

      if (self.be < self.ee) and not (self.be == 0 and self.ee == self.range):
            u, s, vh = np.linalg.svd(self.dataRF, full_matrices=False)
            self.dataRF  =  np.dot(u[:,self.be:self.ee] * s[self.be:self.ee], vh[self.be:self.ee,:])
            print("SVDING")
            # print(self.be,self.ee, self.range)

      #   print(dataToPlot.shape)
      
      #   thre = 10**(-self.DRVar.get()/20)*np.amax(dataRF)
      #   print (thre,np.amax(dataRF),np.amin(dataRF))
      #   dataRF[np.where((dataRF < thre )&(dataRF >0))] = 0 
      #   dataRF[np.where((dataRF > -thre )&(dataRF < 0))] = 0         
      #   self.dataRF = dataRF
      #   print (thre,np.amax(self.dataRF),np.amin(self.dataRF)) 

      self.dataToPlot = self.clean(self.dataRF)




      self.image.set_data(self.dataToPlot)
      self.extent = [self.Time[0],self.Time[-1], self.endIdx *self.scale,self.startIdx *self.scale]


      self.axpMid.set_ydata(np.zeros_like(self.Time) + (self.startIdx +  self.range/2)*self.scale)
      self.axTop.set_ydata(self.BotInd*self.scale)
      self.axBot.set_ydata(self.TopInd*self.scale)

      self.image.set_extent(self.extent)
      self.canvas.draw()
      pass


   def _updateVerOff(self,value):
      self.startIdx = int(self.VerOffVar.get()- self.range/2) 
      self.endIdx = self.startIdx + self.range

      self.TopInd = np.zeros_like(self.Time) + (self.startIdx +  self.range/4)
      self.BotInd = np.zeros_like(self.Time) + (self.endIdx - self.range/4)
      self.updateImage()
      pass

   def _updateRange(self,value):
      self.range = int(self.RangeVar.get())
      self._updateVerOff(0)
      pass

   def _update(self,value):
      self.image.set_clim(vmin=-self.DRVar.get(), vmax= 0)
      self.canvas.draw()
      #   self.updateImage()
      pass

   def _SVDupdate(self,value):
      self.be = int(self.MinVar.get())
      self.ee = int(self.MaxVar.get())
      self.updateImage()
      pass
   
   def clean(self,X):
      img = hilbert(X.T).T  
      # img = hilbert(X)
      img= img/np.amax(img)
      img = np.abs(img)
      img  = 20*np.log10(img)
      return img

   def clean1(self,X):
      #   img = X
      img = hilbert(X.T).T  
      img= img/np.amax(img)
      img = np.abs(img)
      #   img  = 20*np.log10(img)
      return img

   
   def corrr(self,x,y,scale):
      kind = 'quadratic'
      xsize = x.shape[0]
      xaxis = np.linspace(0,xsize,xsize)

      upaxis = np.linspace(0,xsize,xsize*scale)
      fx = interpolate.interp1d(xaxis, x,kind= kind)
      fy = interpolate.interp1d(xaxis, y,kind= kind)

      x = fx(upaxis)
      y = fy(upaxis)

      return signal.correlate(x,y,mode='same')

   
   def Wall(self):
      upsample  = 10
      data = self.clean1(self.dataRF)[:,::2]

      x = data.shape[0]
      y = data.shape[1]
      xx = int(x/2)
      corrMapTop = np.zeros((upsample*xx,y))
      corrMapBot = np.zeros((upsample*xx,y))
      sigTop = data[:,100][0:xx]
      sigBot = data[:,100][xx:]

      for i in range(0,y):
            sigTop1 = data[:,i][0:xx]       
            corrMapTop[:,i] = self.corrr(sigTop1,sigTop,upsample)

            sigBot1 = data[:,i][xx:]       
            corrMapBot[:,i] = self.corrr(sigBot1,sigBot,upsample)
                  
      TopInd = np.argmax(corrMapTop,axis=0)/upsample
      BotInd = xx + np.argmax(corrMapBot,axis=0)/upsample
      
      self.TopInd = self.startIdx +TopInd
      self.BotInd = self.startIdx + BotInd
      corrInd = BotInd  - TopInd 
      self.wall = corrInd*self.scale
      print(corrInd.shape)
      print(self.Time.shape)

      #   self.axTop.set_ydata(self.BotInd*self.scale)
      #   self.axBot.set_ydata(self.TopInd*self.scale)
      #   self.canvas.draw()
      self.updateImage()

      # plt.figure(figsize=(15,4))
      # plt.subplot(211)
      # plt.imshow(data,extent=[self.Time[0],self.Time[-1],x,0 ],aspect='auto',cmap='gray')
      # plt.plot(self.Time,np.zeros_like(self.Time) + xx,'w')
      # plt.plot(self.Time,BotInd,'m')
      # plt.plot(self.Time,TopInd,'r' )


      plt.figure(figsize=(15,4))
      plt.subplot(211)
      plt.imshow(data,extent=[self.Time[0],self.Time[-1],x*self.scale,0 ],aspect='auto',cmap='gray')
      plt.plot(self.Time,np.zeros_like(self.Time) + xx*self.scale,'w')
      plt.plot(self.Time,BotInd*self.scale,'m')
      plt.plot(self.Time,TopInd*self.scale,'r' )
      plt.ylabel('Width (mm)')
      plt.xlabel('Time (Seconds)')
      plt.locator_params(axis='y', nbins=8)
      plt.locator_params(axis='x', nbins=10)
      

      plt.subplot(313)
      plt.plot(self.Time[2:],corrInd[2:]*self.scale )
      plt.title('Diameter')
      plt.ylabel('Width (mm)')
      plt.xlabel('Time (Seconds)')
      plt.locator_params(axis='y', nbins=8)
      plt.locator_params(axis='x', nbins=10)
      plt.margins(0)
      


      # DETECTING PEAKS!
      
      x = corrInd*self.scale 
      x[0] = x[3]
      x[1] = x[3]
      x[2] = x[3]
      
      peaks, _ = find_peaks(-x, distance=200)
      self.peaks = peaks
      valleys, _ = find_peaks(x, distance=200)
      valleys = valleys[1:]

      plt.figure(figsize=(6,4))

      plt.subplot(211)
      plt.plot(self.Time,x)
      plt.plot(self.Time[peaks], x[peaks], "x",)
      plt.plot(self.Time[valleys], x[valleys], "o",)

      plt.subplot(212)
      self.Average(x,self.peaks)

      plt.show()

      pass

   def Flow(self):

      clean = 1
      poly = 1

      dataRFRaw1 = self.dataRF.copy()[:,0::2]
      dataRFRaw2 = self.dataRF.copy()[:,1::2]
      x = dataRFRaw1.shape[0]
      y = dataRFRaw1.shape[1]

      window = 20
      window = 21
      if poly == 1:
         dataRF1,slowdataRF1 = self.declutter(dataRFRaw1,window)
         dataRF2,slowdataRF2 = self.declutter(dataRFRaw2,window)
      else:
         dataRF1  = dataRFRaw1
         dataRF2 = dataRFRaw2

 
      if clean == 1:
         data1 = self.clean(dataRFRaw1)
         data2 = self.clean(dataRFRaw2)

      else:
         data1 = self.clean1(dataRFRaw1)
         data2 = self.clean1(dataRFRaw2)
      
 

      S =  int(x/2)

      D1 = self.clean1(dataRF1)
      D2 = self.clean1(dataRF2)

      plt.figure(figsize=(15,4))
      plt.plot(dataRFRaw1[S],'g')
      if poly == 1:
         plt.plot(slowdataRF1[S],'k')
      plt.plot(dataRF1[S],'b')
      plt.plot(dataRF2[S],'r')
      plt.xlim(1900, 2500) 
      plt.ylim(-20, 40) 
     

     
      print('starting flow calculations')

      print('Shape of data array')
      print(data1.shape,data2.shape)

      print('Shape of time array')
      print(self.Time.shape)

      coff_vector = np.ones(y)



      for i in range(0,y):
         ii = int(self.TopInd[i]) - self.startIdx
         jj = int(self.BotInd[i]) - self.startIdx

         # data[0:ii,2*i:2*i+2] = 0
         # data[jj:,2*i:2*i+2] = 0

         # dataRF1[0:ii,i] = 0
         # dataRF1[jj:,i] = 0

         # dataRF2[0:ii,i] = 0
         # dataRF2[jj:,i] = 0

         coff_vector[i] = self.corr_coff2(dataRF1[ii:jj,i],dataRF2[ii:jj,i])

         # coff_vector[i] = self.corr_coff2(self.clean1(dataRF1[:,i]),self.clean1(dataRF2[:,i]))

         # coff_vector[i] = self.corr_coff2(dataRF1[:,i],dataRF2[:,i])

         # coff_vector[i] = self.corr_coff2(D1[ii:jj,i],D2[ii:jj,i])

       


      
      flow = coff_vector

      # coff_vector = coff_vector+ 0.1

      # coff_vector[coff_vector>1] = 1
      # # coff_vector[coff_vector == -1] = 1
      # coff_vector[coff_vector<0] = 0.001

      # coff_vector[coff_vector==0] = 

      # dt = 200e-6
      # for i in range(0,y):
      #    try:
      #       flow[i] = ((-2*math.log(coff_vector[i]))**0.5)/dt
      #    except:
      #       print("ERROR",str(i),coff_vector[i])




      # flow = coff_vector

      # alpha = (-1/np.log(coff_vector**2))**0.5
      # flow = alpha

      # p = 0.2
      # dr = 200e-6
      # alpha = ((-1/(2*np.log(p**2)))**0.5)*dr
      # dt = dr
      # D = ((-2*np.log(coff_vector))**0.5)/dt
      # flow = D*alpha

      # flow_smooth = uniform_filter1d(flow, size=20)
      flow_smooth = flow

      plt.figure(figsize=(15,4))
      plt.subplot(211)
      # plt.imshow(data,extent=[self.Time[0],self.Time[-1],x,0 ],aspect='auto',cmap='gray')
      # plt.imshow(self.clean1(dataRF1),extent=[self.Time[0],self.Time[-1],x,0 ],aspect='auto',cmap='gray')

      plt.imshow(self.clean1(dataRF1),extent=[0,dataRFRaw1.shape[1],x,0 ],aspect='auto',cmap='gray')
      plt.plot(self.TopInd - self.startIdx,'m')
      plt.plot(self.BotInd - self.startIdx,'r')
      
      # flow_smooth = np.flip(flow_smooth)

      # plt.plot(self.Time, x*(1-(flow_smooth/np.amax(flow_smooth))))
      # plt.plot(self.Time, x*(1-(self.wall/np.amax(self.wall))))

      # plt.plot( x*(1-(flow_smooth/np.amax(flow_smooth))))
      # plt.plot( x*(1-(self.wall/np.amax(self.wall))))

      

      plt.subplot(313)
      # plt.plot(self.Time, flow)
      # plt.plot(self.Time, flow_smooth)
      plt.plot( flow_smooth)
      plt.title('Flow')
      plt.ylabel('[1 - r]')
      plt.xlabel('Time (Seconds)')
      plt.margins(0)
   





      plt.figure(figsize=(15,4))
      plt.subplot(511)
      plt.imshow( self.clean(dataRF1),extent=[self.Time[0],self.Time[-1],x,0 ],aspect='auto',cmap='gray')

      plt.subplot(512)
      plt.imshow(self.clean(dataRF2),extent=[self.Time[0],self.Time[-1],x,0 ],aspect='auto',cmap='gray')

      avgRF = (dataRF1+dataRF2)/2
      plt.subplot(513)
      plt.imshow(self.clean(avgRF),extent=[self.Time[0],self.Time[-1],x,0 ],aspect='auto',cmap='gray')

      plt.subplot(514)
      diff = dataRF1-dataRF2
      plt.imshow((diff),extent=[self.Time[0],self.Time[-1],x,0 ],aspect='auto',cmap='gray')


      plt.subplot(515)
      diff = self.clean1(diff)
      plt.imshow((diff),extent=[self.Time[0],self.Time[-1],x,0 ],aspect='auto',cmap='gray')



      

      plt.figure(figsize=(15,4))

      plt.subplot(211)
      if clean == 1:
         plt.imshow(self.clean(dataRF1),extent=[0,dataRFRaw1.shape[1],x,0 ],aspect='auto',cmap='gray')
      else:
         plt.imshow(self.clean1(dataRF1),extent=[0,dataRFRaw1.shape[1],x,0 ],aspect='auto',cmap='gray')

      plt.plot(self.TopInd - self.startIdx,'m')
      plt.plot(self.BotInd - self.startIdx,'r')   

      plt.subplot(212)
      self.Average(flow_smooth,self.peaks)


      plt.show()
      pass

   def corr_coff(self,x):
      return stats.pearsonr(x[:,0]  ,x[:,1]  )[0]

   def corr_coff2(self,x,y):
      return stats.pearsonr(x ,y  )[0]

   def declutter(self,x,window):
      # slowX = uniform_filter1d(x, size=window,axis=1)

      slowX = savgol_filter(x, window, 5, deriv=0, delta=1.0, axis=1, mode='interp', cval=0.0)
     

      # fs = self.fs
      # slowX = butter_lowpass_filter(x,100,fs,5)
      # # out = butter_highpass_filter(x,300,fs,5)
      
      
      out = x-slowX
      print("SlowX")
      print(slowX.shape)
      return out,slowX

      t = range(0,x.shape[1])
      for i in range(0,x.shape[0]):
         p30 = np.poly1d(np.polyfit(t,x[i,:], 10))
         x[i,:] = x[i,:] - p30(t)
      return x
   

   def Average(self,x,peaks):

      peak_num = len(peaks)
      peak_dist = len(x)

      for i in range(0,int((peak_num/2)-1)):
         peak_dist = min(peak_dist,  peaks[i+1] - peaks[i]     )
      
      print("PEAK DIST")
      print(peak_dist)
            
      timeStep = self.Time[10]-self.Time[9]
      t = np.arange(0,peak_dist)*timeStep

      ave = np.zeros(peak_dist)

      for peak in peaks[:-2]:

         cycle = x[peak:peak+peak_dist]
         ave = ave + cycle
         plt.plot( t,cycle, c='#0f0f0f20')

      
      data = ave/(len(peaks)- 2)
      
      plt.plot( t,data ,"-")

      np.savetxt("output/time.csv", t, delimiter=",")
      np.savetxt("output/data.csv", data, delimiter=",")

      np.savetxt("output/Fulltime.csv", self.Time, delimiter=",")
      np.savetxt("output/Fulldata.csv", x, delimiter=",")

      np.savetxt("output/Peaks_" + str(peak_dist) + ".csv", peaks, delimiter=",")
      

  


   def masks(self,vec):
    d = np.diff(vec)
    dd = np.diff(d)

    # Mask of locations where graph goes to vertical or horizontal, depending on vec
    to_mask = ((d[:-1] != 0) & (d[:-1] == -dd))
    # Mask of locations where graph comes from vertical or horizontal, depending on vec
    from_mask = ((d[1:] != 0) & (d[1:] == dd))
    return to_mask, from_mask


   def Spectrum(self):
      dataRFRaw1 = self.dataRF.copy()[:,::2]
     
      x = dataRFRaw1.shape[0]
      S =  int(x/2)
      Data = dataRFRaw1[S]


      print(self.fs)

 

      plt.figure()
      wind = 50
      spectrum2D, ff,tt,im = plt.specgram(Data,NFFT=wind, Fs=self.fs, noverlap=wind-5,cmap="jet_r")
      print(spectrum2D.shape)
      plt.figure()
      plt.imshow(np.log(spectrum2D),extent= [tt[0], tt[-1], ff[0], ff[-1]   ],cmap='jet_r',aspect=0.01,origin='lower' )
     

      be = 0
      ee = 5

      u, s, vh = np.linalg.svd(np.log(spectrum2D),full_matrices=False)
      SVD  =  np.dot(u[:,be:ee] * s[be:ee], vh[be:ee,:])


      plt.figure()
      plt.imshow(SVD,extent= [tt[0], tt[-1], ff[0], ff[-1]   ],cmap='jet_r',aspect=0.01,origin='lower' )

      plt.figure()
      plt.plot( s )
      

      plt.show()

      pass
   

     
if __name__ == "__main__":
   # file_name= 'Rabbit_Full/'+'Aor_F_10_25'
   # file_name= 'Rabbit_Full/'+'Aor_F_20_24'
   file_name= 'Rabbit_Full/'+'Aor_M_20'
   # file_name= 'Rabbit_Full/'+'Aor_F'

   root = tk.Tk()
   MainApp(root,file_name,0).grid(row=0, column=1, padx=10, pady=5, sticky='NW')
   root.mainloop()