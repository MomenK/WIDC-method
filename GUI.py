from scipy.signal import hilbert, chirp
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# from matplotlib.backend_bases import key_press_handler
import matplotlib.pyplot as plt

import numpy as np
from os import listdir
from os.path import isfile, join
from scipy import signal
import tkinter as tk
from scipy import interpolate

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


class MainApp(tk.Frame):
     def __init__(self, parent,path,index,*args, **kwargs):
        print(path,index)
        super().__init__(parent,*args, **kwargs)
        
        # Read Data ******************************************************************************************
        XF,TF = self.capture(path,index)
        X = np.load(XF )
        Time = np.load(TF)
        Time = Time - Time[0]
        X = butter_highpass_filter(X.T,1*1e6,20*1e6,order =5).T  # MUST BE ROW ARRAY 32*1000
        # data = self.clean(X)
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
        self.startIdx = 500
        self.range = 200
        self.endIdx = self.startIdx + self.range
        self.data = data
        self.dataRF = self.data[self.startIdx :self.endIdx ,:]
        self.dataToPlot = self.clean(self.dataRF)

        self.be = 0
        self.ee = self.range
        print(self.ee)

        self.Time = Time

        self.scale = 1.540*0.5*(1/20)
        self.extent = [self.Time[0],self.Time[-1], self.endIdx *self.scale,self.startIdx *self.scale]
        
        # **********************************************  Scale
        self.VerOffVar =  tk.DoubleVar()
        self.VerOffScale = tk.Scale( parent, orient=tk.VERTICAL,from_=0, to=600, resolution=20, \
        length=300, label='Vertical offset ',variable = self.VerOffVar, command= self._updateVerOff )  
        
        self.DRVar =  tk.DoubleVar()
        self.Scale = tk.Scale( parent, orient=tk.HORIZONTAL,from_=1, to=60, resolution=1, \
        length=300, label='Dynamic range (dB)',variable = self.DRVar, command= self._update ) 

        self.MinVar =  tk.DoubleVar()
        self.MinScale = tk.Scale( parent, orient=tk.HORIZONTAL,from_=0, to=self.range, resolution=1, \
        length=300, label='Minimum Rank',variable = self.MinVar, command= self._SVDupdate ) 

        self.MaxVar =  tk.DoubleVar()
        self.MaxScale = tk.Scale( parent, orient=tk.HORIZONTAL,from_=0, to=self.range, resolution=1, \
        length=300, label='Maximum Rank',variable = self.MaxVar, command= self._SVDupdate ) 
        
        # # ********************************************** Figure
        fig = plt.figure(figsize =(15,5) )#
        self.image = plt.imshow(self.dataToPlot, cmap='gray',aspect=0.1,extent=self.extent)
        plt.title('Image')
        plt.ylabel('Width (mm)')
        plt.xlabel('Time (Seconds)')

        self.canvas = FigureCanvasTkAgg(fig, master=parent)  # A tk.DrawingArea.
        self.canvas.draw()

       #  Button 
        self.button = tk.Button(master=parent,bg='whitesmoke', text="Wall", command=self.Wall)



     def place(self,x,y):
        self.VerOffScale.grid(row=x, column=y+1, padx=5, pady=5, sticky='w'+'e'+'n'+'s')
        self.button.grid(row=x+1, column=y+1, padx=5, pady=5, sticky='w'+'e'+'n'+'s')
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
      #   self.dataToPlot[0:100,:] = 0

        self.image.set_data(self.dataToPlot)
        self.extent = [0,6, self.endIdx *self.scale,self.startIdx *self.scale]

        self.image.set_extent(self.extent)
        self.canvas.draw()
        pass


     def _updateVerOff(self,value):
        self.startIdx = int(self.VerOffScale.get())
        self.endIdx = self.startIdx + self.range
        self.updateImage()
        pass

     def _update(self,value):
      #   thre = 10**(-self.DRVar.get()/20)
      #   self.dataToPlot[self.dataToPlot < thre] = thre
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

     
     def corrr(self,x,y):
        kind = 'cubic'
        xsize = x.shape[0]
        xaxis = np.linspace(0,xsize,xsize)

        upaxis = np.linspace(0,xsize,xsize*10)
        fx = interpolate.interp1d(xaxis, x,kind= kind)
        fy = interpolate.interp1d(xaxis, y,kind= kind)

        x = fx(upaxis)
        y = fy(upaxis)
      #   x = self.clean1(x)
      #   y = self.clean1(y)
        return signal.correlate(x,y,mode='same')

     
     def Wall(self):

        data = self.clean1(self.dataRF)

        x = data.shape[0]
        y = data.shape[1]
        xx = int(x/2)
        corrMapTop = np.zeros((10*xx,y))
        corrMapBot = np.zeros((10*xx,y))

        sigTop = data[:,100][0:xx]
        sigBot = data[:,100][xx:]

        
        for i in range(0,y):
            sigTop1 = data[:,i][0:xx]       
            corrMapTop[:,i] = self.corrr(sigTop1,sigTop)

            sigBot1 = data[:,i][xx:]       
            corrMapBot[:,i] = self.corrr(sigBot1,sigBot)
                    
        plt.figure()
      #   Image = plt.imshow(corrMap,cmap='gray', aspect='auto')
        TopInd = np.argmax(corrMapTop,axis=0)[::2]*self.scale /10
        BotInd = np.argmax(corrMapBot,axis=0)[::2] *self.scale /10
        corrInd = xx*self.scale - TopInd +  BotInd
     
      #   corrInd =  xx*self.scale + (TopInd - BotInd)

        print(corrInd.shape)
        print(self.Time.shape)

      #   plt.figure()
        plt.close()
        plt.figure(figsize=(15,2))
      
        plt.plot(self.Time,corrInd)
        plt.title('Diameter')
        plt.ylabel('Width (mm)')
        plt.xlabel('Time (Seconds)')
        plt.locator_params(axis='y', nbins=10)
        plt.locator_params(axis='x', nbins=10)
      #   plt.xlim(0, 6) 
        plt.margins(0)
        plt.show()
        pass

        
      

     
if __name__ == "__main__":
    file_name= 'Rabbit_Full/'+'Aor_F_10_25'
   #  file_name= 'Rabbit_Full/'+'Aor_F_20_26'
    root = tk.Tk()
    MainApp(root,file_name,0).grid(row=0, column=1, padx=10, pady=5, sticky='NW')
    root.mainloop()