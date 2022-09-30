#sample
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import hilbert
from sklearn.model_selection import train_test_split

from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import tensorflow as tf
from tensorflow import keras

#-1. read from xlsx into dataframe
dataset = "dataset\T_Wand_000.xlsx"
df = pd.read_excel(dataset)
#peak of given signal
peak = np.zeros((len(df.index),), dtype=int)
label = np.zeros((len(df.index),), dtype=int)
df = df.iloc[0: ,17:]

#print(df.shape)
#[532 rows x 16368 columns]
#window(124)*class(264) = 16368 and overlap(62)

#looping thru all df
for i in range(0, len(df.index)):
  f= df.iloc[i,0:]
  
  n = f.size          #size of the signal
  dt= 0.05            #randomly choosen sampling rate
  time=np.arange(n)   #time of the signal
  
  fhat = np.fft.fft(f,n)
  PSD = fhat * np.conj(fhat) / n
  freq = (1/(dt*n))*np.arange(n)
  L = np.arange(0, n//2, dtype='int')
  
  indices = PSD > 1.5
  PSDclean = PSD * indices
  fhat = indices * fhat
  ffilt = np.fft.ifft(fhat)
  
  analytical_signal = hilbert(ffilt.real)
  env = np.abs(analytical_signal)
  x, _ = find_peaks(env, distance=n)
  
  fig, axs = plt.subplots(3,1)  
  #4. plot orignal noisy signal
  plt.sca(axs[0])
  plt.plot(time,f, label='Noisy')
  plt.xlim(time[0], time[-1])
  plt.xlabel('Time')
  plt.ylabel('Amplitude')
  plt.legend()
  
  #5. plot FFT of noisy and filtered signal
  plt.sca(axs[1])
  plt.plot(freq[L], PSD[L], color= 'c', linewidth=2, label='Noisy')
  plt.plot(freq[L], PSDclean[L], color= 'k', linewidth=1.5, label='Filtered')
  plt.xlim(freq[L[0]], freq[L[-1]])
  plt.xlabel('Frequency')
  plt.ylabel('Power')
  plt.legend()
  
  #6. plot filtered signal with upper envelope and peaks marked as 'x'
  plt.sca(axs[2])
  plt.plot(time, ffilt, label='Filtered Signal')
  plt.plot(time, env, label='Envelope')
  plt.plot(x, env[x], "x")
  plt.xlim(time[0], time[-1])
  plt.xlabel('Time')
  plt.ylabel('Amplitude')
  plt.legend()
  
  plt.show()

  peak[i] = x