#new dataset and cnn
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

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#-1. read from xlsx into dataframe
dataset = "dataset\Wand_000.xlsx"
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
  
  #fig, axs = plt.subplots(3,1)  
  ##4. plot orignal noisy signal
  #plt.sca(axs[0])
  #plt.plot(time,f, label='Noisy')
  #plt.xlim(time[0], time[-1])
  #plt.xlabel('Time')
  #plt.ylabel('Amplitude')
  #plt.legend()
  #
  ##5. plot FFT of noisy and filtered signal
  #plt.sca(axs[1])
  #plt.plot(freq[L], PSD[L], color= 'c', linewidth=2, label='Noisy')
  #plt.plot(freq[L], PSDclean[L], color= 'k', linewidth=1.5, label='Filtered')
  #plt.xlim(freq[L[0]], freq[L[-1]])
  #plt.xlabel('Frequency')
  #plt.ylabel('Power')
  #plt.legend()
  #
  ##6. plot filtered signal with upper envelope and peaks marked as 'x'
  #plt.sca(axs[2])
  #plt.plot(time, ffilt, label='Filtered Signal')
  #plt.plot(time, env, label='Envelope')
  #plt.plot(x, env[x], "x")
  #plt.xlim(time[0], time[-1])
  #plt.xlabel('Time')
  #plt.ylabel('Amplitude')
  #plt.legend()
  #
  #plt.show()

  peak[i] = x
  
n_slice = 62

for i in range(0,len(peak)):
  label[i] = peak[i]//n_slice


y_label_len = len(df.columns)//n_slice
y_label = np.zeros((len(df.index),y_label_len), dtype=int)


for i in range(0,len(label)):
  y_label[i,label[i]] = 1


xtrain, xtest, ytrain, ytest=train_test_split(df, y_label, test_size=0.25)
verbose, epochs, batch_size = 1, 10, 32
model = Sequential()

#model the CNN approach 1
model.add(Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=(len(df.columns),1)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=5))
model.add(Flatten())
model.add(Dense(1052, activation='relu'))
model.add(Dense(263, activation='softmax'))

#model the CNN approach 2
#model.add(Conv1D(64, 3, activation="relu", input_shape=(len(df.columns),1)))
#model.add(Dense(16, activation="relu"))
#model.add(MaxPooling1D())
#model.add(Flatten())
#model.add(Dense(263, activation = 'softmax'))

#compile the model and fit it
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size, verbose=verbose)

#evaluate model and get accuracy
_, accuracy = model.evaluate(xtest, ytest, batch_size=batch_size, verbose=verbose)
accuracy = accuracy * 100.0

#save the model
#model.save("trained_model") #uncomment to save the model
