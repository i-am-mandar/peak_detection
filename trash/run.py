import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

dataset = "dataset\T_File_5.xlsx"
df = pd.read_excel(dataset)

df = df.iloc[0: ,6:]
#print(df)

#length of column
col_len = len(df.columns)
print(col_len)

#length of row
row_len = len(df.index)
#print(row_len)

#first row of df
row_1 = df.head(1)
#200 * 17 = 3400
step = 200
for i in range(step, col_len+1,(step//2)):
    y = row_1.iloc[: ,(i-step) : i]
    n = y.size
    timestep = 0.01
    ffty = np.fft.rfft(y)
    fftx = np.fft.rfftfreq(n, d=timestep)
    #print(len(ffty[0]))
    #print(len(fftx))
    plt.grid()
    plt.plot(fftx, np.abs(ffty[0]))
    plt.show()
print("done")



# 
# Generic Approach
# 0. Get a sample window and sampling rate to use in FFT
# 1. Convert time domain signal to frequency domain using Fast Fourier Transform (fft is library in python)
# 2. Analyse the frequency and apply filter (As per Prof use band pass filter with min. frequency of 30kHz and max. frequency of 50kHz), again get back the signal to time domain using Reverse Fast Fourier Transform
# 3. Use method called as "Envelope of Signal" (Hilbert Transform is one method) which gives you max and min of the whole singal at every point in time
# 4. Label the new data at every point in time
# 5. Use these label data and feed it to your neural network
#
