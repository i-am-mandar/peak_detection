import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = "dataset\T_File_5.xlsx"
df = pd.read_excel(dataset)

df = df.iloc[0: ,6:]
#print(df)

#length of column
col_len = len(df.columns)
#print(col_len)

#length of column
row_len = len(df.index)
#print(row_len)

#first row of df
row_1 = df.head(1)

n = col_len
y = row_1

ffty = np.fft.fft(y)
fftx = np.fft.fftfreq(n, d=1)[:n//2]
#default sample space (inverse of sampling rate) is 1

print(ffty)
print(fftx)

#plt.grid()
#plt.plot(fftx, 2.0/n * np.abs(ffty[0:n//2]))
#plt.show()



# 
# Generic Approach
# 0. Get a sample window and sampling rate to use in FFT
# 1. Convert time domain signal to frequency domain using Fast Fourier Transform (fft is library in python)
# 2. Analyse the frequency and apply filter (As per Prof use band pass filter with min. frequency of 30kHz and max. frequency of 50kHz), again get back the signal to time domain using Reverse Fast Fourier Transform
# 3. Use method called as "Envelope of Signal" (Hilbert Transform is one method) which gives you max and min of the whole singal at every point in time
# 4. Label the new data at every point in time
# 5. Use these label data and feed it to your neural network
#

