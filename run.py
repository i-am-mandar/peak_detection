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

y = row_1.iloc[: ,128 : 256]
print(y)
n = 128
print(y)

ffty = np.fft.rfft(y)
#fftx = np.fft.fftfreq(n, 1/5000)[:n//2]
fftx = np.fft.rfftfreq(n, 1/5000)
#default sample space (inverse of sampling rate) is 1

print(ffty[0])
print(fftx)

plt.grid()
#abs_ffty = 2.0/n * np.abs(ffty[0:n//2])
#plt.plot(fftx, abs_ffty[0])
plt.plot(fftx, np.abs(ffty[0]))
plt.show()



# 
# Generic Approach
# 0. Get a sample window and sampling rate to use in FFT
# 1. Convert time domain signal to frequency domain using Fast Fourier Transform (fft is library in python)
# 2. Analyse the frequency and apply filter (As per Prof use band pass filter with min. frequency of 30kHz and max. frequency of 50kHz), again get back the signal to time domain using Reverse Fast Fourier Transform
# 3. Use method called as "Envelope of Signal" (Hilbert Transform is one method) which gives you max and min of the whole singal at every point in time
# 4. Label the new data at every point in time
# 5. Use these label data and feed it to your neural network
#

