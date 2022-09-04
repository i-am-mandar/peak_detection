#approach 3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import hilbert

dataset = "dataset\T_File_5.xlsx"
df = pd.read_excel(dataset)

df = df.iloc[0: ,6:]
#print(df)

#for i in range(0, len(df.index)):
  #f= df.iloc[i,0:]

f= df.iloc[0,0:]
n = f.size          #size of the signal
dt= 0.05            #randomly choosen sampling rate
time=np.arange(n)   #time of the signal

#1. calculate fft and take complex conjugate to eleminate imaginary value
fhat = np.fft.fft(f,n)
PSD = fhat * np.conj(fhat) / n
freq = (1/(dt*n))*np.arange(n)
L = np.arange(0, n//2, dtype='int')

#2. discarding any frequency below power 1.5 and calculate inverse fft
indices = PSD > 1.5
PSDclean = PSD * indices
fhat = indices * fhat
ffilt = np.fft.ifft(fhat)

#3. using hilbert transformation to calculate upper envelope
analytical_signal = hilbert(ffilt.real)
env = np.abs(analytical_signal)
x, _ = find_peaks(env, height=0.02, width=20)

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

#plt.show()

#7. save the real value of peaks
peaks = np.zeros(n, dtype=float)
peaks[x] = np.abs(ffilt[x].real)
#x = [ 935 2105 3140]

#8. normalize the peak values between 0 to 1
fuzzy = []
t_max = 1
t_min = 0
diff = t_max - t_min
diff_arr = max(peaks) - min(peaks)
for i in peaks:
  temp = (((i - min(peaks))*diff)/diff_arr) + t_min
  fuzzy.append(temp)

#print(np.nonzero(fuzzy))

print("done")


#approach
#1. calculate fft and take complex conjugate to eleminate imaginary value
#2. discarding any frequency below power 1.5 and calculate inverse fft
#3. using hilbert transformation to calculate upper envelope
#4. plot orignal noisy signal
#5. plot FFT of noisy and filtered signal
#6. plot filtered signal with upper envelope and peaks marked as 'x'
#7. save the real value of peaks
#8. normalize the peak values between 0 to 1
