#approach 3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import hilbert
from scipy.signal.windows import hamming

dataset = "dataset\T_File_5.xlsx"
df = pd.read_excel(dataset)

df = df.iloc[0: ,6:]
#print(df)

#for i in range(0, len(df.index)):
  #f= df.iloc[i,0:]

f= df.iloc[0,0:]
n = f.size          #size of the signal
time=np.arange(n)   #time of the signal

#1. hamming winfow of n
window = hamming(n, sym=True)

#2. discarding any frequency below power 1.5 and calculate inverse fft
denoised_signal = f * window

#3. using hilbert transformation to calculate upper envelope
analytical_signal = hilbert(denoised_signal)
env = np.abs(analytical_signal)
x, _ = find_peaks(env, height=0.02, width=20)

fig, axs = plt.subplots(2,1)

#4. plot orignal noisy signal
plt.sca(axs[0])
plt.plot(time,f, label='Noisy')
plt.xlim(time[0], time[-1])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()


#5. plot filtered signal with upper envelope and peaks marked as 'x'
plt.sca(axs[1])
plt.plot(time, denoised_signal, label='Filtered Signal')
plt.plot(time, env, label='Envelope')
plt.plot(x, env[x], "x")
plt.xlim(time[0], time[-1])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()

plt.show()

#6. save the real value of peaks
peaks = np.zeros(n, dtype=float)
peaks[x] = np.abs(denoised_signal[x])
#x = [ 935 2105 3140]

#7. normalize the peak values between 0 to 1
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


