#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from scipy.fft import fft, fftfreq
#
#dataset = "dataset\T_File_5.xlsx"
#df = pd.read_excel(dataset)
#
#df = df.iloc[0: ,6:]
##print(df)
#
##length of column
#col_len = len(df.columns)
#print(col_len)
#
##length of row
#row_len = len(df.index)
##print(row_len)
#
##first row of df
#row_1 = df.head(1)
##425 * 8 = 3400
#step = 425
#for i in range(step, col_len+1,step):
#    y = row_1.iloc[: ,(i-step) : i]
#    n = y.size
#    timestep = 0.01

#    ffty = np.fft.rfft(y)

#    fftx = np.fft.rfftfreq(n, d=timestep)
#    print(len(ffty[0]))
#    print(len(fftx))

#    plt.grid()
#    plt.plot(fftx, np.abs(ffty[0]))
#    plt.show()


#Added by Rizwan for getting 2 sine wave, normalized it and used fftfreq to plot
import numpy as np 
import scipy, matplotlib
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq

SAMPLE_RATE = 44100  # Hertz
DURATION = 5  # Seconds



def generate_sine_wave(freq, sample_rate, duration):
  x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
  frequencies = x * freq
  # 2pi because np.sin takes radians
  y = np.sin((2 * np.pi) * frequencies)
  return x, y

# Generate a 2 hertz sine wave that lasts for 5 seconds
x, y = generate_sine_wave(2, SAMPLE_RATE, DURATION)

_, nice_tone = generate_sine_wave(400, SAMPLE_RATE, DURATION)
_, noise_tone = generate_sine_wave(4000, SAMPLE_RATE, DURATION)
noise_tone = noise_tone * 0.3

mixed_tone = nice_tone + noise_tone
normalized_tone = np.int16((mixed_tone / mixed_tone.max()) * 32767)

N = SAMPLE_RATE * DURATION

yf = fft(normalized_tone)
print(yf[395:405])
xf = fftfreq(N, 1 / SAMPLE_RATE)
#The maximum frequency is half the sample rate
points_per_freq = len(xf) / (SAMPLE_RATE / 2)

# Our target frequency is 4000 Hz
target_idx = int(points_per_freq * 4000)

yf[target_idx - 1 : target_idx + 2] = 0
plt.plot(xf, np.abs(yf))
plt.show()
end

# 
# Generic Approach
# 0. Get a sample window and sampling rate to use in FFT
# 1. Convert time domain signal to frequency domain using Fast Fourier Transform (fft is library in python)
# 2. Analyse the frequency and apply filter (As per Prof use band pass filter with min. frequency of 30kHz and max. frequency of 50kHz), again get back the signal to time domain using Reverse Fast Fourier Transform
# 3. Use method called as "Envelope of Signal" (Hilbert Transform is one method) which gives you max and min of the whole singal at every point in time
# 4. Label the new data at every point in time
# 5. Use these label data and feed it to your neural network
#

