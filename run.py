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


# Added by Rizwan for getting 2 sine wave, normalized it and used fftfreq to plot
# import numpy as np 
# import scipy, matplotlib
# from matplotlib import pyplot as plt
# from scipy.fft import fft, fftfreq

# SAMPLE_RATE = 44100  # Hertz
# DURATION = 5  # Seconds



# def generate_sine_wave(freq, sample_rate, duration):
    # x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    # frequencies = x * freq
    # # 2pi because np.sin takes radians
    # y = np.sin((2 * np.pi) * frequencies)
    # return x, y

# # Generate a 2 hertz sine wave that lasts for 5 seconds
# #x, y = generate_sine_wave(2, SAMPLE_RATE, DURATION)

# _, nice_tone = generate_sine_wave(400, SAMPLE_RATE, DURATION)
# _, noise_tone = generate_sine_wave(4000, SAMPLE_RATE, DURATION)
# noise_tone = noise_tone * 0.3

# mixed_tone = nice_tone + noise_tone
# normalized_tone = np.int16((mixed_tone / mixed_tone.max()) * 32767)

# N = SAMPLE_RATE * DURATION

# yf = fft(normalized_tone)
# xf = fftfreq(N, 1 / SAMPLE_RATE)

# plt.plot(xf, np.abs(yf))
# plt.show()
# end

# 
# Generic Approach
# 0. Get a sample window and sampling rate to use in FFT
# 1. Convert time domain signal to frequency domain using Fast Fourier Transform (fft is library in python)
# 2. Analyse the frequency and apply filter (As per Prof use band pass filter with min. frequency of 30kHz and max. frequency of 50kHz), again get back the signal to time domain using Reverse Fast Fourier Transform
# 3. Use method called as "Envelope of Signal" (Hilbert Transform is one method) which gives you max and min of the whole singal at every point in time
# 4. Label the new data at every point in time
# 5. Use these label data and feed it to your neural network
#

