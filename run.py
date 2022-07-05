import pandas as pd
import matplotlib.pyplot as plt

dataset = "dataset\T_File_5.xlsx"
df = pd.read_excel(dataset)

df = df.iloc[0: ,6:]
#print(df)

for i in range(len(df)):
    irow = df.iloc[i,0:]
    irow.plot()

#LWP - is not useful
#smooth_data = pd.Series(row).rolling(window=7).mean().plot(style='k')
#smooth_data = pd.Series(row).rolling(window=7).mean().plot()

#implement band pass filter here



plt.show()



# 
# Generic Approach
# 1. Convert time domain signal to frequency domain using Fast Fourier Transform (fft is library in python)
# 2. Analyse the frequency and apply filter (As per Prof use band pass filter with min. frequency of 30kHz and max. frequency of 50kHz), again get back the signal to time domain using Reverse Fast Fourier Transform
# 3. Use method called as "Envelope of Signal" (Hilbert Transform is one method) which gives you max and min of the whole singal at every point in time
# 4. Label the new data at every point in time
# 5. Use these label data and feed it to your neural network
#
# sliding window with , atleast half of the length, start of signal, hamming window / alized frequency (fake frequency) this is before fft
# About the dataset
# ->Skip columns from A to F. 
# ->Each row comprises a sampled time signal.
#
# -------------------------------------------------
# old info
# -> know the frequency for LPF
# -> must band pass filter 30khz to 50kHz
# -> envelope of the signal 
# -> method
# -> right to left 
# hilbert transform for envelope, best get an envelope
# 
# sliding window
# 
# rectifying layer cnn
#
#----------------------------------------------
# you might need to take a look
# -> https://github.com/avhn/peakdetect
#
#
#
#
