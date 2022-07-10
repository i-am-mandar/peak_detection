# peak_detection

Generic Approach
0. Get a sample window and sampling rate to use in FFT
1. Convert time domain signal to frequency domain using Fast Fourier Transform (fft is library in python)
2. Analyse the frequency and apply filter (As per Prof use band pass filter with min. frequency of 30kHz and max. frequency of 50kHz), again get back the signal to time domain using Reverse Fast Fourier Transform
3. Use method called as "Envelope of Signal" (Hilbert Transform is one method) which gives you max and min of the whole singal at every point in time
4. Label the new data at every point in time
5. Use these label data and feed it to your neural network
