import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import numpy as np
from scipy.signal import find_peaks

df = pd.read_excel('dataset/T_File_5.xlsx')
for i in range(0, len(df.index)):
  signal = df.iloc[i,6:]
  signal.index = list(range(3400))

  signal = signal.rolling(window=10).mean()
  signal.fillna(0, inplace=True)
  signal.plot()

  analytical_signal = hilbert(signal)
  env = np.abs(analytical_signal)
  pd.Series(env).plot()

  x, _ = find_peaks(env, height=0.02, width=20)

  print(x)
  plt.plot(x, env[x], "x")
  plt.show()
print("done")


# other approach
# 1. rollowing average filter
# 2. fill 0 if nan
# 3. apply hilbert transform to get upper envelope and take abs avlue
# 4. find peaks
