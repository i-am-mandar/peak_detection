#predict using saved model
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd

dataset = "dataset\T_Wand_000.xlsx"
df = pd.read_excel(dataset)
df = df.iloc[0: ,17:]

f = df.iloc[0,0:]

x = f.values.reshape(1,-1)

print(x.shape)

model = keras.models.load_model('trained_model')
predict_y=model.predict(x)
classes_y=np.argmax(predict_y,axis=1)
n_slice = 62

peak = classes_y[0] * n_slice

print(peak)