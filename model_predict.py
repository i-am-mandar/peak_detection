#predict using saved model
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd

def predict(in_data):

  reshape_in_data = in_data.values.reshape(1,-1)
  model = keras.models.load_model('trained_model')
  predict_y=model.predict(reshape_in_data)
  classes_y=np.argmax(predict_y,axis=1)
  n_slice = 62
  
  peak = classes_y[0] * n_slice
  print(peak)
  adjust = 50
  return peak + adjust