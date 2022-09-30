from CNN_Model import CNNModel
import tensorflow as tf
from sklearn.model_selection import train_test_split

def main():
  dataset = "dataset\Wand_000.xlsx"
  dataset1 = "dataset\T_Wand_000.xlsx" #test data with 2 index (or rows)
  print('Reading dataset: ', dataset)
  obj = CNNModel(dataset)
  print('Reducing noise and labelling data...')
  x_data = obj.reduce_noise_and_label()
  print('Grouping labelled data...')
  y_data = obj.group_labeled_data()
  xtrain, xtest, ytrain, ytest=train_test_split(x_data, y_data, test_size=0.25)
  print('Training Model...')
  obj.train_model(xtrain, xtest, ytrain, ytest)
  

def check_GPUs():
  print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Calling main function
if __name__=="__main__":
  check_GPUs()
  main()