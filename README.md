# peak_detection

Goal is to detect peak in given signal(s) in directory `\dataset` with computation intelligence and not just signal processing.

## approach A

0. read from xlsx into dataframe

1. calculate fft and take complex conjugate to eleminate imaginary value

2. discarding any frequency below power 1.5 and calculate inverse fft

3. using hilbert transformation to calculate upper envelope

4. plot orignal noisy signal (code commented)

5. plot FFT of noisy and filtered signal (code commented)

6. plot filtered signal with upper envelope and peaks marked as 'x' (code commented)

7. save the real value of peaks 

8. group the peaks in chunk of n_slice = 62

9. create a label as `((len(df.index),len(df.columns)//n_slice))` and mark the cell as 1 where the peak is present

10. create a 1D CNN network and put in the input with label

11. save the modeled CNN netowork for prediction

12. give test input and get peak


## approach B

0. read test input into dataframe

1. using scipy.signal.find_peak() and get peak


## compare results from approach A and B and display results in web based UI

0. Run `app.py` to get web based UI to upload sample file and  get results


## how to run the experiment

0. Run `pip install -r requirement.txt` and get all libs

1. Run `python model_train.py` so we have the trained model save

2. Run python `app.py` to get web based UI and upload sample input from `\static\files` and get comparision

## results

0. input - `\static\files\sample_input_001.xlsx`

![image](https://github.com/i-am-mandar/peak_detection/blob/mandar/results/result_sample_input_001.png)


1. input - `\static\files\sample_input_002.xlsx`

![image](https://github.com/i-am-mandar/peak_detection/blob/mandar/results/result_sample_input_002.png)

Note: You can always run each file `CNN_Model.py`, `model_train.py` and `model_predict.py` for individual output 