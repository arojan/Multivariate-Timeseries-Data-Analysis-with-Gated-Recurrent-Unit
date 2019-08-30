#!/usr/bin/env python
# coding: utf-8

"""
Created on Fri Aug 30 15:05:40 2019

@author: Rojan
"""


# Disabling any minor warmings and logs
import warnings
warnings.filterwarnings("ignore")


# Importing necessary libraries
import h5py
import numpy as np
from numpy import concatenate

import pandas as pd
from pandas import read_csv, concat, DataFrame

from matplotlib import pyplot
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from keras.models import Sequential 
from keras.layers import Dense, Dropout, GRU


# Importing the dataset along with some changes
data = read_csv("household_power_consumption.csv",
                   parse_dates={'dt' : ['Date', 'Time']},
                   infer_datetime_format=True, 
                   index_col= 0,
                   na_values=['nan', '?'])


# Replacing the '?' median of the data
median_value = data.median()
data = data.fillna(median_value)

# Ensuring all dataset is in float
values = data.values
values = values.astype('float32')


# normalizing input features
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(values)
scaled = pd.DataFrame(scaled)


# Creating a timeseries function to convert the timeseries dataset
def timeseries_data(data, lookback=1, predicted_col=1):
    temp=data.copy()
    temp["id"]= range(1, len(temp)+1)
    temp = temp.iloc[:-lookback, :]
    temp.set_index('id', inplace =True)
    
    predicted_value=data.copy()
    predicted_value = predicted_value.iloc[lookback:,predicted_col]
    predicted_value.columns=["Predicted"]
    predicted_value= pd.DataFrame(predicted_value)
    
    predicted_value["id"]= range(1, len(predicted_value)+1)
    predicted_value.set_index('id', inplace =True)
    final_df= pd.concat([temp, predicted_value], axis=1)
    return final_df


# Reframing the dataset
reframed_df= timeseries_data(scaled, 1,0)
reframed_df.fillna(0, inplace=True)
reframed_df.columns = ['var1(t-1)', 'var2(t-1)', 'var3(t-1)', 'var4(t-1)', 'var5(t-1)', 'var6(t-1)', 'var7(t-1)','var1(t)']


# Spliting the dataset into train and test sets
values = reframed_df.values
train_ = values[:, :-1]
labels = values[:, -1]
train_X, test_X, train_y, test_y = train_test_split(train_, labels, test_size=0.27, random_state=0)


# Reshaping input to be 3D [samples, time steps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

batch_size = 64
epochs = 15

# Creating a model out of a sequential model and appending the LSTM layers for timeseries computations
model = Sequential()
model.add(GRU(20, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(Dropout(0.01))
model.add(GRU(40))
model.add(Dropout(0.01))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Printing the model summary including parameters of the designed network
model.summary()


# Training the model
model_ = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=1, 
                   validation_data=(test_X, test_y), shuffle=False)


# Saving the final model at the end of the training
model.save("Model/model.hdf5")
print("Final model saved successfully!")


# Plotting/saving the graph on the loss and validation loss of the model throughout the training process
pyplot.plot(model_.history['loss'], label='LSTM training', color='red')
pyplot.plot(model_.history['val_loss'], label='LSTM testing', color= 'blue')
pyplot.xlabel('Number of epochs')
pyplot.ylabel('Loss Metrics')
pyplot.title(f'Loss vs Validation Loss Chart')
pyplot.legend()
pyplot.savefig(fname=f'Graph/loss chart.png', dpi = 350)
pyplot.show()
