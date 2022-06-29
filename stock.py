import math
import numpy as np
import pandas_datareader as datareader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

TICKER = 'SPY'
START = '2019-01-01'
END = '2022-06-18'

data = datareader.DataReader(TICKER, data_source='yahoo', start=START, end=END)
print(data.shape)

filteredData = data.filter(['Close'])
dataset = filteredData.values
trainingDataLen = len(dataset)
scaler = MinMaxScaler(feature_range=(0,1))
scaledData = scaler.fit_transform(dataset)

trainingData = scaledData[0:trainingDataLen, :]
xTrain = []
yTrain = []

for i in range(60, trainingDataLen):
    xTrain.append(trainingData[i-60:i, 0])
    yTrain.append(trainingData[i, 0])

xTrain, yTrain = np.array(xTrain), np.array(yTrain)
print(xTrain.shape)

xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(xTrain.shape[1], 1)))
model.add(LSTM(100, return_sequences=False))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adagrad', loss='mean_squared_error')
model.fit(xTrain, yTrain, batch_size=5, epochs=10)

quote = datareader.DataReader(TICKER, data_source='yahoo', start=START, end=END)
newdf = quote.filter(['Close'])

last60 = newdf[-60:].values
last60Scaled = scaler.transform(last60)

xTest = []
xTest.append(last60Scaled)
xTest = np.array(xTest)
xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))
predPrice = model.predict(xTest)
predPrice = scaler.inverse_transform(predPrice)
print(predPrice)

