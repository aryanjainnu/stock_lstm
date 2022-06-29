import math
import numpy as np
import pandas_datareader as datareader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

TICKER = 'AAPL'
START = '2019-01-01'
END = '2022-06-27'

data = datareader.DataReader(TICKER, data_source='yahoo', start=START, end=END)

"""
THIS WILL SHOW THE CLOSE PRICE HISTORY
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD($)',fontsize=18)
plt.show()
"""

filteredData = data.filter(['Close'])
dataset = filteredData.values
trainingDataLen = math.ceil(len(dataset) * 0.8)

scaler = MinMaxScaler(feature_range=(0, 1))
scaledData = scaler.fit_transform(dataset)

trainingData = scaledData[0:trainingDataLen, :]
xTrain = []
yTrain = []

for i in range(60, trainingDataLen):
    xTrain.append(trainingData[i - 60:i, 0])
    yTrain.append(trainingData[i, 0])

xTrain, yTrain = np.array(xTrain), np.array(yTrain)

xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(xTrain.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xTrain, yTrain, batch_size=1, epochs=10)

test_data = scaledData[trainingDataLen - 60:, :]

xTest = []
yTest = dataset[trainingDataLen:, :]
for i in range(60, len(test_data)):
    xTest.append(test_data[i - 60:i, 0])

xTest = np.array(xTest)
xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))
predictions = model.predict(xTest)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean((predictions - yTest) ** 2))

train = data[:trainingDataLen]
valid = data[trainingDataLen:]
valid['Predictions'] = predictions
"""
TESTING VS. REAL PLOT
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
"""

apple_quote = datareader.DataReader(TICKER, data_source='yahoo', start=START, end=END)
new_df = apple_quote.filter(['Close'])
last_60_days = new_df[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)
next_day = []
next_day.append(last_60_days_scaled)
next_day = np.array(next_day)
next_day = np.reshape(next_day, (next_day.shape[0], next_day.shape[1], 1))
pred_price = model.predict(next_day)
pred_price = scaler.inverse_transform(pred_price)

print(pred_price)
