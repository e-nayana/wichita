import math

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Load csv
data = pd.read_csv('/willy/EURUSD.csv')
data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

data.date = pd.to_datetime(data.date, format='%d.%m.%Y %H:%M:%S.%f')
data = data.set_index(data.date)
data = data[['open', 'high', 'low', 'close', 'volume']]
#
prices = data.drop_duplicates(keep=False)
prices = prices[prices.high != prices.low]




filename = 'K:\python\wichita\winnig_app\model_stored.sav'
data_set = prices.close.values
training_data_set_len = math.ceil(len(data_set) * 0.8)

data_set = np.reshape(data_set, (data_set.shape[0], 1))

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data_set = scaler.fit_transform(data_set)

training_data_set = scaled_data_set[0:training_data_set_len, :]

x_train = []
y_train = []
for i in range(60, training_data_set_len):
    x_train.append(training_data_set[i-60:i, 0])
    y_train.append(training_data_set[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Resahpe the data as LSTM required 3d data
x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))


# model = Sequential()
# model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# model.add(LSTM(50, return_sequences=False))
# model.add(Dense(25))
# model.add(Dense(1))
#
# model.compile(optimizer='adam', loss='mean_squared_error')
# model.fit(x_train, y_train, batch_size=1, epochs=1)
#
#
# joblib.dump(model, filename)
model = joblib.load(filename)


test_data = scaled_data_set[training_data_set_len - 60: , :]

x_test = []
y_test = data_set[training_data_set_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, :])


x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
prediction = model.predict(x_test)
prediction = scaler.inverse_transform(prediction)

rmse = np.sqrt(np.mean(prediction-y_test)**2) #0.0011576784373741936
print(rmse)


trained_data = prices[:training_data_set_len]
valid = prices[training_data_set_len:]
valid['predict'] = prediction

plt.figure(figsize=(16, 8))
plt.title('Prediction visualization')
plt.xlabel('Date', fontsize=18)
plt.ylabel('EURUSD', fontsize=18)
# plt.plot(trained_data['close'])
plt.plot(valid[['close', 'predict']])
plt.legend(['Train', 'Valid', 'Predic'], loc='lower right')
plt.show()




















