# Description : This program uses an artificial reccurent neural network callef Long Short Term Memory (LSTM)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

dataset_train = pd.read_csv('K:\python\wichita\price_perdictor_simple\EURUSD.csv')
training_set = dataset_train.iloc[:, 1:2]
training_set = training_set.drop_duplicates(keep=False)

training_set = training_set.values

scalor = MinMaxScaler(feature_range=(0, 1))
scaled_training_dataset = scalor.fit_transform(training_set)

x_train = []
y_train = []
for i in range(60, len(training_set)-1):
    x_train.append(scaled_training_dataset[i-60:i, 0])
    y_train.append(scaled_training_dataset[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

#
# # np_array.shap() ----- get the shape of the array
# # np_array.reshape(arry, (number_of_row, number_of_columns, leafs))
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# #Initialize the RNN
regressor = Sequential()

# # #Add the first LSTM layer and some dropout regulations
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#Add the second LSTM layer and some dropout regulations
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

#Add the third LSTM layer and some dropout regulations
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

 #Add the fourth LSTM layer and some dropout regulations
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

#add the output layer
regressor.add(Dense(units=1))

#compile the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

regressor.fit(x_train, y_train, epochs=100, batch_size=32)

#Getting the prediction and visualize the result
#Getting the real stock price in 2019-05-01 to 2020-04-30
# dataset_test = pd.read_csv('K:\python\wichita\price_perdictor_simple\data_real.csv')
# real_stock_price = dataset_test.iloc[:, 1:2].values

# #Getting predicted stock price of above range
# dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
# inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
# inputs = inputs.reshape(-1, 1)
# inputs = scalor.transform(inputs)
# x_test = []
#
# for i in range(60, len(inputs)-1):
#     x_test.append(inputs[i-60:i, 0])
#
# x_test = np.array(x_test)
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
#
# predicted_stock_price = regressor.predict(x_test)
# predicted_stock_price = scalor.inverse_transform(predicted_stock_price)
#
# #Plot the real and predicted result
#
#
# plt.plot(real_stock_price, color='red', label='Real')
# plt.plot(predicted_stock_price, color='green', label='Predicted')
# plt.title('EURUSD')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# plt.show()























