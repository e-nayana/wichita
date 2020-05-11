import math
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


class ModelTrainer:

    def __init__(self, model_label, currency_pair_history_data_location=''):
        self.model_label = model_label
        self.currency_pair_history_data_location = currency_pair_history_data_location

        self.base_directory = os.getcwd() + "\willy"
        self.trained_lstm_dump_location = self.base_directory + '\storage_app' + '\model_dump_' + model_label + '.sav'

        self.scaler = None
        self.model = None
        self.price_frame = None

    def init_model(self, feature_range):
        self.scaler = MinMaxScaler(feature_range=feature_range)

        # Prepare prices
        data = pd.read_csv(self.currency_pair_history_data_location)
        data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        data.date = pd.to_datetime(data.date, format='%d.%m.%Y %H:%M:%S.%f')
        data = data.set_index(data.date)
        data = data[['open', 'high', 'low', 'close', 'volume']]
        prices = data.drop_duplicates(keep=False)
        prices = prices[prices.high != prices.low]

        # prices = prices.iloc[:320]

        # Add calculated features
        # Open close mean
        open_close_mean = prices[['open', 'close', 'high', 'low']].sum(axis=1) / 4

        prices['open_close_mean'] = open_close_mean

        self.price_frame = prices.dropna()

    def test(self):
        print(self.base_directory)

    def train_lstm(self, feature_reverse_count=120):

        data_set = self.price_frame[['close', 'open_close_mean']].values

        training_data_set_len = math.ceil(len(data_set) * 0.8)
        data_set = np.reshape(data_set, (data_set.shape[0], data_set.shape[1]))
        scaled_data_set = self.scaler.fit_transform(data_set)

        ########################################################################################################################

        training_data_set = scaled_data_set[0:training_data_set_len, :]
        x_train = []
        y_train = []
        for i in range(feature_reverse_count, training_data_set_len):
            x_train.append(training_data_set[i - feature_reverse_count:i, :])
            y_train.append(training_data_set[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

        model = Sequential()
        model.add(LSTM(150, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dense(100))
        model.add(Dense(50))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=1, epochs=2)

        joblib.dump(model, self.trained_lstm_dump_location)

    def re_load_lstm(self):

        data_set = self.price_frame.close.values
        training_data_set_len = math.ceil(len(data_set) * 0.8)
        data_set = np.reshape(data_set, (data_set.shape[0], 1))

        scaled_data_set = self.scaler.fit_transform(data_set)

        #############################################################################################################

        model = joblib.load(self.trained_lstm_dump_location)

        test_data = scaled_data_set[training_data_set_len - 120:, :]

        x_test = []
        y_test = data_set[training_data_set_len:, :]

        for i in range(120, len(test_data)):
            x_test.append(test_data[i - 120:i, :])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        prediction = model.predict(x_test)
        prediction = self.scaler.inverse_transform(prediction)

        rmse = np.sqrt(np.mean(prediction - y_test) ** 2)  # 0.0011576784373741936
        print(rmse)

        trained_data = self.price_frame[:training_data_set_len]
        valid = self.price_frame[training_data_set_len:]
        valid['predict'] = prediction

        plt.figure(figsize=(16, 8))
        plt.title('Prediction visualization')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('EURUSD', fontsize=18)
        # plt.plot(trained_data['close'])
        plt.plot(valid[['close', 'predict']])
        plt.legend(['Train', 'Valid', 'Predic'], loc='lower right')
        plt.show()

    def predict_long_future(self, future_period=2, feature_reverse_count=120):

        data_set = self.price_frame[['close', 'open_close_mean']].values
        training_data_set_len = math.ceil(len(data_set) * 0.8)
        data_set = np.reshape(data_set, (data_set.shape[0], data_set.shape[1]))

        scaled_data_set = self.scaler.fit_transform(data_set)

        #############################################################################################################

        model = joblib.load(self.trained_lstm_dump_location)

        y_real = self.price_frame.iloc[training_data_set_len:training_data_set_len+future_period]
        y_predicting = []


        x_test_scaled = [scaled_data_set[training_data_set_len - feature_reverse_count:training_data_set_len, :]]

        for i in range(0, future_period):
            # prepare x for feed the model for prediction
            x_test_scaled_ready = np.array(x_test_scaled)
            x_test_scaled_ready = np.reshape(x_test_scaled_ready, (x_test_scaled_ready.shape[0], x_test_scaled_ready.shape[1], x_test_scaled_ready.shape[2]))
            predicted_y_scaled = model.predict(x_test_scaled_ready)

            # remove the first element and insert the last element which just has been predicted (predicted_y_scaled)
            x_test_scaled = x_test_scaled[0][1:, :]
            # x_test_scaled = [np.append(x_test_scaled, predicted_y_scaled, axis=0)]

            # Preparing predicted_y_scaled to inverse transform
            for j in range(1, 2):
                predicted_y_scaled = [np.append(predicted_y_scaled, 0)]


            # insert predicted value in to the result array
            predicted_y = self.scaler.inverse_transform(predicted_y_scaled)
            y_predicting.append(predicted_y[0][0])

        rmse = np.sqrt(np.mean(y_predicting - y_real.close.values) ** 2)  # 0.0011576784373741936
        print(rmse)

        # trained_data = prices[:training_data_set_len]
        valid = y_real
        valid['predict'] = y_predicting
        # #
        plt.figure(figsize=(16, 8))
        plt.title('Prediction visualization')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('EURUSD', fontsize=18)
        # # plt.plot(trained_data['close'])
        plt.plot(valid[['close', 'predict']])
        plt.legend(['Valid', 'Predic'], loc='lower right')
        plt.show()







