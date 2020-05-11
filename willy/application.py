from willy.lstm_trainer import ModelTrainer

model_trainer = ModelTrainer('HOURLY', 'K:\python\wichita\willy\EURUSD.csv')

model_trainer.init_model((0, 1))
model_trainer.train_lstm()
#
# model_trainer.predict_long_future(1)
# model_trainer.re_load_lstm()

