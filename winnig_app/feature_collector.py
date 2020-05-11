import pandas as pd
from winnig_app import feature_function

# Load csv again
data = pd.read_csv('K:\python\wichita\winnig_app\EURUSD.csv')
data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
data = data.set_index(pd.to_datetime(data.date))
data = data[['date', 'open', 'high', 'low', 'close', 'volume']]

prices = data.drop_duplicates(keep=False)
prices = prices[prices.high != prices.low]

# Create time periods lists
momentum_key = [3, 4, 5, 8, 9, 10]
stochastic_key = [3, 4, 5, 8, 9, 10]
williams_key = [6, 7, 8, 9, 10]
proc_key = [12, 13, 14, 15]
wadl_key = [15]
adosc_key = [2, 3, 4, 5]
macd_key = [15, 30]
cci_key = [15]
bollinger_key = [15]
heikenashi_key = [15]
price_average_key = [2]
slop_key = [3, 4, 5, 10, 20, 30]
fourier_key = [10, 20, 30]
sin_key = [5, 6]

key_list = [momentum_key, stochastic_key, williams_key, proc_key, wadl_key, adosc_key, macd_key, cci_key, bollinger_key, heikenashi_key, price_average_key, slop_key, fourier_key, sin_key]

# Calculate features
momentum_dict = feature_function.momentum(prices, momentum_key)
print('1 - completed')
stochastic_dict = feature_function.stochastic(prices, stochastic_key)
print('2 - completed')
williams_dict = feature_function.williams_R(prices, williams_key)
print('3 - completed')
proc_dict = feature_function.proc(prices, proc_key)
print('4 - completed')
wadl_dict = feature_function.wadl(prices, wadl_key)
print('5 - completed')
adosc_dict = feature_function.adosc(prices, adosc_key)
print('6 - completed')
macd_dict = feature_function.macd(prices, macd_key)
print('7 - completed')
cci_dict = feature_function.cci(prices, cci_key)
print('8 - completed')
bollinger_dict = feature_function.bollinger_band(prices, bollinger_key, 2)
print('9 - completed')

'''
Special calculations for heiken ashi
'''
hk_prices = prices.copy()
hk_prices['Symbol'] = 'SYMB'

hka_resampled_prices = feature_function.OHLC_resample(hk_prices, '15H')

heikenashi_dict = feature_function.heike_nashi(hka_resampled_prices, heikenashi_key)
print('10 - completed')
price_average_dict = feature_function.price_average(prices, price_average_key)
print('11 - completed')
slop_dict = feature_function.price_slop(prices, slop_key)
print('12 - completed')
fourier_dict = feature_function.fourier(prices.close, fourier_key)
print('13 - completed')
sin_dict = feature_function.sin(prices.close, sin_key)
print('14 - completed')

dict_list = [momentum_dict.close, stochastic_dict.result_periods, williams_dict.result_periods, proc_dict.result_periods, wadl_dict.wadl, adosc_dict.result_periods, macd_dict.macd_line, cci_dict.result_periods,
             bollinger_dict.result_periods, heikenashi_dict.ha_candles, price_average_dict.result_periods, slop_dict.result_periods, fourier_dict.coeffs, sin_dict.coeffs]
#
base_feature_column_names = ['momentum', 'stochasctic', 'williams', 'proc', 'wadl', 'adosc', 'macd', 'cci', 'bollinger', 'heikenashi', 'paverage','slop', 'fourier', 'sin']
master_frame = pd.DataFrame(index=prices.index)

for i in range(0, len(dict_list)):

    print('setting feature - ' + base_feature_column_names[i])

    if base_feature_column_names[i] == 'macd':
        column_id = base_feature_column_names[i] + str(key_list[i][0]) + str(key_list[i][1])
        master_frame[column_id] = dict_list[i]

    else:
        for j in dict_list[i]:
            for k in list(dict_list[i][j]):
                column_id = base_feature_column_names[i] + str(j) + k
                master_frame[column_id] = dict_list[i][j][k]

threshole = round(0.7*len(master_frame))

master_frame[['date', 'open', 'high', 'low', 'close']] = prices[['date', 'open', 'high', 'low', 'close']]

master_frame.heikenashi15open = master_frame.heikenashi15open.fillna(method='bfill')
master_frame.heikenashi15high = master_frame.heikenashi15high.fillna(method='bfill')
master_frame.heikenashi15low = master_frame.heikenashi15low.fillna(method='bfill')
master_frame.heikenashi15close = master_frame.heikenashi15close.fillna(method='bfill')


# Drop columns that have 30% or more NAN data
master_frame_cleaned = master_frame.copy()
master_frame_cleaned = master_frame_cleaned.dropna(axis=1, thresh=threshole)
master_frame_cleaned = master_frame_cleaned.dropna(axis=0)

master_frame_cleaned.to_csv('K:\python\wichita\winnig_app\EURUSD_feature_mater_frame.csv')

print('Features preparation completed')















