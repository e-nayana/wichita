import numpy as np
import pandas as pd
import plotly as py
from plotly import tools
import plotly.graph_objs as go

from winnig_app import feature_function

df = pd.read_csv('/willy/EURUSD.csv')
df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

df.date = pd.to_datetime(df.date, format='%d.%m.%Y %H:%M:%S.%f')
df = df.set_index(df.date)
df = df[['open', 'high', 'low', 'close', 'volume']]
df = df.drop_duplicates(keep=False)
df = df.iloc[:1000]

print("=====================================features========================================")
'''
Original data
'''
trace_original = go.Candlestick(x=df.index, open=df.open, high=df.high, low=df.low, close=df.close, name="EURUSD Forex")

'''
MY CALCULATION
'''
# open_close_mean = df[['open', 'close']].sum(axis=1)/2           open_close_mean = df[['open', 'close', 'high', 'low']].sum(axis=1)/4
# ma_of_mean = open_close_mean.ewm(span=15, adjust=False).mean()


'''
HEIKE NASHI
'''
# ha_result = feature_function.heike_nashi(df, ['H'])
# ha = ha_result.ha_candles['H']
# trace_ha = go.Candlestick(x=ha.index, open=ha.open, high=ha.high, low=ha.low, close=ha.close, name="EURUSD Forex HA(H)")

'''
FOURIER AND SIN
'''
# fourier_fittings = feature_function.sin(ma_of_mean, [20],  'defference')
# print(fourier_fittings.coeffs)

'''
WADL
'''
# WADL_result = feature_function.wadl(df, [15])
# WADL_line = WADL_result.wadl[15]
# print(WADL_line)
# WADL_trace = go.Scatter(x=WADL_line.index, y=WADL_line.line, marker_color='green')

'''
RESAMPLING
'''
# resampled = feature_function.OHLC_resample(df, '15H')

'''
MOMENTUM
'''
# close_momentum = feature_function.momentum(df, [10]).close[10]
# close_momentum_trace = go.Scatter(x=close_momentum.index, y=close_momentum.close, marker_color='green')

'''
STOCHASTIC
'''
# stochastic_result = feature_function.stochastic(df, [14]).result_periods[14]
# stochastic_trace = go.Scatter(x=stochastic_result.index, y=stochastic_result.K, marker_color='green')
# stochastic_trace = go.Scatter(x=stochastic_result.index, y=stochastic_result.D, marker_color='green')

'''
WILLIAMS
'''
# williams_r = feature_function.williams_R(df, [15]).result_periods[15]
# williams_r_trace = go.Scatter(x=williams_r.index, y=williams_r.line, marker_color='green')

'''
PROC
'''
# proc = feature_function.proc(df, [30]).result_periods[30]
# print(proc)
# proc_trace = go.Scatter(x=proc.index, y=proc.line, marker_color='green')

'''
ADOSC
'''
# adosc = feature_function.adosc(df, [30]).result_periods[30]
# adosc_trace = go.Scatter(x=adosc.index, y=adosc.line, marker_color='green')

'''
MACD
'''
macd = feature_function.macd(df, [15,30]).macd_line
macd_trace = go.Scatter(x=macd.index, y=macd.line, marker_color='green')

'''
CCI
'''
cci = feature_function.cci(df, [30]).result_periods[30]
cci_trace = go.Scatter(x=cci.index, y=cci.line, marker_color='green')

'''
BOLLINGER BANDS
'''
# bb = feature_function.bollinger_band(df, [20], 2).result_periods[20]
# cci_trace1 = go.Scatter(x=bb.index, y=bb.upper, marker_color='black')
# cci_trace2 = go.Scatter(x=bb.index, y=bb.mid, marker_color='red')
# cci_trace3 = go.Scatter(x=bb.index, y=bb.lower, marker_color='black')

'''
PRICE AVERAGE
'''
# price_average = feature_function.price_average(df, [20]).result_periods[20]
# price_average_trace1 = go.Scatter(x=price_average.index, y=price_average.close, marker_color='black')

'''
PRICE SLOP
'''
price_slop = feature_function.price_slop(df, [20]).result_periods[20]
price_slop_trace = go.Scatter(x=price_slop.index, y=price_slop.line, marker_color='black')






"""
Some basic calculations and graphing
"""
# ma = df.close.rolling(center=False, window=50).mean()
# ema = df.close.ewm(span=15, adjust=False).mean()
#

# trace1 = go.Scatter(x=df.index, y=ma, marker_color='green')
# trace2 = go.Scatter(x=df.index, y=ema, marker_color='black')
# trace3 = go.Bar(x=df.index, y=df.volume, marker_color='rgb(225, 158, 158)')

# data = [trace_original, macd_trace]
fig = py.subplots.make_subplots(rows=2, cols=1, shared_xaxes=True)

fig.append_trace(trace_original, 1, 1)
fig.append_trace(price_slop_trace, 2, 1)
# fig.append_trace(ma_of_mean_trace, 1, 1)
# fig.append_trace(trace2, 1, 1)
# fig.append_trace(trace3, 2, 1)

py.offline.plot(fig, filename='graph.html')



