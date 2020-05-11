import warnings

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.optimize import OptimizeWarning
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class Holder:
    1

# Heiken Ashi
def heike_nashi(prices, periods):
    """
    Heiken Ashi candle graph
    :param price:
    :param periods:
    :return:
    """
    result = Holder
    dict = {}

    ha_close = prices[['open', 'high', 'low', 'close']].sum(axis=1) / 4
    ha_open = ha_close.copy()
    ha_high = ha_close.copy()
    ha_low = ha_close.copy()

    for i in range(1, len(prices)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
        ha_high.iloc[i] = np.array([prices.high.iloc[i], ha_open.iloc[i], ha_close.iloc[i]]).max()
        ha_low.iloc[i] = np.array([prices.low.iloc[i], ha_open.iloc[i], ha_close.iloc[i]]).min()

    df = pd.concat((ha_open, ha_high, ha_low, ha_close), axis=1)
    df.columns = ['open', 'high', 'low', 'close']

    df.index = df.index.droplevel(0)

    dict[periods[0]] = df
    result.ha_candles = dict

    return result


# Detrending
def detrending(values, method='defference'):
    """
    :param values:
    :param method:
    :return:
    """

    global detrended
    if method == 'defference':
        detrended = values[1:] - values[:-1].values

    elif method == 'linear':
        x = np.arange(0, len(values)).reshape(-1, 1)
        y = values.values.reshape(-1, 1)

        linear_regression_model = LinearRegression()
        linear_regression_model.fit(x, y)

        Y = linear_regression_model.predict(x)
        Y = Y.reshape((len(values)))

        detrended = values - Y
    else:
        print("Error method not found")

    return detrended


def fourier_series_function(x, a0, a1, b1, w):
    """
    :param x:
    :param a0:
    :param a1:
    :param b1:
    :param w:
    :return:
    """

    f = a0 + a1*np.cos(w*x) + b1*np.sin(w*x)
    return f


def sin_series_function(x, a0, b1, w):
    """
    :param x:
    :param a0:
    :param b1:
    :param w:
    :return:
    """
    f = a0 + b1 + np.sin(w*x)
    return f


def fourier(values, periods, detrending_method='defference'):

    """
    Calculate Fourier series level one fitting coefficients
    :param values:
    :param periods:
    :param detrending_method:
    :return:
    """
    results = Holder()
    dict = {}
    plot = False
    detrended_values = detrending(values, detrending_method)


    for i in range(0, len(periods)):

        coefficients = []

        for j in range(periods[i], len(values) - periods[i]):
            x = np.arange(0, periods[i])
            y = detrended_values.iloc[j-periods[i]:j]
            res = []

            with warnings.catch_warnings():
                warnings.simplefilter('error', OptimizeWarning)

                try:
                    res = scipy.optimize.curve_fit(fourier_series_function, x, y)[0]

                except (RuntimeError, OptimizeWarning):
                    res = np.empty((4))
                    res[0:] = np.NAN
            if plot:
                xt = np.linspace(0, periods[i], 100)
                yt = fourier_series_function(xt, res[0], res[1], res[2], res[3])
                plt.plot(x, y)
                plt.plot(xt, yt)
                plt.show()

            coefficients = np.append(coefficients, res, axis=0)

        warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

        coefficients = np.array(coefficients)
        coefficients = coefficients.reshape((int(len(coefficients)/4), 4))
        df = pd.DataFrame(coefficients, index=values.iloc[periods[i]:-periods[i]].index)
        df.columns = ['a0', 'a1', 'b1', 'w']

        df = df.fillna(method='bfill')
        dict[periods[i]] = df

    results.coeffs = dict
    return results


def sin(values, periods, detrending_method='defference'):

    """
    Calculate sin fitting coefficients
    :param values:
    :param periods:
    :param detrending_method:
    :return:
    """

    results = Holder()
    dict = {}
    plot = False
    detrended_values = detrending(values, detrending_method)

    for i in range(0, len(periods)):

        coefficients = []

        for j in range(periods[i], len(values) - periods[i]):
            x = np.arange(0, periods[i])
            y = detrended_values.iloc[j-periods[i]:j]
            res = []

            with warnings.catch_warnings():
                warnings.simplefilter('error', OptimizeWarning)

                try:
                    res = scipy.optimize.curve_fit(sin_series_function, x, y)[0]

                except (RuntimeError, OptimizeWarning):
                    res = np.empty((3))
                    res[0:] = np.NAN
            if plot:
                xt = np.linspace(0, periods[i], 100)
                yt = sin_series_function(xt, res[0], res[1], res[2])
                plt.plot(x, y)
                plt.plot(xt, yt)
                plt.show()

            coefficients = np.append(coefficients, res, axis=0)

        warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

        coefficients = np.array(coefficients)
        coefficients = coefficients.reshape((int(len(coefficients)/3), 3))
        df = pd.DataFrame(coefficients, index=values.iloc[periods[i]:-periods[i]].index)
        df.columns = ['a0', 'b1', 'w']

        df = df.fillna(method='bfill')
        dict[periods[i]] = df

    results.coeffs = dict
    return results


def wadl(prices, periods):

    """
    Calculate william accumulation distribution line
    :param prices:  sending all the prices as this function requires all high low and volumes
    :param periods: [5,12,15]
    :return:
    """

    results = Holder()
    dict = {}

    for i in range(0, len(periods)):
        WAD = []
        for j in range(periods[i], len(prices)-periods[i]):

            TRH = np.array([prices.high.iloc[j], prices.close.iloc[j-1]]).max()
            TRL = np.array([prices.low.iloc[j], prices.close.iloc[j-1]]).min()

            PM = np.NAN
            if prices.close.iloc[j] > prices.close.iloc[j-1]:
                PM = prices.close.iloc[j] - TRL
            elif prices.close.iloc[j] < prices.close.iloc[j-1]:
                PM = prices.close.iloc[j] - TRH
            elif prices.close.iloc[j] == prices.close.iloc[j-1]:
                PM = 0
            else:
                prices('WADL error cannot compare CC and LC')

            AD = PM * prices.volume.iloc[j]
            WAD = np.append(WAD, AD)

        # cumsum - Add current element with previous element
        WAD = WAD.cumsum()
        WAD = pd.DataFrame(WAD, index=prices.iloc[periods[i]:-periods[i]].index)
        WAD.columns = ['line']

        dict[periods[i]] = WAD

    results.wadl = dict
    return results


def OHLC_resample(data_frame, time_frame, column='ask'):
    grouped = data_frame.groupby('Symbol')
    if np.any(data_frame.columns == 'Ask'):

        if column == 'ask':
            ask = grouped['Ask'].resample(time_frame).ohlc()
            ask_vol = grouped['AskVol'].resample(time_frame).count()

            resampled = pd.DataFrame(ask)
            resampled['AskVol'] = ask_vol

        elif column == 'bid':
            bid = grouped['Bid'].resample(time_frame).ohlc()
            bid_vol = grouped['BidVol'].resample(time_frame).count()

            resampled = pd.DataFrame(bid)
            resampled['BidVol'] = bid_vol


        else:
            raise ValueError('Column must be a string. Either ask or bid')

    elif np.any(data_frame.columns == 'close'):
        open = grouped['open'].resample(time_frame).ohlc()
        close = grouped['close'].resample(time_frame).ohlc()
        high = grouped['high'].resample(time_frame).ohlc()
        low = grouped['low'].resample(time_frame).ohlc()
        volume = grouped['volume'].resample(time_frame).count()

        resampled = pd.DataFrame(open)
        resampled['close'] = close
        resampled['high'] = high
        resampled['low'] = low
        resampled['volume'] = volume

    resampled = resampled.dropna()
    return resampled


def momentum(prices, periods):
    """
    This uses the prices frame as this uses open and close
    but we could use only ma of mean
    :param prices:
    :param periods:
    :return:
    """

    result = Holder()
    open = {}
    close = {}

    for i in range(0, len(periods)):

        open[periods[i]] = pd.DataFrame(prices.open.iloc[periods[i]:] - prices.open.iloc[:-periods[i]].values, index=prices.iloc[periods[i]:].index)
        close[periods[i]] = pd.DataFrame(prices.close.iloc[periods[i]:] - prices.open.iloc[:-periods[i]].values, index=prices.iloc[periods[i]:].index)

        open[periods[i]].columns = ['open']
        close[periods[i]].columns = ['close']

    result.open = open
    result.close = close
    return result


def stochastic(prices, periods):

    '''
    Stochastic
    :param prices:
    :param periods:
    :return:
    '''

    result = Holder()
    result_periods = {}

    for i in range(0, len(periods)):
        Ks = []
        for j in range(periods[i], len(prices)-periods[i]):
            C = prices.close.iloc[j+1]
            H = prices.high.iloc[j-periods[i]:j].max()
            L = prices.low.iloc[j-periods[i]:j].min()

            if H==L:
                K = 0
            else:
                K = 100*(C - L)/(H - L)

            Ks = np.append(Ks, K)

        df = pd.DataFrame(Ks, index=prices.iloc[periods[i]+1:-periods[i]+1].index)
        df.columns = ['K']
        #rolling(3) 3 could be window but need to be checked
        df['D'] = df.K.rolling(3).mean()
        df = df.dropna()

        result_periods[periods[i]] = df

    result.result_periods = result_periods
    return result


def williams_R(prices, periods):

    result = Holder()
    result_periods = {}

    for i in range(0, len(periods)):
        williams_R_line = []
        for j in range(periods[i], len(prices)-periods[i]):

            C = prices.close.iloc[j+1]
            H = prices.high.iloc[j-periods[i]:j].max()
            L = prices.low.iloc[j-periods[i]:j].min()

            if H==L:
                R = 0
            else:
                R = 100*(H-C)/(H-L)

            williams_R_line = np.append(williams_R_line, R)

        df = pd.DataFrame(williams_R_line, index=prices.iloc[periods[i]+1:-periods[i]+1].index)

        df.columns = ['line']
        df.dropna()
        result_periods[periods[i]] = df

    result.result_periods = result_periods
    return result


def proc(prices, periods):

    '''
    PROC function (price rate of change)
    :param prices:
    :param periods:
    :return:
    '''

    result = Holder()
    result_periods = {}

    for i in range(0, len(periods)):

        result_periods[periods[i]] = pd.DataFrame((prices.close.iloc[periods[i]:] - prices.close.iloc[:-periods[i]].values) / prices.close.iloc[:-periods[i]].values)
        result_periods[periods[i]].columns = ['line']

    result.result_periods = result_periods
    return result


def adosc(prices, periods):

    """
    Accumulation distribution oscillator
    :param prices:
    :param periods:
    :return:
    """

    result = Holder()
    result_periods = {}

    for i in range(0, len(periods)):
        adosc_line = []

        for j in range(periods[i], len(prices)-periods[i]):

            C = prices.close.iloc[j+1]
            H = prices.high.iloc[j-periods[i]:j].max()
            L = prices.low.iloc[j-periods[i]:j].min()
            V = prices.volume.iloc[j+1]

            if H==L:
                accumulation_point = 0
            else:
                accumulation_point = ((C-L) - (H-C))/(H-L)

            adosc_line = np.append(adosc_line, accumulation_point)


        adosc_line = adosc_line.cumsum()
        df = pd.DataFrame(adosc_line, index=prices.iloc[periods[i]+1:-periods[i]+1].index)
        df.columns = ['line']
        result_periods[periods[i]] = df

    result.result_periods = result_periods
    return result


def macd(prices, macd_periods):

    """
    MACD - moving average convergence distribution
    :param prices:
    :param macd_periods:
    :return:
    """

    result = Holder()

    EMA1 = prices.close.ewm(span=macd_periods[0]).mean()
    EMA2 = prices.close.ewm(span=macd_periods[1]).mean()

    macd_line = pd.DataFrame(EMA1 - EMA2)
    macd_line.columns = ['line']

    macd_signal_line = macd_line.rolling(3).mean()
    macd_signal_line.columns = ['signal_line']

    result.macd_line = macd_line
    result.macd_signal_line = macd_signal_line

    return result


def cci(prices, periods):

    """
    Commodity channel index
    :param prices:
    :param periods:
    :return:
    """
    results = Holder()
    result_periods = {}

    for i in range(0, len(periods)):

        ma = prices.close.rolling(periods[i]).mean()
        std = prices.close.rolling(periods[i]).std()

        daviation = (prices.close - ma)/std

        result_periods[periods[i]] = pd.DataFrame((prices.close-ma)/(0.015*daviation))
        result_periods[periods[i]].columns = ['line']

    results.result_periods = result_periods
    return results


def bollinger_band(prices, periods, daviation):

    result = Holder()
    result_periods = {}

    for i in range(0, len(periods)):

        mid = prices.close.rolling(periods[i]).mean()
        std = prices.close.rolling(periods[i]).std()

        upper = mid+daviation*std
        lower = mid-daviation*std

        df = pd.concat((upper, mid, lower), axis=1)
        df.columns = ['upper', 'mid', 'lower']
        result_periods[periods[i]] = df

    result.result_periods = result_periods
    return result


def price_average(prices, periods):
    """
    Moving averages of each column open high low close
    :param prices:
    :param periods:
    :return:
    """
    result = Holder()
    result_periods = {}

    for i in range(0, len(periods)):
        result_periods[periods[i]] = pd.DataFrame(prices[['open', 'high', 'low', 'close']].rolling(periods[i]).mean())

    result.result_periods = result_periods
    return result


def price_slop(prices, periods):

    """
    We are using close value for this as it's been mentioned in research paper
    Find the slop by the liner regression
    the slop is the m of y = mx+c
    :param prices:
    :param periods:
    :return:
    """

    results = Holder()
    result_periods = {}

    for i in range(0, len(periods)):
        line = []

        for j in range(periods[i], len(prices)-periods[i]):
            y = prices.high.iloc[j-periods[i]:j].values
            x = np.arange(0, len(y))

            liner_function = stats.linregress(x, y=y)

            slope = liner_function.slope
            line = np.append(line, slope)

        line = pd.DataFrame(line, index=prices.iloc[periods[i]:-periods[i]].index)
        line.columns = ['line']
        result_periods[periods[i]] = line

    results.result_periods = result_periods
    return results














































































