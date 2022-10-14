import time
import datetime
from ta import add_all_ta_features
from ta.trend import MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, AroonIndicator, IchimokuIndicator
from ta.volume import AccDistIndexIndicator
from ta.volatility import AverageTrueRange
from ta.volatility import BollingerBands
from main_test import TestIconomi, MY_STRATEGY
import pandas as pd
from collections.abc import Iterable
from logger import logger, root
from multiprocessing import Pool
from matplotlib import pyplot as plt
import numpy as np
import gc
import os
from scipy.signal import argrelextrema
from automl import select_x_y_for_training, get_model_accuracy
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import re
from pandas import DataFrame
import matplotlib.pyplot as pyplot

DECIMAL_PLACES = 2
MINIMUM_PERCENTAGE_STRUCTURE = 0.05#10**(-DECIMAL_PLACES)
QUERY_RATE_LIMIT = 500 #NUMBER OF QUERY ALLOWED PER MINUTE
QUANTILE_PERCENTAGE = 0.9
SYNCH_INTERVAL = 60*60*24*14
test_iconomi = TestIconomi()
INTERVAL = 28*60*24
BINANCE_INTERVAL = 1000
DELTA_INTERVAL = INTERVAL//2

list_of_asset_tickers = ['BTC']
plt.switch_backend('agg')
pd.options.mode.chained_assignment = None  # default='warn'


def convert_columns_to_numeric(fun_df):
    fun_df = fun_df.dropna(axis=0, how='all')
    fun_df = fun_df.apply(pd.to_numeric, errors='ignore')
    return fun_df


def select_base_columns(fun_df):
    fun_df = fun_df[['timestamp', 'BTC_open', 'BTC_high', 'BTC_low', 'BTC_close', 'BTC_volume']]
    return fun_df


def fill_prices_from_candles(fun_df, ticker='BTC'):
    fun_df[ticker] = fun_df[['BTC_open', 'BTC_close']].mean(axis=1)
    return fun_df


def divide_by_mean_volume(fun_df):
    fun_df['BTC_volume'] = fun_df['BTC_volume'] / fun_df['BTC_volume'].mean()
    return fun_df


def flush_memory_variables(variables):
    del(variables)
    gc.collect()


def parallelize_queries(function, queries, mode='imap'):
    """
    This function parallelize the execution of a function: <function> as input
    and an iterable: <queries>. An optional parameter is taken: <QUERY_RATE_LIMIT>:INTEGER which represents the
    maximum number of request per minute
    :param function:
    :param queries:
    :param QUERY_LATE_LIMIT: A number of requests per minute allowed
    :return: res: list --> The iterable of results of the parallel execution of the function defined in the input
    """

    pool = Pool(os.cpu_count()//2)
    if mode == 'imap':
        results = pool.imap(function, queries, chunksize=10)

    elif mode == 'star':
        results = pool.starmap(function, queries, chunksize=10)

    else:
        results = pool.map(function, queries, chunksize=10)

    pool.close()
    pool.join()
    pool.terminate()
    flush_memory_variables([pool])

    return results


def reattempt_asset_performance_retrieval(api_url, function):
    tick = api_url.split("assets/")[-1].split("/")[0]
    start_time = time.time()
    prices_response = 'api.iconomi.com'

    while (((prices_response is None) or ('api.iconomi.com' in prices_response)) and ((time.time()-start_time)<7)):
        time.sleep(2)
        print(f"ST ASSETS - Reattempting to retrieve information for strategy {tick}")
        prices_response = function(ticker=tick)

    if 'api.iconomi.com' in prices_response or prices_response is None:
        print(f"ST ASSETS - WARNING: The information for strategy {tick} could not be retrieved !!!")
        logger.warn(f"ST ASSETS - WARNING: The information for strategy {tick} could not be retrieved !!!")

    return prices_response

def return_variation_of_strategy_for_stop_loss_rule(ticker=MY_STRATEGY):

    prices_response = test_iconomi.test_get_strategy_prices(ticker=ticker)
    price = 0
    if prices_response is None:
        print(f"ST ASSETS - WARNING: Reattempting to retrieve strategy {ticker} structure")
        prices_response = test_iconomi.test_get_strategy_prices(ticker=ticker)
    if 'api.iconomi.com' in str(prices_response).lower():
        prices_response = reattempt_asset_performance_retrieval(prices_response, test_iconomi.test_get_strategy_prices)

    if (prices_response is not None) and ('api.iconomi.com' not in prices_response):
        price = prices_response['changeAll'] if 'changeAll' in prices_response.keys() else 0

    else:
        print(f"ST ASSETS - WARNING: Could not retrieve time-based percentage yields of {ticker}")

    return price

def return_reputation_yield_of_strategy(response_all_strategies_tickers):
    """(prices_response is None)
    This function retrieves the performance of a strategy
    :param response_all_strategies_tickers: an iterable of or a ticker string
    :return: all_responses: dict --> the performance of the strategy 'ticker'
    """
    all_responses = {}

    if not isinstance(response_all_strategies_tickers, Iterable) or type(response_all_strategies_tickers) == str:
        response_all_strategies_tickers = [response_all_strategies_tickers]

    try:
        for i, ticker in enumerate(response_all_strategies_tickers):

            if ticker =='SCND':
                all_responses[ticker] = -999
                continue
            prices_response = test_iconomi.test_get_asset_prices(ticker=ticker)
            if prices_response is None:
                print(f"ST ASSETS - WARNING: Reattempting to retrieve strategy {ticker} structure")
                prices_response = test_iconomi.test_get_asset_prices(ticker=ticker)
            if 'api.iconomi.com' in str(prices_response).lower():
                prices_response = reattempt_asset_performance_retrieval(prices_response, test_iconomi.test_get_asset_prices)

            if (prices_response is not None) and ('api.iconomi.com' not in prices_response):
                change24h = prices_response['change24h'] if 'change24h' in prices_response.keys() else 0
                change7d = prices_response['change7d'] if 'change7d' in prices_response.keys() else 0
                change1m = prices_response['change1m'] if 'change1m' in prices_response.keys() else 0
                change3m = prices_response['change3m'] if 'change3m' in prices_response.keys() else 0
                change6m = prices_response['change6m'] if 'change6m' in prices_response.keys() else 0
                change12m = prices_response['change12m'] if 'change12m' in prices_response.keys() else 0
                changeAll = prices_response['changeAll'] if 'changeAll' in prices_response.keys() else 0
                price = prices_response['price'] if 'price' in prices_response.keys() else 0
                aum = prices_response['aum'] if 'aum' in prices_response.keys() else 0

                # reputation_score = changeAll
                reputation_score = price
                all_responses[ticker] = reputation_score

            else:
                print(f"ST ASSETS - WARNING: Could not retrieve time-based percentage yields of {ticker}")

    except Exception as err:
        print(err)

    return all_responses


def get_price_change_of_strategy(previous_timestamp, ticker=MY_STRATEGY):
    """
    This function retrieve the price change of a strategy
    :param previous_timestamp: float
    :param ticker: string
    :return:
    """
    ret = test_iconomi.get_price_history(previous_timestamp=previous_timestamp, ticker=ticker)
    print(ret)


def get_last_rebalance_date_from_strategies(ticker, field='lastRebalanced'):
    """
    This function return the last rebalancedDate of the strategy 'ticker'. Optional field can be specified as long as
    matching the ICONOMI API body response
    :param ticker: string
    :param field: string
    :return: last_rebalance_date: float --> Date of last rebalance for strategy 'ticker'
    """
    res = test_iconomi.test_get_structure(ticker)
    if res is not None:
        if field in res:
            last_rebalance_date = res[field]
            return last_rebalance_date
        return 0
    return 0


def collect_indicators(indicators):
    response = {}
    for ticker in indicators:
        for indicator_type in indicators[ticker].keys():
            indicator_type_formatted = indicator_type.split('_', 3)[-1].split('_')[0]
            for value_key in indicators[ticker][indicator_type]:
                response[f"{ticker}_{indicator_type_formatted}_{value_key}"] = indicators[ticker][indicator_type][value_key]
    return response


def generate_stats_model(x, y, prediction):
    # x['BTC'] = x[['BTC_open', 'BTC_close']].mean(axis=1)
    # x['BTC_predicted_index'] = prediction
    # x = pd.concat([x,y], axis=1)
    #
    # col = 'BTC'
    #
    # x['BTC_index'] = x['BTC_index'].astype(int)
    # x['BTC_predicted_index'] = x['BTC_predicted_index'].astype(int)
    # x['BTC_predicted_index_min'] = x['BTC'][x['BTC_predicted_index']==1]
    # x['BTC_predicted_index_max'] = x['BTC'][x['BTC_predicted_index']==-1]
    # x.reset_index(drop=True, inplace=True)
    # x['BTC_index_min'] = x['BTC'][x['BTC_predicted_index']==1]
    # x['BTC_index_max'] = x['BTC'][x['BTC_predicted_index']==-1]
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.scatter([i for i in range(len(x))], x['BTC'], c='c')
    # ax1.scatter([i for i in range(len(x))], x['BTC_index_min'], c='y')
    # ax1.scatter([i for i in range(len(x))], x['BTC_index_max'], c='g')
    # ax1.scatter([i for i in range(len(x))], x['BTC_predicted_index_min'], c='r')
    # ax1.scatter([i for i in range(len(x))], x['BTC_predicted_index_max'], c='m')
    # plt.savefig(HYSTORICAL_DATA_ROOT+"plots/ml/"+f"plot_{datetime.datetime.strftime(datetime.datetime.today() , '%d_%m_%Y_%H_%M_%S')}.png")
    # plt.close(fig)
    return


def return_generator_from_df(df, step):
    for i in range(len(df)//step):
        yield df.iloc[i*step:(i+1)*step,:]
    if len(df)%step != 0:
        yield df.iloc[i*step::,:]

def iterate_zip_dfs(df, step=14*60*24):
    columns = df.columns
    grp = df.groupby('BTC_index')

    if len(grp) != 3:
        my_iter = min(len(grp.get_group(-1).values), len(grp.get_group(1).values))
    else:
        my_iter = min(len(grp.get_group(-1).values), len(grp.get_group(1).values), len(grp.get_group(0).values))


    step = step // 3 if len(grp) ==3 else step//2
    for i in range(my_iter//step):
        df_max = grp.get_group(-1).iloc[i*step:(i+1)*step,:]
        df_min = grp.get_group(1).iloc[i*step:(i+1)*step,:]
        if len(grp)==3:
            df_pass = grp.get_group(0).iloc[i*step:(i+1)*step,:]
            df = pd.concat([df_max, df_min, df_pass], axis=0, ignore_index=True).sample(frac=1)
        else:
            df = pd.concat([df_max, df_min], axis=0, ignore_index=True).sample(frac=1)
        df.columns = columns
        if len(df)>0:
            yield df


def shift_column_compute_old(mydf, custom_interval=None):

    # mydf = mydf[mydf.select_dtypes(include='object').apply(lambda x: x.str.contains('429')==False).all(axis=1)]
    # mydf = mydf[mydf.select_dtypes(include='string').apply(lambda x: x.str.contains('429')==False).all(axis=1)]

    if type(custom_interval)==list:
        col = 'BTC'
        mydf[f'{col}_index_sum'] = 0
        for i, interval in enumerate(custom_interval):
            mydf.reset_index(drop=True, inplace=True)
            mydf[f'{col}_min_{interval}'] = mydf[col][
                (mydf[col].shift(interval) > mydf[col]) & (mydf[col].shift(-interval) > mydf[col])]
            mydf.reset_index(drop=True, inplace=True)
            mydf[f'{col}_max_{interval}'] = mydf[col][
                (mydf[col].shift(interval) < mydf[col]) & (mydf[col].shift(-interval) < mydf[col])]
            mydf.reset_index(drop=True, inplace=True)

            conditions = [
                mydf[f'{col}_max_{interval}'].isna() & mydf[f'{col}_min_{interval}'].isna(),
                mydf[f'{col}_min_{interval}'].notna(),
                mydf[f'{col}_max_{interval}'].notna()
            ]
            choices = [0, 1, -1]
            mydf[f'{col}_index_{interval}'] = np.select(conditions, choices, default=0)
            mydf[f'{col}_index_sum'] = mydf[f'{col}_index_sum'] + mydf[f'{col}_index_{interval}']

    else:
        col = 'BTC'
        interval = DELTA_INTERVAL if len(mydf)/DELTA_INTERVAL >4 else BINANCE_INTERVAL
        interval = interval if len(mydf)/interval >4 else round(len(mydf)/4)
        if custom_interval:
           interval = round(custom_interval)
        mydf.reset_index(drop=True, inplace=True)
        mydf[f'{col}_min'] = mydf[col][(mydf[col].shift(interval) > mydf[col]) & (mydf[col].shift(-interval) > mydf[col])]
        mydf.reset_index(drop=True, inplace=True)
        mydf[f'{col}_max'] = mydf[col][(mydf[col].shift(interval) < mydf[col]) & (mydf[col].shift(-interval) < mydf[col])]
        mydf.reset_index(drop=True, inplace=True)

        conditions = [
            mydf[f'{col}_max'].isna() & mydf[f'{col}_min'].isna(),
            mydf[f'{col}_min'].notna(),
            mydf[f'{col}_max'].notna()
        ]
        choices = [0, 1, -1]
        mydf[f'{col}_index'] = np.select(conditions, choices, default=0)
    # generate_stats_index(mydf)
    # mydf = mydf.drop([f'{col}_min', f'{col}_max'], axis=1)

    return mydf

def retrieve_new_estimator(chunk, current_pipeline):
    chunk = convert_columns_to_numeric(chunk)
    # chunk = divide_by_mean_volume(chunk)
    chunk = fill_prices_from_candles(chunk)
    chunk = shift_column_compute(chunk, 14*60*24)
    chunk = ta_batch_make(chunk, day_conversion_factor=60*24 if 60*24*52>len(chunk) else 1)
    chunk = chunk[(chunk['BTC_index'] == 1) | (chunk['BTC_index'] == -1)]
    chunk = select_custom_training_columns(chunk)
    X, y = select_x_y_for_training(chunk)
    if len(set(y)) != 2:
        print("NOT ALL LABELS FOUND IN CHUNK")
        return current_pipeline
    print("PREVIOUS:", get_model_accuracy(X, y, current_pipeline))
    if 'SGD' in str(current_pipeline.named_steps.estimator.__class__()):
        current_pipeline.partial_fit(X, y, classes=np.unique(y))
    else:
        current_pipeline.fit(X, y)

    # export_model(current_pipeline, MODEL_FILE_PATH)

    print("CURRENT:", get_model_accuracy(X, y, current_pipeline))
    return current_pipeline


def select_custom_training_columns(df, ticker='BTC'):
    column_names = [f"{ticker}",f"{ticker}_index","macd_3","macd_14","macd_28","osc_3","osc_14","osc_28","adx_3","adx_14","adx_28","accdist","avtr_3","avtr_14","avtr_28","aroon_3","ich_3","ich_14","ich_28","rsi_14","rsi_3","rsi_28"]
    df = df[pd.Index(column_names)]
    return df


def ta_batch_make(candle, day_conversion_factor=60*24, subset=True, ticker='BTC'):
    if not subset:
        if f'{ticker}_index' not in candle.columns:
            candle = candle[[f'{ticker}_open', f'{ticker}_high', f'{ticker}_low', f'{ticker}_close', f'{ticker}_volume', f'{ticker}', f'{ticker}_index_sum']]
        else:
            candle = candle[[f'{ticker}_open', f'{ticker}_high', f'{ticker}_low', f'{ticker}_close', f'{ticker}_volume', f'{ticker}', f'{ticker}_index']]
        print('GENERATING TA FOR N.RECORDS: ', len(candle))
        candle = add_all_ta_features(candle, open=f"{ticker}_open", high=f"{ticker}_high", low=f"{ticker}_low", close=f"{ticker}_close",
                                 volume=f"{ticker}_volume", fillna=True, conversion_interval=day_conversion_factor)

    if subset:
        bb_3 = BollingerBands(candle[f'{ticker}_close'], window=3*day_conversion_factor, window_dev=1*day_conversion_factor)
        bb_14 = BollingerBands(candle[f'{ticker}_close'], window=14*day_conversion_factor, window_dev=3*day_conversion_factor)
        bb_28 = BollingerBands(candle[f'{ticker}_close'], window=28*day_conversion_factor, window_dev=6*day_conversion_factor)

        avtr_14 = AverageTrueRange(candle[f'{ticker}_high'], candle[f'{ticker}_low'], candle[f'{ticker}_close'], window=14*day_conversion_factor)
        avtr_3 = AverageTrueRange(candle[f'{ticker}_high'], candle[f'{ticker}_low'], candle[f'{ticker}_close'], window=3*day_conversion_factor)
        avtr_28 = AverageTrueRange(candle[f'{ticker}_high'], candle[f'{ticker}_low'], candle[f'{ticker}_close'], window=28*day_conversion_factor)

        accdist = AccDistIndexIndicator(candle[f'{ticker}_high'], candle[f'{ticker}_low'], candle[f'{ticker}_close'], candle[f'{ticker}_volume'])

        adx_3 = ADXIndicator(candle[f'{ticker}_high'], candle[f'{ticker}_low'], candle[f'{ticker}_close'], window=3*day_conversion_factor)
        adx_14 = ADXIndicator(candle[f'{ticker}_high'], candle[f'{ticker}_low'], candle[f'{ticker}_close'], window=14*day_conversion_factor)
        adx_28 = ADXIndicator(candle[f'{ticker}_high'], candle[f'{ticker}_low'], candle[f'{ticker}_close'], window=28*day_conversion_factor)

        aroon_3 = AroonIndicator(candle[f'{ticker}_close'], window=3*day_conversion_factor)
        aroon_14 = AroonIndicator(candle[f'{ticker}_close'], window=14*day_conversion_factor)
        aroon_28 = AroonIndicator(candle[f'{ticker}_close'], window=28*day_conversion_factor)

        ich_3 = IchimokuIndicator(candle[f'{ticker}_high'], candle[f'{ticker}_low'], window1=1, window2=3, window3=14)
        ich_14 = IchimokuIndicator(candle[f'{ticker}_high'], candle[f'{ticker}_low'], window1=3, window2=14, window3=28)
        ich_28 = IchimokuIndicator(candle[f'{ticker}_high'], candle[f'{ticker}_low'], window1=14*day_conversion_factor, window2=28, window3=56)

        macd_3 = MACD(candle[f'{ticker}_close'], window_slow=14*day_conversion_factor, window_fast=3, window_sign=3*day_conversion_factor)
        macd_14 = MACD(candle[f'{ticker}_close'], window_slow=28*day_conversion_factor, window_fast=14, window_sign=3*day_conversion_factor)
        macd_28 = MACD(candle[f'{ticker}_close'], window_slow=56*day_conversion_factor, window_fast=28, window_sign=14*day_conversion_factor)

        osc_3 = StochasticOscillator(candle[f'{ticker}_high'], candle[f'{ticker}_low'], candle[f'{ticker}_close'], window=3*day_conversion_factor, smooth_window=1*day_conversion_factor)
        osc_14 = StochasticOscillator(candle[f'{ticker}_high'], candle[f'{ticker}_low'], candle[f'{ticker}_close'], window=14*day_conversion_factor, smooth_window=3*day_conversion_factor)
        osc_28 = StochasticOscillator(candle[f'{ticker}_high'], candle[f'{ticker}_low'], candle[f'{ticker}_close'], window=28*day_conversion_factor, smooth_window=3*day_conversion_factor)

        rsi_3 = RSIIndicator(candle[f'{ticker}_close'], window=3*day_conversion_factor)
        rsi_14 = RSIIndicator(candle[f'{ticker}_close'], window=14*day_conversion_factor)
        rsi_28 = RSIIndicator(candle[f'{ticker}_close'], window=28*day_conversion_factor)


        candle['macd_3'] = macd_3.macd()
        candle['macd_14'] = macd_14.macd()
        candle['macd_28'] = macd_28.macd()

        candle['osc_3'] = osc_3.stoch()
        candle['osc_14'] = osc_14.stoch()
        candle['osc_28'] = osc_28.stoch()

        candle['adx_3'] = adx_3.adx()
        candle['adx_14'] = adx_14.adx()
        candle['adx_28'] = adx_28.adx()

        candle['accdist'] = accdist.acc_dist_index()

        candle['avtr_3'] = avtr_3.average_true_range()
        candle['avtr_14'] = avtr_14.average_true_range()
        candle['avtr_28'] = avtr_28.average_true_range()

        candle['aroon_3'] = aroon_3.aroon_indicator()
        candle['aroon_14'] = aroon_14.aroon_indicator()
        candle['aroon_28'] = aroon_28.aroon_indicator()

        candle['ich_3'] = ich_3.ichimoku_base_line()
        candle['ich_14'] = ich_14.ichimoku_base_line()
        candle['ich_28'] = ich_28.ichimoku_base_line()

        candle['rsi_14'] = rsi_14.rsi()
        candle['rsi_3'] = rsi_3.rsi()
        candle['rsi_28'] = rsi_28.rsi()

        candle['bb_3'] = bb_3.bollinger_mavg()
        # candle['bb_3_low'] = bb_3.bollinger_lband_indicator()
        # candle['bb_14_high'] = bb_14.bollinger_hband_indicator()
        # candle['bb_14_low'] = bb_14.bollinger_lband_indicator()
        # candle['bb_28_high'] =bb_28.bollinger_hband_indicator()
        # candle['bb_28_low'] =bb_28.bollinger_lband_indicator()


        print('GENERATED TA FOR N.RECORDS: ', len(candle))
    return candle


def retrieve_btc_historical_data(granulation=None, fromv=None, to=None):
    klines = test_iconomi.get_price_history('BTC', granulation, fromv, to)
    return klines


def shift_column_compute(mydf, custom_interval=None, ticker='BTC'):

    # mydf = mydf[mydf.select_dtypes(include='object').apply(lambda x: x.str.contains('429')==False).all(axis=1)]
    # mydf = mydf[mydf.select_dtypes(include='string').apply(lambda x: x.str.contains('429')==False).all(axis=1)]
    interval = DELTA_INTERVAL if len(mydf)/DELTA_INTERVAL >4 else BINANCE_INTERVAL
    interval = interval if len(mydf)/interval >4 else round(len(mydf)/4)
    if custom_interval:
       interval = round(custom_interval)

    mydf[f'{ticker}_min'] = mydf.iloc[argrelextrema(mydf[f'{ticker}'].values, np.less_equal,
                                      order=interval)[0]][f'{ticker}']
    mydf[f'{ticker}_max'] = mydf.iloc[argrelextrema(mydf[f'{ticker}'].values, np.greater_equal,
                                      order=interval)[0]][f'{ticker}']

    conditions = [
        mydf[f'{ticker}_max'].isna() & mydf[f'{ticker}_min'].isna(),
        mydf[f'{ticker}_min'].notna(),
        mydf[f'{ticker}_max'].notna()
    ]
    choices = [0, 1, -1]
    mydf[f'{ticker}_index'] = np.select(conditions, choices, default=0)
    # generate_stats_index(mydf)

    return mydf


# def standardise(fun_df):
#     """
#     This function standardise a list of tuple. It is used to standardise the reputation score of each strategy before
#     weighting each strategy weight by that score.
#     :param list_of_tuple: list(tuple(3:)) --> a list of tuple with 3 elements at least
#     :return: df: Pandas Dataframe --> The list of tuple with the standardised reputation column![](file_folder/strategies_weight.png)
#     """
#     maximum = max(fun_df[[column for column in fun_df.columns if ('delta' in column and "std" not in column)]].max(axis=1))
#     minimum = min(fun_df[[column for column in fun_df.columns if ('delta' in column and "std" not in column)]].min(axis=1))
#     dict_to_create = {}
#     for col in fun_df.columns:
#         if ('timestamp' not in col and 'diff' not in col) and ('delta' not in col):
#             fun_df[f"{col}_delta_std"] = (fun_df[f"{col}_delta"] - minimum) / (maximum - minimum)
#
#     return fun_df


def retrieve_structure_of_strategies(dict_of_strategies_reputation):
    """

    :param dict_of_strategies_reputation: tuple --> A tuple with (0) 'ticker' and (1) 'reputation' score of each strategy
    :return: all_value_and_weights_tuple_list: list(tuple(4:) --> The list of assets per each strategy with relative
    weight according to the reputation of the strategy and the weight it has in that strategy
    """
    all_value_and_weights_tuple_list = []
    ticker = dict_of_strategies_reputation[0]

    if ticker == 'SCND':
        return (ticker, -999, -999, -999, -999)

    else:
        strategy_structure = test_iconomi.test_get_structure(ticker=ticker)

    if strategy_structure is None:
        print(f"ST ASSETS - WARNING: Structure of strategy {ticker} could not be retrieved!")
        strategy_structure = test_iconomi.test_get_structure(ticker=ticker)

    if 'api.iconomi.com' in str(strategy_structure).lower():
        strategy_structure = reattempt_asset_performance_retrieval(strategy_structure, test_iconomi.test_get_structure)

    if type(strategy_structure) is not dict or len(strategy_structure.keys())==0:
        raise Exception(f"ST ASSETS - ERROR: {ticker} strategy's structure could not be retrieved")
    else:
        logger.info(f"ST ASSETS - Executed Request of strategies assets % for: {ticker} - {datetime.datetime.strftime(datetime.datetime.today() , '%d/%m/%Y-%H:%M')}")
        print(f"ST ASSETS - Executed Request of strategies assets % for: {ticker} - {datetime.datetime.strftime(datetime.datetime.today() , '%d/%m/%Y-%H:%M')}")


        for value in strategy_structure['values']:
            asset_name = value['assetName'] if 'assetName' in value.keys() else None
            if asset_name is not None and 'other assets' not in asset_name:
                weight = value['rebalancedWeight'] if 'rebalancedWeight' in value.keys() else None
                weight_ticker = value['assetTicker'] if 'rebalancedWeight' in value.keys() else None
                reputation = dict_of_strategies_reputation[1]
                weighting_coefficient = reputation * weight
                if weight is not None and weight_ticker is not None:
                    all_value_and_weights_tuple_list.append((ticker, weight_ticker, weight, reputation, weighting_coefficient))
                else:
                    print(f"ST ASSETS - Could not retrieve weights of asset: {asset_name} for strategy: {ticker} - {datetime.datetime.strftime(datetime.datetime.today() , '%d/%m/%Y-%H:%M')}")

    return all_value_and_weights_tuple_list


def check_percent_of_stable_in_strategy(strategy_name):

    values = retrieve_structure_of_strategies((strategy_name, 0))
    percentage_of_stable = 0
    list_of_assets = []
    for ticker, weight_ticker, weight, reputation, weighting_coefficient in values:
        if (weight_ticker != 'USDC') and (weight_ticker != 'TUSD') and (weight_ticker != 'USDT') and (weight_ticker != 'PAXG'):
            pass
        else:
            percentage_of_stable += weight
        list_of_assets.append((weight_ticker, weight))

    return percentage_of_stable, list_of_assets


# def order_list_of_assets(fun_df):
#     """
#     This function aggregates all weights of each assets across all strategies. It then takes the superior quantile and
#     filter out those assets that are below the 50% quantile. Eeventually it rescale the final list of asset to a sum of 1
#     (eg. from format convention of ICONOMI API) and return the result in a list form
#     :param datatrame_of_tuples: Pandas Dataframe --> The dataframe coming from <standardise> function.
#     :return: zipped_ticker_value_tuple: list --> List of assets and weight to write into our strategy
#     """
#
#     dict_to_create = {}
#     for col in fun_df.columns:
#         if 'timestamp' not in col and 'diff' not in col and 'delta' in col:
#             largest_nr = round(DELTA / 5) if round(DELTA / 5) > 1 else 1
#             index_mean = sum(fun_df.iloc[fun_df.nlargest(largest_nr, [f'{col}']).index, :][f'{col}'] \
#                              * (fun_df.nlargest(largest_nr, [f'{col}']).index + 1))
#             dict_to_create[col.replace("_delta","")] = index_mean
#
#     fun_df = pd.DataFrame.from_dict(dict_to_create.items())
#     Q1 = fun_df.quantile(QUANTILE_PERCENTAGE)
#     trueList = ~((fun_df < Q1))
#     weight_dataframe = fun_df[trueList].dropna()
#
#     ratio = 1 / sum(weight_dataframe[1]) #The ratio is used to bring the weight to the same order of magnitude of the strategy input body
#
#     weight_dataframe[2] = weight_dataframe[1] * ratio
#
#     zipped_ticker_value_tuple = list(zip(weight_dataframe[0].array, weight_dataframe[2].array))
#     return zipped_ticker_value_tuple


def redistribute_small_weights(zipped_ticker_value_tuple):
    """
    This function redistribute any weight of any asset that may be falling below the MINIMUM_PERCENTAGE_STRUCTURE.
    This depends on the parameter specified in the beginning of this file. If MINIMUM_PERCENTAGE_STRUCTURE is coupled to
    DECIMAL_PLACES (eg. MINIMUM_PERCENTAGE_STRUCTURE=10**(-DECIMAL_PLACES)), then this part may not be mandatory,
    otherwise becomes handy in case of any inconsistency of the weights to sum up exactly to 1 or having a format
    not compatible with ICONOMI API post body (eg. number of decimal accepted by ICONOMI)
    :param zipped_ticker_value_tuple: a list of tuple of assets' tickers with asset weights
    :return: list_with_ticker_and_weights: list --> A list of assets' ticker and weights to write in the body of the structure post request
    """
    list_with_ticker_and_weights = []
    tot = 0
    weights_to_redistribute = []
    for el in reversed(sorted(zipped_ticker_value_tuple, key=lambda x: x[1])):
        weight = round(el[1], DECIMAL_PLACES)
        if tot + weight > 1:
            if 1 - tot == 0:
                break
            if 1 - tot > MINIMUM_PERCENTAGE_STRUCTURE:
                list_with_ticker_and_weights.append((el[0], round(1 - tot, DECIMAL_PLACES)))
            else:
                weights_to_redistribute.append((el[0], round(1 - tot, DECIMAL_PLACES)))
            break
        else:
            if weight >= MINIMUM_PERCENTAGE_STRUCTURE:
                list_with_ticker_and_weights.append((el[0], weight))
                tot += weight

            elif 0 <= weight and weight < MINIMUM_PERCENTAGE_STRUCTURE:
                weights_to_redistribute.append((el[0], weight))
                tot += weight

            else:
                weights_to_redistribute.append((el[0], weight))
                tot += weight

    for i, ticker_weight in enumerate(list_with_ticker_and_weights):
        ticker, weight = ticker_weight
        for out_ticker, out_weight in weights_to_redistribute:
            list_with_ticker_and_weights[i] = (ticker, weight+out_weight)
            weights_to_redistribute.remove((out_ticker, out_weight))
            break
        if len(weights_to_redistribute) == 0:
            break

    return list_with_ticker_and_weights


def generate_structure_values_body(list_with_ticker_and_weights):
    """
    This function creates the body of the post request for structure writing.
    :param list_with_ticker_and_weights: list --> The output of <redistribute_missing_weight> function
    :return: a list of values (eg. ticker and weights) to append to the final body of the request
    """
    bodies = []
    for el in list_with_ticker_and_weights:
        body_to_add = {
            "rebalancedWeight": str(el[1]),
            "assetTicker": f"{el[0]}"
        }
        bodies.append(body_to_add)

    return bodies


def redistribute_missing_weight(tuple_of_ticker_and_weights):
    """
    This function redistribute missing weight to the sum of 1 by adding the leftover to the sum of 1 to the first asset
    weight in the <tuple_of_ticker_and_weights>
    :param tuple_of_ticker_and_weights: list --> Ticker and weights tuple list
    :return: tuple_of_ticker_and_weights: the redistributed list of assets and weights to write
    """
    for i,el in enumerate(tuple_of_ticker_and_weights):
        tuple_of_ticker_and_weights[i] = el[0], round(el[1], DECIMAL_PLACES)

    tot = round(sum([el[1] for el in tuple_of_ticker_and_weights]),DECIMAL_PLACES)
    missing_weight = round(1 - tot, DECIMAL_PLACES)
    if (missing_weight != 0 or missing_weight != 0.000) and (missing_weight != 1):

        count = len(tuple_of_ticker_and_weights)
        if missing_weight/len(tuple_of_ticker_and_weights) >= MINIMUM_PERCENTAGE_STRUCTURE:
            weight_to_redistribute = missing_weight/len(tuple_of_ticker_and_weights)

        elif missing_weight/5 >= DECIMAL_PLACES:
            count = 5
            weight_to_redistribute = missing_weight/5

        elif missing_weight/2 >= DECIMAL_PLACES:
            count = 2
            weight_to_redistribute = missing_weight/2

        else:
            count = 1
            weight_to_redistribute = missing_weight

        weight_redistributed = 0
        for i in range(len(tuple_of_ticker_and_weights)):
            tuple_of_ticker_and_weights[i] = tuple_of_ticker_and_weights[i][0], round(tuple_of_ticker_and_weights[i][1]+ weight_to_redistribute, DECIMAL_PLACES)
            weight_redistributed += weight_to_redistribute
            if i+1>=count:
                break
        if missing_weight - weight_redistributed != 0:
            tuple_of_ticker_and_weights[0] = tuple_of_ticker_and_weights[0][0], round(tuple_of_ticker_and_weights[0][1]+(missing_weight-weight_redistributed), DECIMAL_PLACES)

    return tuple_of_ticker_and_weights


def check_structure_conforms_requirements(bodies):
    """
    This function assert that the sum of all asset is 1, otherwise rise an error
    :param bodies: the list of weights and ticker
    :return:
    """
    tot = round(sum([round(float(el['rebalancedWeight']), DECIMAL_PLACES) for el in bodies]), DECIMAL_PLACES)
    try:
        assert tot == 1.00
    except AssertionError as err:
        raise err


def transform_list_of_dict_to_dict(tickers_with_reputation):
    """
    This function transform a list of dictionaries into a dictionary
    :param tickers_with_reputation: the list of tickers
    :return: dict_tickers_with_reputation: dict --> The dictionary with key: ticker and value: weight
    """
    dict_tickers_with_reputation = {}
    for el in tickers_with_reputation:
        if el is not None:
            dict_tickers_with_reputation = {**dict_tickers_with_reputation, **el}
    return dict_tickers_with_reputation


def get_statistics_all_time(ticker):
    ret = test_iconomi.get_statistics(ticker=ticker)
    return ret


def preprocess_df(df, custom_interval_for_ta=1):
    ticker = df[0]
    df = df[1]
    df = pd.DataFrame.from_records(df)
    df = select_custom_training_columns(ticker)
    # df = select_base_columns(df)
    df = convert_columns_to_numeric(df)
    df = fill_prices_from_candles(df, ticker=ticker)
    df= shift_column_compute(df, DELTA_INTERVAL//(60*24), ticker=ticker)
    df = ta_batch_make(df, day_conversion_factor=custom_interval_for_ta, ticker=ticker)
    df = df.fillna(0)
    # df = df[(df['BTC_index'] == 1) | (df['BTC_index'] == -1)]
    df.drop(['bb_3',f'{ticker}_min', f'{ticker}_max'], axis=1, inplace=True, errors='ignore')
    return {ticker: df}


# def collect_binance_data_per_ticker(ticker_list, interval='1h'):
#     mydict = {}
#     for ticker in ticker_list:
#         mydict[ticker] = {**mydict, **remove_ticker_not_in_binance([ticker], interval=interval)}
#     return mydict


# def transform_into_nparray_and_predict(df, model=m):
#     _, temp_x, _, temp_y, btc_price = prepare_keras_data(list(df.values())[0])
#     # temp_x, temp_y = select_x_y_for_training(df)
#     y_pred_full = model.predict(temp_x)
#     return temp_x[-1::], y_pred_full, temp_y, btc_price


# def plot_price_tickers(y_pred_full, temp_y, btc_price):
#     from matplotlib import pyplot
#     a = pd.concat([btc_price.reset_index(), pd.DataFrame.from_dict(temp_y),
#                    pd.DataFrame.from_records(y_pred_full)], axis=1, ignore_index=True)
#     a.plot(subplots=True)
#     pyplot.savefig(HYSTORICAL_DATA_ROOT + "plots/" + 'keras_model_variables_Assets.png')
#     pyplot.close()

def test_result_w_binance_data(m, interval, custom_interval_for_ta=None):

    df = pd.read_csv(f'{root.rsplit(os.sep, 1)[0]}/BTC_data.csv')

    try:
        if not custom_interval_for_ta:
            custom_interval_for_ta = 1
        if interval == '1h':
            interval = 60
        elif interval=='1d':
            interval = 60*24
        elif interval == '1m':
            interval = 1
        df = pd.DataFrame.from_records(df['BTC'])
        df = convert_columns_to_numeric(df)
        df = fill_prices_from_candles(df)
        df= shift_column_compute(df, DELTA_INTERVAL//interval)
        df = ta_batch_make(df, day_conversion_factor=custom_interval_for_ta)
        df = df.fillna(0)
        df.drop(['bb_3','BTC_min', 'BTC_max'], axis=1, inplace=True)
        _, temp_x, _, temp_y, btc_price = prepare_keras_data(df)
        y_pred_full = m.predict(temp_x)
        y_pred_full = pd.DataFrame.from_records(y_pred_full)

        a = pd.concat([btc_price.iloc[-999::].reset_index(drop=True),
                       pd.DataFrame.from_dict(list(temp_y)).reset_index(drop=True),
                       y_pred_full.reset_index(drop=True),
                       pd.DataFrame.from_dict(np.gradient(np.array(list(y_pred_full[0]), dtype=float), 1))]
                      , axis=1, ignore_index=True)


        a[5] = (((a[3] / a[3].shift(1)) > 0) & ((a[3].shift(1) / a[3].shift(2)) < 0)) | (
                ((a[3] / a[3].shift(1)) < 0) & ((a[3].shift(1) / a[3].shift(2)) > 0)) & ((abs(a[3] / a[3].shift(1)) > 0.01))
        a[6] = a[2]
        a[6][~a[5]] = None

        a.plot(subplots=True)
        pyplot.savefig(os.environ['HOME']+f'keras_model_variables_{interval}.png')
        pyplot.close()

        return float(a[6].tail(1))

    except Exception as err:
        print("PREPROCESSING OF BINANCE DATA FOR PREDICTION ERROR: ", err)

        return np.nan

# def test_result_w_binance_data(m, interval, custom_interval_for_ta=None, tickers=['BTC', 'CVX']):
#
#     binance_data = collect_binance_data_per_ticker(tickers, interval='1h')
#     dfs = []
#     for el in binance_data:
#         preprocess_df(el)
#         dfs.append(el)
#
#     # dfs = parallelize_queries(preprocess_df, binance_data.items(), mode='map')
#     dfs_out = parallelize_queries(transform_into_nparray_and_predict, dfs, mode='map')
#
#     parallelize_queries(plot_price_tickers, dfs_out, mode='a')
#
#     return dfs_out[0][0], dfs_out[0][3]


def prepare_keras_data(initial_data, ticker='BTC'):
    initial_data = initial_data[pd.Index(
        [col for col in initial_data.columns if 'unix' in col] + sorted([col for col in initial_data.columns if f'{ticker}_index' not in col and 'unix' not in col
                                                                         and 'volume' not in col.lower()])  +  [col for col in initial_data.columns if
                                                                      f'{ticker}_index' in col])]

    values = initial_data.values
    # integer encode direction
    encoder = LabelEncoder()
    values[:, -1] = encoder.fit_transform(values[:, -1])
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    # drop columns we don't want to predict
    reframed.drop(['var1(t)', 'var29(t)'], axis=1, inplace=True)
    nr_col = len([col for col in reframed.columns if re.match('var29\(t-', col)])
    reframed = reframed[pd.Index(
        [col for col in reframed.columns if 'var29(t-' not in col] + [col for col in reframed.columns if
                                                                      re.match('var29\(t-', col)])]
    # split into train and test sets
    values = reframed.values
    n_train_hours = 5 * INTERVAL
    train = values[2*INTERVAL:-n_train_hours, :]
    test = values[-n_train_hours:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-nr_col], train[:, -nr_col]
    test_X, test_y = test[:, :-nr_col], test[:, -nr_col]
    # reshape input to be 3D [samples, timesteps, features]

    # train_X = train_X[0:len(train_X) // INTERVAL * INTERVAL]
    # test_y = test_y[0:len(test_y)//INTERVAL*INTERVAL]
    # train_y = train_y[0:len(train_y)//INTERVAL*INTERVAL]
    # test_y = test_y.reshape(len(test_y)//INTERVAL, INTERVAL)
    # train_X = train_X.reshape(train_X.shape[0] // INTERVAL, INTERVAL, train_X.shape[1])
    # train_y = train_y.reshape(len(train_y)//INTERVAL, INTERVAL)
    # test_X = test_X.reshape(test_X.shape[0]//INTERVAL, INTERVAL, test_X.shape[1])


    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    return train_X, test_X, train_y, test_y, initial_data.iloc[-n_train_hours:,1]


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
