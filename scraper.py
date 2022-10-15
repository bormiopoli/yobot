import datetime, time, pandas as pd, os
import requests
from binance.spot import Spot


# BINANCE API BACKUP: 3KRQ5LDAWDGLVAPQ
CSV_BACKUP_FILE_PATH = "BTC_data.csv"
BINANCE_API_KEY = os.environ['BINANCE_API_KEY']
BINANCE_SECRET = os.environ['BINANCE_SECRET']
MYSECRET = os.environ['MYSECRET']


def remove_ticker_not_in_binance(my_list, client, interval='1m', n_records=0, ):

    all_candles = {}
    for i, value in enumerate(my_list):

        data = {}

        ticker = value +"USDT"
        # Get klines of BTCUSDT at 1m interval
        try:
            data[value] = get_binance_k_lines(ticker, interval, value, client=client, n_records=n_records)

        except Exception as err:

            client = Spot(key=BINANCE_API_KEY, secret=BINANCE_SECRET)
            data[value] = get_binance_k_lines(ticker, interval, value, client)
            print(err)

    return data


def convert_columns_to_numeric(fun_df):
    fun_df = fun_df.dropna(axis=0, how='all')
    fun_df = fun_df.apply(pd.to_numeric, errors='ignore')
    return fun_df


def get_binance_k_lines(ticker, interval, value, myclient, all_data={}, n_records=0):

    if n_records == 0:
        n_records = 1000

    k_values = myclient.klines(ticker, interval=interval, limit=n_records)

    all_data["unix"] = []
    all_data[value + "_open"] = []
    all_data[value + '_high'] = []
    all_data[value + '_low'] = []
    all_data[value + '_close'] = []
    all_data[value + '_volume'] = []

    for i, value_series in enumerate(k_values):
        if i >= n_records:
            break

        if n_records > 1:
            all_data["unix"].append(value_series[0])
            all_data[value + "_open"].append(value_series[1])
            all_data[value + '_high'].append(value_series[2])
            all_data[value + '_low'].append(value_series[3])
            all_data[value + '_close'].append(value_series[4])
            all_data[value + '_volume'].append(value_series[5])
        elif n_records == 1:
            all_data["unix"] = value_series[0]
            all_data[value + "_open"] = value_series[1]
            all_data[value + '_high'] = value_series[2]
            all_data[value + '_low'] = value_series[3]
            all_data[value + '_close'] = value_series[4]
            all_data[value + '_volume'] = value_series[5]
    return all_data


# Producer/Writer
def retrieve_data_producer():
    start = True
    while True:
        client = Spot(key=BINANCE_API_KEY, secret=BINANCE_SECRET)
        try:
            # if os.path.exists(CSV_BACKUP_FILE_PATH):
            #     current_saved_data = pd.read_csv(CSV_BACKUP_FILE_PATH)

            temp_dict_of_strategies_reputation = remove_ticker_not_in_binance(['BTC'], client=client)
            # temp_dict_of_strategies_reputation = {**temp_dict_of_strategies_reputation, **indicators}
            df = pd.DataFrame.from_dict(temp_dict_of_strategies_reputation['BTC'])
            #df = select_base_columns(df)
            df = convert_columns_to_numeric(df)
            # df = divide_by_mean_volume(df)
            df = df[['unix', 'BTC_open', 'BTC_high', 'BTC_low', 'BTC_close',
                     'BTC_volume']]  # SELECT COLUMNS IN RIGHT ORDER FOR LATER CONCAT

            if df is not None:
                print("df is not None")
                # df = add_all_ta_features(df, open="BTC_open", high="BTC_high", low="BTC_low", close="BTC_close",
                # volume="BTC_volume", fillna=True, conversion_interval=1)

                # df = ta_batch_make(df, day_conversion_factor=60*24 if len(df)>60*24*52 else round(60*24*len(df)//208))
                if start == False:
                    df = df.tail(1)

                header_condition = False if os.path.isfile(CSV_BACKUP_FILE_PATH) else True
                df.to_csv(CSV_BACKUP_FILE_PATH, index=False, header=header_condition, mode='a')
                print('APPENDED DATA INTO CSV - ',
                      datetime.datetime.strftime(datetime.datetime.today(), '%d/%m/%Y-%H:%M'))
            else:
                print("df is None")

            time.sleep(60)
            start = False

        except Exception as err:
            print("ERROR IN data producer : ", str(err))
            continue


if '__main__' == __name__:
    retrieve_data_producer()