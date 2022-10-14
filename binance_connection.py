import json
import os
import time
import requests
from binance.spot import Spot

#BINANCE API BACKUP: 3KRQ5LDAWDGLVAPQ

BINANCE_API_KEY = os.environ['BINANCE_API_KEY']
BINANCE_SECRET = os.environ['BINANCE_SECRET']
MYSECRET = os.environ['MYSECRET']

# values = ['1INCH', 'AAVE', 'ADaA', 'ADX', 'AION', 'AKRO', 'ALGO', 'ALPHA', 'ANKR', 'ANT', 'AR', 'AST', 'ATOM', 'AUDIO', 'AVAX', 'AXS', 'BADGER', 'BAL', 'BAND', 'BAT', 'BCH', 'BLZ', 'BNB', 'BNT', 'BTC', 'BTS', 'CAKE', 'CELO', 'CHR', 'CHZ', 'COMP', 'COTI', 'CRO', 'CRV', 'CVC', 'CVX', 'DASH', 'DCR', 'DENT', 'DGB', 'DNT', 'DOGE', 'DOT', 'DYDX', 'EGLD', 'ELF', 'ENJ', 'EOS', 'ETC', 'ETH', 'FET', 'FIL', 'FLOW', 'FRONT', 'FTM', 'FTT', 'FXS', 'GALA', 'GAS', 'GNT', 'GRT', 'HBAR', 'ICP', 'ICX', 'INJ', 'IRIS', 'JASMY', 'KAVA', 'KMD', 'KNC', 'KP3R', 'KSM', 'LDO', 'LINA', 'LINK', 'LOOM', 'LRC', 'LSK', 'LTC', 'MANA', 'MATIC', 'MDX', 'MINA', 'MIR', 'MITH', 'MKR', 'MLN', 'MTL', 'NANO', 'NEAR', 'NEO', 'NULS', 'OCEAN', 'OGN', 'OMG', 'ONE', 'OXT', 'PAXG', 'PNT', 'POWR']
client = Spot(key=BINANCE_API_KEY, secret=BINANCE_SECRET)


def get_binance_k_lines(ticker, interval, value, all_data={}, myclient=client, n_records=0):

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


def remove_ticker_not_in_binance(my_list, interval='1m', n_records=0):

    all_candles = {}
    for i, value in enumerate(my_list):

        data = {}

        ticker = value +"USDT"
        # Get klines of BTCUSDT at 1m interval
        try:
            data[value] = get_binance_k_lines(ticker, interval, value, n_records=n_records)

        except Exception as err:

            client = Spot(key=BINANCE_API_KEY, secret=BINANCE_SECRET)
            data[value] = get_binance_k_lines(ticker, interval, value, myclient=client)
            print(err)

    return data


def retrieve_indicator(ticker, candles):
    all_indicators = {}
    if type(ticker) == str:
        ticker = [ticker]
    for tick in ticker:
        all_indicators[tick] = get_indicator(tick, candles=candles)
    return all_indicators


def get_indicator(ticker, interval='1m'):
    indicators = {}
    indicator_codes = ['rsi', 'aroonosc', 'macd', 'wad', 'obv', 'stoch', 'aroon', 'adx', 'fibonacciretracement']
    #indicator_codes = ['rsi', 'aroonosc']

    for indicator_ticker in indicator_codes:
        try:
            url = f"https://api.taapi.io/{indicator_ticker}?secret={MYSECRET}&exchange=binance&symbol={ticker}/USDT&interval={interval}"
            indicator = requests.get(url)
            content = indicator.content.decode('ascii')
            json_content = json.loads(content)
            indicators[indicator_ticker] = json_content
            print(f"RETRIEVED {indicator_ticker} FOR {ticker}")
            time.sleep(15)
        except Exception as error:
            print(error)
            print(f"THIS INDICATOR: {indicator_ticker} COULD NOT BE RETRIEVED")

    return indicators


def retrieve_bulk_indicators_body(ticker, interval="1m", MY_SECRET=MYSECRET):
    if type(ticker) == str:
        body = {
            "secret": f"{MY_SECRET}",
            "construct": {
                "exchange": "binance",
                "symbol": f"{ticker}/USDT",
                "interval": f"{interval}",
                "indicators": [
                    {
                        "indicator": "rsi"
                    },
                    {
                        "indicator": "aroonosc",
                    },
                    {
                        "indicator": "macd",
                    },
                    {
                        "indicator": "wad",
                    },
                    {
                        "indicator": "obv",
                    },
                    {
                        "indicator": "stoch",
                    },
                    {
                        "indicator": "aroon",
                    },
                    {
                        "indicator": "adx",
                    },
                    {
                        "indicator": "fibonacciretracement",
                    }
                ]
                }
            }
        return body

    elif type(ticker) == list:
        to_return = {}
        for el in ticker:
            to_return[el] = retrieve_bulk_indicators_body(el)
        return to_return


def get_bulk_indicators(ticker="BTC", interval="1m", MY_SECRET=MYSECRET):
    url = "https://api.taapi.io/bulk"
    bodies = retrieve_bulk_indicators_body(ticker)
    all_indicators = {}
    for ticker in bodies:
        body = bodies[ticker]
        response = requests.post(url=url, json=body)
        if response.status_code == 200:
            response = response.content.decode('ascii')
            response = json.loads(response)
            all_indicators[ticker] = {}
            for el in response['data']:
                all_indicators[ticker][el['id']] = el['result']
    return all_indicators



