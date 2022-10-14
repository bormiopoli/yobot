import datetime, time
from notifications import authenticate_for_gmail_notifications, create_message_with_attachment,\
    generate_notification
from functions import redistribute_small_weights, redistribute_missing_weight, check_structure_conforms_requirements, \
    generate_structure_values_body, MY_STRATEGY, DECIMAL_PLACES, check_percent_of_stable_in_strategy
from functions import test_iconomi, SYNCH_INTERVAL
from logger import logger, root
from functions import test_result_w_binance_data, test_result_w_binance_data_old
import pandas as pd
from multiprocessing import Queue, Process
from keras.models import load_model
import warnings
import numpy as np

warnings.filterwarnings("ignore")


def select_columns_for_training(mydf):
    mydf = mydf[['BTC_index', 'BTC', 'macd', 'osc', 'rsi', 'bollinger_mavg']]
    return mydf


def generate_structure_consumer(prediction, i=0, last_btc_amount=[], last_stable_percent=0):

    gmail_service = authenticate_for_gmail_notifications()


    try:
        stable_percent, list_of_assets = check_percent_of_stable_in_strategy(MY_STRATEGY)

    except Exception as error:
        print("GENERATION OF STRATEGY:", error)
        return

    btc_chosens_strategy_amount = float(str(round(prediction, DECIMAL_PLACES)))
    if  btc_chosens_strategy_amount != round(1-stable_percent, DECIMAL_PLACES) and pd.isna(prediction) == False:
        stables_percent_in_body = round(1-btc_chosens_strategy_amount, DECIMAL_PLACES)
        bodies = [('BTC', btc_chosens_strategy_amount),
                  ('USDT', stables_percent_in_body)
                  #,('PAXG', stables_percent_in_body/3)
                  #,('USDC', stables_percent_in_body/3)
                  ]
        dict_with_ticker_and_weights = redistribute_small_weights(bodies)
        dict_with_ticker_and_weights = redistribute_missing_weight(dict_with_ticker_and_weights)
        if last_stable_percent == round(1-btc_chosens_strategy_amount, DECIMAL_PLACES):
            print(f"SAME PERCENTAGE STRUCTURE: Skipping execution - {last_stable_percent} - Btc Chosen Amount: {btc_chosens_strategy_amount}")
            return

        if len(last_btc_amount)>=2:
            if (abs(last_btc_amount[-2] - btc_chosens_strategy_amount) >= 0.03) and abs(last_btc_amount[-1] - btc_chosens_strategy_amount) >= 0.03:
                pass
            else:
                print(f"Skipping execution - Same structure as the previously set - BTC % : {btc_chosens_strategy_amount}; Current Stable % : {stable_percent} - Array : {last_btc_amount}")
                return
            last_btc_amount.__delitem__(-2)

        last_btc_amount.append(btc_chosens_strategy_amount)
        last_stable_percent = round(1-btc_chosens_strategy_amount, DECIMAL_PLACES)
        print(
            f"CONDITION TRIGGERED: Probabilities: {prediction}; Stable Percent: {stable_percent}; Bodies: {dict_with_ticker_and_weights}; Array : {last_btc_amount}")

    else:
        print("NO CONDITION TRIGGERED - Prob.", prediction)
        return


    print(
        f"ST ASSETS --- STARTING EXECUTION {i} --- {datetime.datetime.strftime(datetime.datetime.today(), '%d/%m/%Y-%H:%M')}")
    bodies_to_write = generate_structure_values_body(dict_with_ticker_and_weights)

    try:
        check_structure_conforms_requirements(bodies_to_write)
    except:
        create_message_with_attachment(gmail_service, 'me',
                                       message_text=f'ST ASSETS 1h - YoBOT assertion error: The structure does not sum to 1.0 - {bodies_to_write}')
        return

    # HERE THE STRUCTURE OF THE STRATEGY IS CHANGED
    try:
        # check_strategy_performance = test_iconomi.test_get_structure(MYSTRATEGY)
        test_iconomi.test_set_structure(values=bodies_to_write)
        generate_notification(gmail_service, None, bodies_to_write, i)


        # if check_strategy_performance:
        #     current_performance = sum([el['estimatedProfit'] for el in check_strategy_performance['values']])
        #     performance += current_performance
        #     print("PERFORMANCE: ", performance, "CURRENT PERFORMANCE: ", current_performance)
        #     bodies_to_write.append({'HOURLY - GAIN SINCE EPOCH 0': performance})

        with open(f"{root}/timestamp_file", 'w+') as fw:
            fw.write(str(time.time()))
            start_time = time.time()

    except Exception as error:
        print(
            f"Set Structure failed: {error} - {datetime.datetime.strftime(datetime.datetime.today(), '%d/%m/%Y-%H:%M')}")
        logger.warning(
            f"Set Structure failed: {error} - {datetime.datetime.strftime(datetime.datetime.today(), '%d/%m/%Y-%H:%M')}")
        logger.info(
            f"This is the structure that failed for execution {i} \n: {bodies_to_write} - {datetime.datetime.strftime(datetime.datetime.today(), '%d/%m/%Y-%H:%M')}")
        print(
            f"This is the structure that failed for execution {i} \n: {bodies_to_write} - {datetime.datetime.strftime(datetime.datetime.today(), '%d/%m/%Y-%H:%M')}")
        return

    try:
        pass
        # # GENERATE STATISTICS REPORT
        # # generate_plotly_chart(df, f'{root}/strategies_weight.png', 'bar_polar')
        # # generate_plotly_chart(df, f'{root}/rebalanced_structure.png',
        # #                       'bar_polar')
        # generate_plotly_chart(bodies_to_write,
        #                       f'{root}/contributions.png', "histogram")
        # # generate_plotly_chart(all_strategies, f'{root}/first_50_strategies_ranking.png',
        # #                       graph_type_enum="bar")
    except Exception as error:
        print(
            f"Set Structure failed: {error} - {datetime.datetime.strftime(datetime.datetime.today(), '%d/%m/%Y-%H:%M')}")
        raise NotImplementedError

    return last_btc_amount, last_stable_percent



def reassign_columns(chunk):
    chunk.columns = ['BTC_open', 'BTC_high', 'BTC_low', 'BTC_close', 'BTC_volume', 'BTC',
                     'BTC_index', 'volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi',
                     'volume_em', 'volume_sma_em', 'volume_vpt', 'volume_vwap', 'volume_mfi',
                     'volume_nvi', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl',
                     'volatility_bbw', 'volatility_bbp', 'volatility_bbhi',
                     'volatility_bbli', 'volatility_kcc', 'volatility_kch', 'volatility_kcl',
                     'volatility_kcw', 'volatility_kcp', 'volatility_kchi',
                     'volatility_kcli', 'volatility_dcl', 'volatility_dch', 'volatility_dcm',
                     'volatility_dcw', 'volatility_dcp', 'volatility_atr', 'volatility_ui',
                     'trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'trend_sma_fast',
                     'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow',
                     'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_ind_diff',
                     'trend_trix', 'trend_mass_index', 'trend_dpo', 'trend_kst',
                     'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_conv',
                     'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b',
                     'trend_stc', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg', 'trend_cci',
                     'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_aroon_up',
                     'trend_aroon_down', 'trend_aroon_ind', 'trend_psar_up',
                     'trend_psar_down', 'trend_psar_up_indicator',
                     'trend_psar_down_indicator', 'momentum_rsi', 'momentum_stoch_rsi',
                     'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d', 'momentum_tsi',
                     'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr',
                     'momentum_ao', 'momentum_roc', 'momentum_ppo', 'momentum_ppo_signal',
                     'momentum_ppo_hist', 'momentum_pvo', 'momentum_pvo_signal',
                     'momentum_pvo_hist', 'momentum_kama', 'others_dr', 'others_dlr',
                     'others_cr']
    return chunk


def chunk_iter_df(df, chunksize=SYNCH_INTERVAL):
    chunks = []
    remaining = len(df)%chunksize
    for c in range(len(df)):
        chunks.append(df.iloc[c*chunksize:(c+1)*chunksize,:])
    if remaining:
        chunks.append(df.iloc[(c+1)*chunksize::,:])
    return chunks


if __name__ == '__main__':

    i = 1
    start_time = time.time()
    m = load_model(f'{root}/keras_model_TD_X')

    print("STARTING")


    i=0
    last_btc_amount=[]
    last_stable_percent = 0

    while True:
        try:
            mytime = time.time()

            # indicator_value = test_result_w_binance_data(m, interval='1m')
            indicator_value = test_result_w_binance_data_old(m, interval='1m')

            if indicator_value is np.nan:
                pass
            else:
                mytuple = generate_structure_consumer(indicator_value,
                                                      i,
                                                      last_btc_amount=last_btc_amount,
                                                      last_stable_percent=last_stable_percent)
                if mytuple is not None:
                    last_btc_amount = mytuple[0]
                    last_stable_percent = mytuple[1]
            i += 1
            time.sleep(round(60 - (time.time()-mytime)) if (time.time()-mytime)<59 else 1)

        except Exception as error:

            try:
                gmail_service = authenticate_for_gmail_notifications()
                create_message_with_attachment(gmail_service, 'me',
                                            message_text=f'ST ASSETS - YoBOT CRASHED 0.0 - ERROR: {error}')
                print(f"ST ASSETS - YoBOT CRASHED 0.0 - ERROR: {error} - {datetime.datetime.strftime(datetime.datetime.today() , '%d/%m/%Y-%H:%M')}")
                logger.error(f"ST ASSETS - YoBOT CRASHED 0.0 - ERROR: {error} - {datetime.datetime.strftime(datetime.datetime.today() , '%d/%m/%Y-%H:%M')}")
                time.sleep(round(60 - (time.time()-mytime)) if (time.time()-mytime)<59 else 1)
            except Exception as err:
                print(err)
                continue
            continue
