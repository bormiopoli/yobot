#!/bin/python3
import hmac
import hashlib
import base64
import os
import unittest
import requests
import time
import json
from logger import logger

MYSTRATEGY = os.environ['MYSTRATEGY']

class TestIconomi():

    # test api secret
    API_URL = "https://api.iconomi.com"
    API_SECRET = os.environ['API_SECRET']
    API_KEY = os.environ['API_KEY']

    def get_statistics(self, ticker, type='ALL_TIME'):
        ret = self.get(f'/v1/assets/{ticker}/statistics')
        try:
            reputation = ret['returns'][type]
        except:
            reputation = None
        return reputation

    def get_price_history(self, ticker=MYSTRATEGY, granulation=None, fromv=None, to=None):
        granulation = "?="+str(granulation) if granulation is not None else ""
        fromv = "?="+str(fromv) if fromv is not None else ""
        to = "?="+str(to) if to is not None else ""
        ret = self.call('GET', f'/v1/strategies/{ticker}/pricehistory{granulation+fromv+to}', '')
        return ret

    def test_strategies(self, field='ticker'):
        ret = self.get('/v1/strategies')
        yields = []
        for el in ret:
            yields.append(el[field])
        return yields

    def get_strategy_tradable_assets(self, field='ticker'):
        ret = self.get('/v1/assets')
        yields = []
        for el in ret:
            if el['supported'] == True and el['useInStrategy'] == True and el['status'] == 'ONLINE':
                yields.append(el[field])
        return yields

    def get_balance(self):
        self.get('/v1/user/balance')

    def test_activity(self):
        self.get('/v1/user/activity')

    def test_get_structure(self, ticker=MYSTRATEGY):
        ret = self.get('/v1/strategies/' + ticker + '/structure')
        return ret

    def test_get_asset_prices(self, ticker=MYSTRATEGY):
        ret = self.get('/v1/assets/' + ticker + '/price')
        return ret

    def test_get_strategy_prices(self, ticker=MYSTRATEGY):
        ret = self.get('/v1/strategies/' + ticker + '/price')
        return ret

    def test_set_structure(self, values=None, ticker = MYSTRATEGY, speed="MEDIUM"):
        if values == None:
            raise "insert 'values' field"
        payload = {
          'ticker': ticker,
          'values': values,
          'speedType': speed
        }

        self.post('/v1/strategies/' + ticker + '/structure', payload)

    #def websocket_subscribe(self, ticker):


    def test_withdraw(self):
        payload = {
            'amount': '0.02',
            'currency': 'ETH',
            'address': ''
        }

        self.post('/v1/user/withdraw', payload)

    def generate_signature(self, payload, request_type, request_path, timestamp): 
        data = ''.join([timestamp, request_type.upper(), request_path, payload]).encode()
        signed_data = hmac.new(self.API_SECRET.encode(), data, hashlib.sha512)
        return base64.b64encode(signed_data.digest())

    def get(self, api):      
        ret = self.call('GET', api, '')
        return ret

    def post(self, api, payload):
        self.call('POST', api, payload)
        
    def call(self, method, api, payload):
        timestamp = str(int(time.time() * 1000.0))

        jsonPayload = payload
        if method == 'POST':
          jsonPayload = json.dumps(payload)

        requestHeaders = {
            'Content-Type': 'application/json',
            'ICN-API-KEY' : self.API_KEY,
            'ICN-TIMESTAMP' : timestamp,
            'ICN-SIGN' : self.generate_signature(jsonPayload, method, api, timestamp)
        }

        try:
            if method == 'GET':
              response = requests.get(self.API_URL + api, headers = requestHeaders, timeout=10)
              if response.status_code == 200:
                return json.loads(response._content)
              else:
                print(f'Request did not succeed:  {response.reason} - API_URL:{self.API_URL+api}')
                logger.warn(f'Request did not succeed:  {response.reason} - API_URL:{self.API_URL+api}')
                if 'unauthorized' in response.reason.lower():
                    if 'permission denied' not in response.reason.lower():
                        return self.API_URL+api
                    else:
                        print(f"WARNING: Permission denied for API: {self.API_URL+api}")

                else:
                    raise ConnectionError('Request did not succeed: ' + response.reason)

            elif method == 'POST':

              response = requests.post(self.API_URL + api, json = payload, headers = requestHeaders, timeout=6)
              if response.status_code == 200:
                return json.loads(response._content)
              else:
                print('Request did not succeed: ' + response.reason, f"; API_URL:{self.API_URL+api}", f" Payload: {payload}")
                logger.warn(f'Request did not succeed: {response.reason} - API_URL:{self.API_URL+api}')
                raise ConnectionError('Request did not succeed: ' + response.reason)

        except Exception as err:
            if ('unauthorized' in str(err).lower()) or ('internal server' in str(err).lower()) or ('Request did not succeed: ' == str(err)) or ('timed out' in str(err)):
                # print(err)
                # logger.info(err)
                pass
            else:
                print(f"RAISING ERROR: {err}")
                logger.error(err)
                raise err


if __name__ == "__main__":
    unittest.main()
