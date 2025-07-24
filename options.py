import os
import requests
from datetime import datetime, timezone
from operator import itemgetter

# === Constants ===
BASE_URL = 'https://www.deribit.com/api/v2/'
UNDERLYING = 'BTC'
STRIKE_RANGE = 500  # Range to consider around ATM
RRR = 1 / 2         # Risk-to-Reward ratio

def get_deribit_access_token():
    url = BASE_URL + "public/auth"
    payload = {
        "grant_type": "client_credentials",
        "client_id": os.getenv("DERIBIT_CLIENT_ID"),
        "client_secret": os.getenv("DERIBIT_CLIENT_SECRET")
    }

    response = requests.get(url, params=payload)
    response.raise_for_status()
    return response.json()['result']['access_token']

def fetch_option_instruments(token):
    headers = {'Authorization': f'Bearer {token}'}
    url = BASE_URL + f'public/get_instruments?currency={UNDERLYING}&kind=option&expired=false'
    response = requests.get(url, headers=headers)
    return response.json()['result']

def get_atm_strike(token):
    headers = {'Authorization': f'Bearer {token}'}
    url = BASE_URL + f'public/ticker?instrument_name=BTC-PERPETUAL'
    response = requests.get(url, headers=headers)
    ticker = response.json()['result']
    return round(ticker['last_price'] / 100.0) * 100

def fetch_orderbook(token, instrument_name):
    headers = {'Authorization': f'Bearer {token}'}
    url = BASE_URL + f'public/ticker?instrument_name={instrument_name}'
    response = requests.get(url, headers=headers)
    return response.json()['result']

def select_best_strangle(options, atm_strike):
    nearest_expiry = sorted(set([o['expiration_timestamp'] for o in options]))[0]
    filtered = [
        o for o in options 
        if o['expiration_timestamp'] == nearest_expiry and abs(o['strike'] - atm_strike) <= STRIKE_RANGE
    ]

    call_opts = [o for o in filtered if o['option_type'] == 'call']
    put_opts  = [o for o in filtered if o['option_type'] == 'put']

    best_call = max(call_opts, key=itemgetter('volume'))
    best_put  = max(put_opts, key=itemgetter('volume'))

    return best_call, best_put

def build_trade_plan(option, token):
    ob = fetch_orderbook(token, option['instrument_name'])
    entry = (ob['best_bid_price'] + ob['best_ask_price']) / 2
    risk = entry / 20  # 5% risk
    sl = entry - risk
    tp = entry + (risk * 2)

    return {
        'instrument': option['instrument_name'],
        'type': option['option_type'],
        'strike': option['strike'],
        'expiry': datetime.fromtimestamp(option['expiration_timestamp'] / 1000, tz=timezone.utc).strftime('%Y-%m-%d'),
        'entry': round(entry, 2),
        'stop_loss': round(sl, 2),
        'take_profit': round(tp, 2),
        'rrr': 1 / RRR
    }

def main():
    print("🔐 Connecting to Deribit...")
    token = get_deribit_access_token()

    print("📊 Fetching BTC option instruments...")
    options = fetch_option_instruments(token)

    print("🔍 Determining ATM strike...")
    atm_strike = get_atm_strike(token)

    print("📈 Selecting best strangle pair...")
    best_call, best_put = select_best_strangle(options, atm_strike)

    print("📋 Building trade plans...")
    call_plan = build_trade_plan(best_call, token)
    put_plan = build_trade_plan(best_put, token)

    report = {
        'Call Option': call_plan,
        'Put Option': put_plan
    }

    print("\n✅ Final BTC Options Strangle Recommendation:")
    for key, val in report.items():
        print(f"\n➡️ {key}")
        for k, v in val.items():
            print(f"  {k}: {v}")

    return report

if __name__ == "__main__":
    main()
