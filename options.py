import os
import requests
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Load credentials from environment variables
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

BASE_URL = "https://www.deribit.com/api/v2"

# Authenticate with Deribit API
def authenticate():
    response = requests.get(
        f"{BASE_URL}/public/auth",
        params={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
    )
    response.raise_for_status()
    return response.json()["result"]["access_token"]

# Fetch instruments for BTC or ETH options
def get_option_instruments(token, currency):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(
        f"{BASE_URL}/public/get_instruments",
        headers=headers,
        params={"currency": currency, "kind": "option", "expired": False},
    )
    response.raise_for_status()
    return response.json()["result"]

# Get greeks and mark price for a given instrument
def get_option_details(token, instrument_name):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(
        f"{BASE_URL}/public/ticker",
        headers=headers,
        params={"instrument_name": instrument_name},
    )
    response.raise_for_status()
    return response.json()["result"]

# Select ATM/OTM call and put options for closest expiry
def select_strangle_options(instruments):
    if not instruments:
        return None, None

    # Sort by expiration date and strike
    instruments.sort(key=lambda x: (x["expiration_timestamp"], x["strike"]))
    expiry = instruments[0]["expiration_timestamp"]
    expiry_instruments = [i for i in instruments if i["expiration_timestamp"] == expiry]

    atm_strike = min(expiry_instruments, key=lambda x: abs(x["strike"] - x["index_price"]))["strike"]

    # Find nearest OTM call and put options
    otm_call = next((i for i in expiry_instruments if i["option_type"] == "call" and i["strike"] > atm_strike), None)
    otm_put = next((i for i in expiry_instruments if i["option_type"] == "put" and i["strike"] < atm_strike), None)

    return otm_call, otm_put

# Format the trade setup

def format_trade_signal(option, details):
    premium = details["mark_price"]
    sl = round(premium * 0.5, 4)
    target = round(premium * 2, 4)
    strike = option["strike"]
    expiry = datetime.utcfromtimestamp(option["expiration_timestamp"] / 1000).strftime("%Y-%m-%d")
    return (
        f"Buy {option['option_type'].upper()} Option\n"
        f"Instrument: {option['instrument_name']}\n"
        f"Strike: {strike}, Expiry: {expiry}\n"
        f"Premium: {premium}, SL: {sl}, Target: {target}\n"
    )

def main():
    token = authenticate()

    for asset in ["BTC", "ETH"]:
        print(f"\n=== {asset} Option Trade Setup ===")
        instruments = get_option_instruments(token, asset)

        # Add index price to each instrument for ATM calc
        index_price = next((i for i in instruments if i["instrument_name"].endswith("C")), {}).get("index_price", 0)
        for i in instruments:
            i["index_price"] = index_price

        call_option, put_option = select_strangle_options(instruments)

        if call_option and put_option:
            call_details = get_option_details(token, call_option["instrument_name"])
            put_details = get_option_details(token, put_option["instrument_name"])

            print(format_trade_signal(call_option, call_details))
            print(format_trade_signal(put_option, put_details))
        else:
            print("No valid strangle options found.")

if __name__ == "__main__":
    main()
