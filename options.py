import os
import time
import requests
import logging
from requests.auth import HTTPBasicAuth

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load credentials from GitHub secrets
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

# Deribit base URL
DERIBIT_BASE = "https://www.deribit.com/api/v2"

# Step 1: Authenticate with Deribit and get access token
def get_access_token():
    logger.info("Authenticating with Deribit API...")
    url = f"{DERIBIT_BASE}/public/auth"
    params = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    access_token = data["result"]["access_token"]
    logger.info("Access token received.")
    return access_token

# Step 2: Get option instruments for BTC or ETH
def get_option_instruments(access_token, currency="BTC"):
    url = f"{DERIBIT_BASE}/public/get_instruments"
    params = {
        "currency": currency,
        "kind": "option",
        "expired": False
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    instruments = response.json()["result"]
    logger.info(f"Retrieved {len(instruments)} {currency} options.")
    return instruments

# Step 3: Get greeks and option data for a specific instrument
def get_option_data(access_token, instrument_name):
    url = f"{DERIBIT_BASE}/public/ticker"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"instrument_name": instrument_name}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()["result"]

# Example routine
def fetch_options_data():
    try:
        access_token = get_access_token()

        for asset in ["BTC", "ETH"]:
            instruments = get_option_instruments(access_token, asset)

            # Take the nearest expiry option (as example)
            option = instruments[0]
            instrument_name = option["instrument_name"]

            option_data = get_option_data(access_token, instrument_name)

            logger.info(f"\nAsset: {asset}")
            logger.info(f"Instrument: {instrument_name}")
            logger.info(f"Mark Price: {option_data.get('mark_price')}")
            logger.info(f"Delta: {option_data.get('greeks', {}).get('delta')}")
            logger.info(f"Gamma: {option_data.get('greeks', {}).get('gamma')}")
            logger.info(f"Theta: {option_data.get('greeks', {}).get('theta')}")
            logger.info(f"Vega: {option_data.get('greeks', {}).get('vega')}")

            time.sleep(1)  # To avoid rate limits

    except Exception as e:
        logger.error(f"Error fetching options data: {e}")

if __name__ == "__main__":
    fetch_options_data()
