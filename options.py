import os
import requests
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# Load credentials from .env
load_dotenv()
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
RECIPIENT_EMAIL = os.getenv("SMTP_EMAIL")

DERIBIT_BASE_URL = "https://www.deribit.com"

# --------------------------
# Deribit API Functions
# --------------------------

def get_access_token():
    url = f"{DERIBIT_BASE_URL}/api/v2/public/auth"
    params = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise Exception(f"Access token error: {data['error']}")
        return data["result"]["access_token"]
    except requests.RequestException as e:
        raise Exception(f"Error fetching access token: {e}")

def get_index_price():
    url = f"{DERIBIT_BASE_URL}/api/v2/public/get_index_price"
    params = {"index_name": "btc_usd"}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data["result"]["index_price"]
    except Exception as e:
        raise Exception(f"Error fetching index price: {e}")

def get_options_instruments():
    url = f"{DERIBIT_BASE_URL}/api/v2/public/get_instruments"
    params = {
        "currency": "BTC",
        "kind": "option",
        "expired": "false"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise Exception(f"Instruments API error: {data['error']}")
        return data.get("result", [])
    except requests.RequestException as e:
        raise Exception(f"Network error while fetching instruments: {e}")

def get_ticker(instrument_name):
    url = f"{DERIBIT_BASE_URL}/api/v2/public/ticker"
    params = {"instrument_name": instrument_name}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise Exception(f"Ticker API error for {instrument_name}: {data['error']}")
        return data.get("result", {})
    except requests.RequestException as e:
        raise Exception(f"Network error while fetching ticker for {instrument_name}: {e}")

def get_price(instrument_name):
    ticker = get_ticker(instrument_name)

    bid = ticker.get("best_bid", 0.0)
    ask = ticker.get("best_ask", 0.0)
    mark_price = ticker.get("mark_price", 0.0)

    if bid and ask:
        price = (bid + ask) / 2
    elif mark_price:
        price = mark_price
    else:
        price = 0.0

    print(f"[DEBUG] {instrument_name} - Bid: {bid}, Ask: {ask}, Mark: {mark_price}, Used: {price}")
    return round(price, 2)

# --------------------------
# Trade Logic
# --------------------------

def select_best_strangle():
    try:
        index_price = get_index_price()
        instruments = get_options_instruments()

        nearest_expiry = min(set(i["expiration_timestamp"] for i in instruments))
        same_expiry = [i for i in instruments if i["expiration_timestamp"] == nearest_expiry]

        atm_call = None
        otm_put = None
        min_call_diff = float("inf")
        max_put_diff = float("-inf")

        for instr in same_expiry:
            strike = instr["strike"]
            option_type = instr["option_type"]
            diff = abs(strike - index_price)

            if option_type == "call":
                if diff < min_call_diff:
                    min_call_diff = diff
                    atm_call = instr

            if option_type == "put" and strike < index_price:
                if strike > max_put_diff:
                    max_put_diff = strike
                    otm_put = instr

        if atm_call and otm_put:
            call_price = get_price(atm_call["instrument_name"])
            put_price = get_price(otm_put["instrument_name"])

            return {
                "call": {
                    "instrument": atm_call["instrument_name"],
                    "strike": atm_call["strike"],
                    "price": call_price,
                },
                "put": {
                    "instrument": otm_put["instrument_name"],
                    "strike": otm_put["strike"],
                    "price": put_price,
                },
                "index_price": index_price
            }
        else:
            return None
    except Exception as e:
        print(f"Strangle selection error: {e}")
        return None

# --------------------------
# Email Reporting
# --------------------------

def send_email(subject, body):
    msg = MIMEMultipart()
    msg["From"] = SMTP_EMAIL
    msg["To"] = SMTP_EMAIL  # Or RECIPIENT_EMAIL if needed
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        print(f"Email failed: {e}")

# --------------------------
# Main Run
# --------------------------

def run_bot():
    print("Running BTC Options Strangle Bot...\n")
    setup = select_best_strangle()

    if setup:
        call = setup["call"]
        put = setup["put"]
        index = setup["index_price"]

        subject = f"BTC Strangle Trade Setup – {datetime.utcnow().strftime('%Y-%m-%d')}"
        body = f"""
BTC Index Price: ${index:,.2f}

🟢 Buy CALL
Instrument: {call['instrument']}
Strike: {call['strike']}
Mark Price: ${call['price']:,.2f}
Entry: ${call['price']:,.2f} | Target: ${round(call['price']*2, 2)} | SL: ${round(call['price']*0.5, 2)}

🔴 Sell PUT
Instrument: {put['instrument']}
Strike: {put['strike']}
Mark Price: ${put['price']:,.2f}
Entry: ${put['price']:,.2f} | Target: ${round(put['price']*2, 2)} | SL: ${round(put['price']*0.5, 2)}

🎯 Strategy: Long CALL + Short PUT (Strangle)
RRR: 1:2
        """
    else:
        subject = f"BTC Options Update – {datetime.utcnow().strftime('%Y-%m-%d')}"
        body = "No suitable strangle setup found based on current filters."

    send_email(subject, body)

