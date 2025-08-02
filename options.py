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

    return round(price, 2), round(mark_price, 2)

# --------------------------
# Trade Logic
# --------------------------

def select_best_strangle():
    try:
        index_price = get_index_price()
        instruments = get_options_instruments()

        # Filter out dead options (no bid/ask/mark price)
        live_instruments = []
        for instr in instruments:
            ticker = get_ticker(instr["instrument_name"])
            if ticker.get("mark_price") or (ticker.get("best_bid") and ticker.get("best_ask")):
                live_instruments.append(instr)

        nearest_expiry = min(set(i["expiration_timestamp"] for i in live_instruments))
        same_expiry = [i for i in live_instruments if i["expiration_timestamp"] == nearest_expiry]

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
            call_entry, call_mark = get_price(atm_call["instrument_name"])
            put_entry, put_mark = get_price(otm_put["instrument_name"])

            return {
                "call": {
                    "instrument": atm_call["instrument_name"],
                    "strike": atm_call["strike"],
                    "entry": call_entry,
                    "mark_price": call_mark,
                },
                "put": {
                    "instrument": otm_put["instrument_name"],
                    "strike": otm_put["strike"],
                    "entry": put_entry,
                    "mark_price": put_mark,
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
    SMTP_EMAIL = os.getenv("SMTP_EMAIL")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

    if not SMTP_EMAIL or not SMTP_PASSWORD:
        print("❌ Environment variables SMTP_EMAIL or SMTP_PASSWORD not found.")
        return

    msg = MIMEMultipart()
    msg["From"] = SMTP_EMAIL
    msg["To"] = SMTP_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    print("📤 Trying to send email from:", SMTP_EMAIL)

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("✅ Email sent successfully.")
    except Exception as e:
        print("❌ Email sending failed:", e)

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
Mark Price: ${call['mark_price']:,.2f}
Entry: ${call['entry']:,.2f} | Target: ${round(call['entry']*2, 2)} | SL: ${round(call['entry']*0.5, 2)}

🔴 Sell PUT
Instrument: {put['instrument']}
Strike: {put['strike']}
Mark Price: ${put['mark_price']:,.2f}
Entry: ${put['entry']:,.2f} | Target: ${round(put['entry']*2, 2)} | SL: ${round(put['entry']*0.5, 2)}

🎯 Strategy: Long CALL + Short PUT (Strangle)
RRR: 1:2
        """
    else:
        subject = f"BTC Options Update – {datetime.utcnow().strftime('%Y-%m-%d')}"
        body = "No suitable strangle setup found based on current filters."

    print(f"Email Subject:\n{subject}")
    print(f"Email Body:\n{body}")

    send_email(subject, body)

run_bot()
