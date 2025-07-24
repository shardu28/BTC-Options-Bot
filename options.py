import os
import requests
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

# Load API credentials from GitHub secrets or .env
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

DERIBIT_BASE_URL = "https://www.deribit.com/api/v2"

# Fetch BTC spot price
def get_btc_spot_price():
    try:
        response = requests.get(f"{DERIBIT_BASE_URL}/public/ticker?instrument_name=BTC-PERPETUAL")
        response.raise_for_status()
        data = response.json()
        return data['result']['last_price']
    except Exception as e:
        print(f"Error fetching BTC spot price: {e}")
        return "Unavailable"

# Fetch crypto news from CryptoPanic
def get_crypto_news():
    try:
        response = requests.get("https://cryptopanic.com/api/v1/posts/?auth_token=demo&currencies=BTC&public=true")
        response.raise_for_status()
        news_data = response.json()
        news_items = news_data.get('results', [])[:5]
        headlines = [f"- {item['title']} ({item['published_at'][:10]})" for item in news_items]
        return "\n".join(headlines)
    except Exception as e:
        print(f"Error fetching news: {e}")
        return "No news available"

# Authenticate and get access token
def get_access_token():
    try:
        response = requests.get(f"{DERIBIT_BASE_URL}/public/auth?grant_type=client_credentials&client_id={CLIENT_ID}&client_secret={CLIENT_SECRET}")
        response.raise_for_status()
        return response.json()['result']['access_token']
    except Exception as e:
        print(f"Error fetching access token: {e}")
        return None

# Fetch option instruments
def fetch_option_instruments(token):
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{DERIBIT_BASE_URL}/public/get_instruments?currency=BTC&kind=option&expired=false", headers=headers)
        response.raise_for_status()
        return response.json()['result']
    except Exception as e:
        print(f"Error fetching options: {e}")
        return []

# Select best strangle trade (based on strike & expiry proximity)
def select_strangle_options(options):
    try:
        sorted_options = sorted(options, key=lambda x: (x['expiration_timestamp'], abs(x['strike'] - 50000)))
        call = next((opt for opt in sorted_options if opt['option_type'] == 'call'), None)
        put = next((opt for opt in sorted_options if opt['option_type'] == 'put'), None)
        return call, put
    except Exception as e:
        print(f"Error selecting strangle options: {e}")
        return None, None

# Send email report
def send_email(subject, body):
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_EMAIL
        msg['To'] = SMTP_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.send_message(msg)
        print("Email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")

# Main process
def main():
    access_token = get_access_token()
    if not access_token:
        print("Cannot proceed without access token.")
        return

    btc_price = get_btc_spot_price()
    crypto_news = get_crypto_news()

    options = fetch_option_instruments(access_token)
    call_option, put_option = select_strangle_options(options)

    if not call_option or not put_option:
        print("No valid strangle setup found.")
        return

    # Define entry and exit logic (1:2 RRR)
    call_entry = call_option['last_price']
    call_tp = round(call_entry * 2, 2)
    call_sl = round(call_entry * 0.5, 2)

    put_entry = put_option['last_price']
    put_tp = round(put_entry * 2, 2)
    put_sl = round(put_entry * 0.5, 2)

    # Email body
    body = f"""
📈 BTC Spot Price: ${btc_price}

📰 Top Crypto News:
{crypto_news}

🟢 STRANGLE TRADE SETUP:

🔸 CALL OPTION
Instrument: {call_option['instrument_name']}
Entry Price: {call_entry}
Target (TP): {call_tp}
Stop Loss: {call_sl}

🔹 PUT OPTION
Instrument: {put_option['instrument_name']}
Entry Price: {put_entry}
Target (TP): {put_tp}
Stop Loss: {put_sl}

⏰ Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    send_email(subject="📬 BTC Options Trade Setup - Daily Strangle", body=body)

if __name__ == "__main__":
    main()
