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

# Fine-tuned option Greek filters (adjusted for $20/day goal)
greek_filters = {
    "delta_min": 0.15,
    "delta_max": 0.45,
    "theta_min": 0.015,
    "vega_max": 0.25,
    "gamma_max": 0.15,
    "min_volume": 30,
    "min_oi": 80
}

def get_option_greeks(option):
    return {
        "delta": option.get("greeks", {}).get("delta", 0),
        "theta": option.get("greeks", {}).get("theta", 0),
        "vega": option.get("greeks", {}).get("vega", 0),
        "gamma": option.get("greeks", {}).get("gamma", 0),
        "volume": option.get("volume", 0),
        "oi": option.get("open_interest", 0),
        "price": option.get("last_price", 0),
        "iv": option.get("iv", 0)
    }

def is_valid_option(option):
    g = get_option_greeks(option)
    return (
        greek_filters["delta_min"] <= abs(g["delta"]) <= greek_filters["delta_max"]
        and g["theta"] >= greek_filters["theta_min"]
        and g["vega"] <= greek_filters["vega_max"]
        and g["gamma"] <= greek_filters["gamma_max"]
        and g["volume"] >= greek_filters["min_volume"]
        and g["oi"] >= greek_filters["min_oi"]
    )

def expected_payoff(option):
    g = get_option_greeks(option)
    delta_value = abs(g["delta"]) * g["price"]
    theta_value = g["theta"] * g["price"]
    vega_penalty = g["vega"] * 0.1
    return delta_value + theta_value - vega_penalty

def rank_options(options):
    valid_opts = [opt for opt in options if is_valid_option(opt)]
    ranked = sorted(valid_opts, key=expected_payoff, reverse=True)
    return ranked

def find_best_strangle(calls, puts, target_payoff=20):
    ranked_calls = rank_options(calls)
    ranked_puts = rank_options(puts)
    best_strangles = []

    for call in ranked_calls:
        for put in ranked_puts:
            total_payoff = expected_payoff(call) + expected_payoff(put)
            if total_payoff >= target_payoff:
                best_strangles.append((call, put, total_payoff))

    if best_strangles:
        best_strangles.sort(key=lambda x: x[2], reverse=True)
        best_call, best_put, payoff = best_strangles[0]
        return {
            "call": best_call,
            "put": best_put,
            "expected_payoff": payoff
        }
    return None

def fetch_option_chain():
    response = requests.get(f"{DERIBIT_BASE_URL}/public/get_instruments?currency=BTC&kind=option")
    instruments = response.json().get("result", [])
    all_options = []
    for instrument in instruments:
        details = requests.get(f"{DERIBIT_BASE_URL}/public/ticker?instrument_name={instrument['instrument_name']}").json()
        if details.get("result"):
            all_options.append(details["result"])
    return all_options

def separate_calls_and_puts(options):
    calls = [opt for opt in options if "C" in opt["instrument_name"]]
    puts = [opt for opt in options if "P" in opt["instrument_name"]]
    return calls, puts

def send_email(subject, body):
    msg = MIMEMultipart()
    msg['From'] = SMTP_EMAIL
    msg['To'] = SMTP_EMAIL
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print("Error sending email:", e)

if __name__ == "__main__":
    all_options = fetch_option_chain()
    calls, puts = separate_calls_and_puts(all_options)
    best_strangle = find_best_strangle(calls, puts, target_payoff=20)

    if best_strangle:
        call = best_strangle["call"]
        put = best_strangle["put"]
        payoff = best_strangle["expected_payoff"]

        body = f"""
        Daily BTC Options Strangle Suggestion:

        📈 Call Option: {call['instrument_name']}
        Price: {call['last_price']} | Delta: {call['greeks']['delta']} | Theta: {call['greeks']['theta']}

        📉 Put Option: {put['instrument_name']}
        Price: {put['last_price']} | Delta: {put['greeks']['delta']} | Theta: {put['greeks']['theta']}

        ✅ Expected Combined Payoff: ${round(payoff, 2)}
        """
        send_email("BTC Options Strangle Suggestion - Daily", body)
    else:
        send_email("BTC Options Alert", "No suitable strangle found today based on current filters.")
