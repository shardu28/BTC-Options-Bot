import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

load_dotenv()
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
DERIVATION = "BTC"  # or 'WBTC' if labeled

LYRA_SUBGRAPH = "https://api.thegraph.com/subgraphs/name/lyra-finance/optimism-v2"
GRAPHQL_QUERY = """
query fetchOptions($market: String!) {
  options(first: 200, where: {market: $market, openInterest_gt: 10}, orderBy: strikePrice, orderDirection: asc) {
    id
    strikePrice
    expiryTimestamp
    isCall
    openInterest
    impliedVolatility
    delta
    volume
    premium
    board { expiryTimestamp }
  }
}
"""

def fetch_options(market="WBTC"):
    resp = requests.post(LYRA_SUBGRAPH, json={'query': GRAPHQL_QUERY, 'variables': {'market': market}})
    data = resp.json().get("data", {}).get("options", [])
    return data

def next_friday_ts():
    today = datetime.utcnow().date()
    fd = today + timedelta((4 - today.weekday()) % 7)
    return int(datetime(fd.year, fd.month, fd.day, 0, 0).timestamp())

def select_strangle(opts):
    target_expiry = next_friday_ts()
    arr = [o for o in opts if o['expiryTimestamp'] == target_expiry]
    if not arr: return None

    calls = [o for o in arr if o['isCall'] and 0.3 <= o['delta'] <= 0.5]
    puts = [o for o in arr if not o['isCall'] and -0.5 <= o['delta'] <= -0.3]

    # sort by liquidity: openInterest × volume
    calls.sort(key=lambda o: o['openInterest'] * o['volume'], reverse=True)
    puts.sort(key=lambda o: o['openInterest'] * o['volume'], reverse=True)
    if not calls or not puts: return None

    call = calls[0]
    put = next((p for p in puts if p['strikePrice'] < call['strikePrice']), puts[0])

    return call, put

def send_email(subject, body):
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        print("❌ Missing SMTP credentials")
        return
    msg = MIMEMultipart()
    msg["From"] = SMTP_EMAIL
    msg["To"] = SMTP_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    print(f"📤 Sending email from {SMTP_EMAIL}...")
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("✅ Email Sent.")
    except Exception as e:
        print("❌ Email error:", e)

def run_bot():
    opts = fetch_options("WBTC")
    out = select_strangle(opts)
    if not out:
        send_email("BTC Strangle Bot – No Trade", "No suitable setup found")
        return
    call, put = out
    body = f"""
💡 BTC Strangle Setup (Expiry Friday)

📈 Buy Call: {call['id']}
 Strike: {call['strikePrice']}
 Premium: {call['premium']}
 IV: {call['impliedVolatility']:.3f}
 Delta: {call['delta']}
 OI×Vol: {call['openInterest']}×{call['volume']}

📉 Sell Put: {put['id']}
 Strike: {put['strikePrice']}
 Premium: {put['premium']}
 IV: {put['impliedVolatility']:.3f}
 Delta: {put['delta']}
 OI×Vol: {put['openInterest']}×{put['volume']}

🎯 Entry Spread: Call premium − Put premium = ${call['premium'] - put['premium']:.2f}
 Target (×2): ${2 * (call['premium'] - put['premium']):.2f}
 Stop Loss (×1): ${(call['premium'] - put['premium']):.2f}

Isolated Strangle: Long Call, Short Put
Risk‑Reward Ratio: 1:2
"""
    send_email("BTC Strangle from Lyra", body)

if __name__ == "__main__":
    run_bot()
