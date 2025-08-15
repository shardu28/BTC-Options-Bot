"""
Delta India – ETH Options Data Extractor (with OI & Spot Price enrichment)
"""

import os
import time
import requests
import pandas as pd
from pydantic import BaseModel, Field, validator
from datetime import datetime
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger("delta-eth-options")

# -----------------------------
# Environment Variables (Secrets)
# -----------------------------
API_KEY = os.getenv("DELTA_INDIA_API_KEY")
API_SECRET = os.getenv("DELTA_INDIA_API_SECRETE")

BASE_URL = "https://api.india.delta.exchange"
TIMEOUT = 15
RETRIES = 3
SLEEP_BETWEEN = 0.4

# -----------------------------
# Pydantic Models
# -----------------------------
class Quotes(BaseModel):
    best_bid: float | None = Field(None)
    best_ask: float | None = Field(None)

    @validator("best_bid", "best_ask", pre=True)
    def to_float(cls, v):
        try:
            return float(v) if v is not None else None
        except:
            return None

class Greeks(BaseModel):
    delta: float | None = None
    iv: float | None = Field(None)

    @validator("*", pre=True)
    def to_float(cls, v):
        try:
            return float(v) if v is not None else None
        except:
            return None

class TickerItem(BaseModel):
    symbol: str
    strike_price: float | None = None
    quotes: Quotes | None = None
    greeks: Greeks | None = None

# -----------------------------
# HTTP GET helper
# -----------------------------
def _get(path: str, params: dict | None = None):
    url = f"{BASE_URL}{path}"
    headers = {"Accept": "application/json"}
    if API_KEY:
        headers["api-key"] = API_KEY

    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            else:
                last_err = f"HTTP {r.status_code}: {r.text}"
                log.warning("Attempt %d failed: %s", attempt, last_err)
        except Exception as e:
            last_err = str(e)
            log.warning("Attempt %d exception: %s", attempt, e)
        time.sleep(SLEEP_BETWEEN)
    raise RuntimeError(f"GET failed after {RETRIES} attempts: {last_err}")

# -----------------------------
# Fetch ETH Option Chain
# -----------------------------
def fetch_eth_options():
    params = {"contract_types": "call_options,put_options", "underlying_asset_symbols": "ETH"}
    raw = _get("/v2/tickers", params)
    result = raw.get("result", raw)

    items = []
    for entry in result:
        try:
            if "quotes" in entry:
                entry["quotes"] = Quotes(**entry["quotes"])
            if "greeks" in entry:
                entry["greeks"] = Greeks(**entry["greeks"])
            items.append(TickerItem(**entry))
        except Exception as e:
            log.debug("Skipping malformed entry: %s", e)
    return items

# -----------------------------
# Fetch Spot Price
# -----------------------------
def fetch_spot_price():
    raw = _get("/v2/underlying-assets", {"symbol": "ETH"})
    assets = raw.get("result", [])
    if assets and isinstance(assets, list):
        return assets[0].get("spot_price")
    return None

# -----------------------------
# Fetch Open Interest for all symbols
# -----------------------------
def fetch_open_interest(symbols):
    oi_map = {}
    for sym in symbols:
        try:
            raw = _get("/v2/open-interest", {"symbol": sym})
            data = raw.get("result", {})
            oi_map[sym] = data.get("open_interest")
        except Exception as e:
            log.debug(f"OI fetch failed for {sym}: {e}")
            oi_map[sym] = None
        time.sleep(SLEEP_BETWEEN)
    return oi_map

# -----------------------------
# Convert to DataFrame and Enrich
# -----------------------------
def to_dataframe(items):
    df = pd.DataFrame([{
        "symbol": it.symbol,
        "side": "CALL" if it.symbol.startswith("C") else "PUT" if it.symbol.startswith("P") else None,
        "strike": it.strike_price,
        "expiry_date": datetime.strptime(it.symbol.split("-")[-1], "%d%b%y").date() if len(it.symbol.split("-")) >= 3 else None,
        "bid": it.quotes.best_bid if it.quotes else None,
        "ask": it.quotes.best_ask if it.quotes else None,
        "mid": (it.quotes.best_bid + it.quotes.best_ask) / 2 if it.quotes and it.quotes.best_bid is not None and it.quotes.best_ask is not None else None,
        "iv": it.greeks.iv if it.greeks else None,
        "delta": it.greeks.delta if it.greeks else None
    } for it in items])

    # Enrich with Spot price
    spot_price = fetch_spot_price()
    df["spot"] = spot_price

    # Enrich with OI
    oi_map = fetch_open_interest(df["symbol"].tolist())
    df["oi"] = df["symbol"].map(oi_map)

    return df

# -----------------------------
# Filter for email report
# -----------------------------
def filter_options(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.dropna(subset=["strike", "delta", "oi", "spot"])
    spot_price = df['spot'].mean()
    lower_strike = spot_price * 0.9
    upper_strike = spot_price * 1.1
    filtered = df[
        (df['oi'] > 500) &
        (df['delta'].abs().between(0.3, 0.7)) &
        (df['strike'].between(lower_strike, upper_strike))
    ]
    return filtered.sort_values(['expiry_date', 'strike', 'side'])

# -----------------------------
# Email sending
# -----------------------------
def send_email_report(df: pd.DataFrame):
    smtp_email = os.environ.get("SMTP_EMAIL")
    smtp_password = os.environ.get("SMTP_PASSWORD")
    if not smtp_email or not smtp_password:
        log.error("SMTP credentials not found.")
        return
    filtered_df = filter_options(df)
    if filtered_df.empty:
        log.warning("No options match filter. No email sent.")
        return
    html_table = filtered_df.to_html(
        index=False,
        columns=["symbol", "side", "strike", "expiry_date", "bid", "ask", "mid", "iv", "delta", "oi", "spot"],
        justify="center"
    )
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Filtered ETH Options Report - {datetime.utcnow().strftime('%Y-%m-%d')}"
    msg["From"] = smtp_email
    msg["To"] = smtp_email
    body = f"<html><body><p>Here is your filtered ETH options report:</p>{html_table}</body></html>"
    msg.attach(MIMEText(body, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(smtp_email, smtp_password)
            server.sendmail(smtp_email, smtp_email, msg.as_string())
        log.info("Email report sent successfully.")
    except Exception as e:
        log.error(f"Failed to send email: {e}")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    log.info("Fetching ETH options data from Delta Exchange India...")
    items = fetch_eth_options()
    df = to_dataframe(items)
    if df.empty:
        log.error("No data fetched!")
    else:
        log.info("Fetched %d contracts", len(df))
        print(df.head(10))
    send_email_report(df)
