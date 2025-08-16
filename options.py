"""
Delta India – ETH Options Data Extractor (Phase 2: Insider Trader Logic Added)
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, field_validator
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
# Pandas display settings
# -----------------------------
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# -----------------------------
# Environment Variables (Secrets)
# -----------------------------
API_KEY = os.getenv("DELTA_INDIA_API_KEY")
API_SECRET = os.getenv("DELTA_INDIA_API_SECRETE")  # stored for later use

BASE_URL = "https://api.india.delta.exchange"
TIMEOUT = 15
RETRIES = 3
SLEEP_BETWEEN = 0.6

if not API_KEY:
    log.warning("⚠️ No API key found in environment — public endpoints will still work but may be rate-limited.")

# -----------------------------
# Pydantic Models
# -----------------------------
class Quotes(BaseModel):
    best_bid: float | None = Field(None)
    best_ask: float | None = Field(None)

    @field_validator("best_bid", "best_ask", mode="before")
    def to_float(cls, v):
        try:
            return float(v) if v is not None else None
        except:
            return None

class Greeks(BaseModel):
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    rho: float | None = None

    @field_validator("*", mode="before")
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
    spot_price: float | None = None
    open_interest: float | None = None
    iv: float | None = None   # ✅ IV from mark_vol
    volume: float | None = None  # ✅ 24h volume

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
# Fetch ETH Option Chain (quotes, greeks, OI, spot, IV, Volume)
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
            # ✅ capture OI, Spot, IV, Volume directly
            entry["open_interest"] = float(entry.get("oi")) if entry.get("oi") is not None else None
            entry["spot_price"] = float(entry.get("spot_price")) if entry.get("spot_price") is not None else None
            iv_val = entry.get("mark_vol")
            if iv_val is not None:
                try:
                    iv_val = float(iv_val)
                except:
                    iv_val = None
            entry["iv"] = iv_val
            vol_val = entry.get("volume")
            if vol_val is not None:
                try:
                    vol_val = float(vol_val)
                except:
                    vol_val = None
            entry["volume"] = vol_val
            items.append(TickerItem(**entry))
        except Exception as e:
            log.debug("Skipping malformed entry: %s", e)
    return items

# -----------------------------
# Convert to DataFrame
# -----------------------------
def to_dataframe(items):
    rows = []
    for it in items:
        # Derive side
        side = None
        if it.symbol.startswith("C-"):
            side = "CALL"
        elif it.symbol.startswith("P-"):
            side = "PUT"

        # Expiry date parsing fix
        expiry_date = None
        expiry_str = it.symbol.split("-")[-1]
        try:
            if expiry_str.isdigit():  # e.g., 260925
                expiry_date = datetime.strptime(expiry_str, "%d%m%y").date()
            else:  # e.g., 26SEP25
                expiry_date = datetime.strptime(expiry_str.upper(), "%d%b%y").date()
        except:
            expiry_date = None

        bid = it.quotes.best_bid if it.quotes else None
        ask = it.quotes.best_ask if it.quotes else None

        rows.append({
            "symbol": it.symbol,
            "side": side,
            "strike": it.strike_price,
            "expiry_date": expiry_date,
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask) / 2 if bid is not None and ask is not None else None,
            "iv": it.iv,
            "delta": it.greeks.delta if it.greeks else None,
            "gamma": it.greeks.gamma if it.greeks else None,
            "theta": it.greeks.theta if it.greeks else None,
            "vega": it.greeks.vega if it.greeks else None,
            "rho": it.greeks.rho if it.greeks else None,
            "oi": it.open_interest,
            "spot": it.spot_price,
            "volume": it.volume
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["oi_notional"] = df["oi"] * df["spot"]
    return df

# -----------------------------
# Standard Filter for Email Report
# -----------------------------
def filter_options(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.dropna(subset=["strike", "delta", "oi", "spot"])
    spot_price = df['spot'].mean()
    lower_strike = spot_price * 0.9
    upper_strike = spot_price * 1.1

    cond_oi = df["oi_notional"] > 50000
    cond_delta = df["delta"].abs().between(0.3, 0.7)
    cond_strike = df["strike"].between(lower_strike, upper_strike)

    filtered = df[cond_oi & cond_delta & cond_strike]
    return filtered.sort_values(['expiry_date', 'strike', 'side'])

# -----------------------------
# Insider Trader Strangle Selector
# -----------------------------
def select_strangles(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Ensure valid data
    df = df.dropna(subset=["strike", "iv", "delta", "gamma", "theta", "vega", "rho", "volume"])

    # ✅ IV percentile
    df["iv_percentile"] = df["iv"].rank(pct=True) * 100

    # Buy strangle filters
    buy_calls = df[(df["side"] == "CALL") &
                   (df["delta"].between(0.20, 0.30)) &
                   (df["gamma"] >= 0.05) &
                   (df["theta"].abs() <= 0.01) &
                   (df["vega"] >= 0.10) &
                   (df["rho"].abs() <= 0.05) &
                   (df["iv_percentile"] <= 30) &
                   (df["volume"] >= 200)]

    buy_puts = df[(df["side"] == "PUT") &
                  (df["delta"].between(-0.30, -0.20)) &
                  (df["gamma"] >= 0.05) &
                  (df["theta"].abs() <= 0.01) &
                  (df["vega"] >= 0.10) &
                  (df["rho"].abs() <= 0.05) &
                  (df["iv_percentile"] <= 30) &
                  (df["volume"] >= 200)]

    # Sell strangle filters
    sell_calls = df[(df["side"] == "CALL") &
                    (df["delta"].between(0.20, 0.30)) &
                    (df["gamma"] <= 0.02) &
                    (df["theta"].abs() >= 0.03) &
                    (df["vega"].between(0.05, 0.10)) &
                    (df["rho"].abs() <= 0.05) &
                    (df["iv_percentile"] >= 70) &
                    (df["volume"] >= 200)]

    sell_puts = df[(df["side"] == "PUT") &
                   (df["delta"].between(-0.30, -0.20)) &
                   (df["gamma"] <= 0.02) &
                   (df["theta"].abs() >= 0.03) &
                   (df["vega"].between(0.05, 0.10)) &
                   (df["rho"].abs() <= 0.05) &
                   (df["iv_percentile"] >= 70) &
                   (df["volume"] >= 200)]

    # Pair calls & puts
    buy_strangles = pd.merge(buy_calls, buy_puts, on="expiry_date", suffixes=("_call", "_put"))
    sell_strangles = pd.merge(sell_calls, sell_puts, on="expiry_date", suffixes=("_call", "_put"))

    return buy_strangles, sell_strangles

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
    buy_strangles, sell_strangles = select_strangles(df)

    if filtered_df.empty and buy_strangles.empty and sell_strangles.empty:
        log.warning("No options match filter. No email sent.")
        return

    html_parts = []

    if not buy_strangles.empty or not sell_strangles.empty:
        html_parts.append("<h3>🎯 Insider Trader Strangles</h3>")
        if not buy_strangles.empty:
            html_parts.append("<h4>Buy Strangles</h4>")
            html_parts.append(buy_strangles.to_html(index=False))
        if not sell_strangles.empty:
            html_parts.append("<h4>Sell Strangles</h4>")
            html_parts.append(sell_strangles.to_html(index=False))

    if not filtered_df.empty:
        html_parts.append("<h3>📊 Filtered ETH Options Report</h3>")
        html_parts.append(filtered_df.to_html(
            index=False,
            columns=["symbol", "side", "strike", "expiry_date", "bid", "ask", "mid", "iv", "delta", "oi", "oi_notional"],
            justify="center"
        ))

    body = f"<html><body>{''.join(html_parts)}</body></html>"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"ETH Options Report - {datetime.utcnow().strftime('%Y-%m-%d')}"
    msg["From"] = smtp_email
    msg["To"] = smtp_email
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
