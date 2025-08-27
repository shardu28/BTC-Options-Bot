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

        # Build mid from any available side to avoid dropping one-sided books
        if bid is None and ask is None:
            mid = None
        elif bid is None:
            mid = ask
        elif ask is None:
            mid = bid
        else:
            mid = (bid + ask) / 2

        rows.append({
            "symbol": it.symbol,
            "side": side,
            "strike": it.strike_price,
            "expiry_date": expiry_date,
            "bid": bid,
            "ask": ask,
            "mid": mid,
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
    
    df = df.dropna(subset=["strike", "delta", "oi", "spot", "volume", "iv"])
    spot_price = df['spot'].mean()

    # Ensure expiry_date is datetime
    df = df.copy()
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    
    # Range tighter around spot (±5%)
    lower_strike = spot_price * 0.95
    upper_strike = spot_price * 1.05

    # Compute days to expiry (fix: use Timestamp.today(), not .date())
    cond_dte = (df['expiry_date'] - pd.Timestamp.today()).dt.days.between(2, 7)
    cond_oi = df["oi"] >= 100
    cond_vol = df["volume"] >= 20
    cond_strike = df["strike"].between(lower_strike, upper_strike)

    filtered = df[cond_dte & cond_oi & cond_vol & cond_strike]
    return filtered.sort_values(['expiry_date', 'strike', 'side'])

def fetch_spot_history(symbol="ETHUSD", resolution="1h", lookback_hours=72):
    """
    Fetch spot OHLC history from Delta API.
    Returns: pandas.DataFrame with OHLCV or empty DataFrame on failure
    """
    try:
        end = int(time.time())  # current unix time (seconds)
        start = end - lookback_hours * 3600

        url = "https://api.india.delta.exchange/v2/history/candles"
        params = {
            "resolution": resolution,
            "symbol": symbol,
            "start": start,
            "end": end,
        }
        headers = {"Accept": "application/json"}

        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json().get("result", [])

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.sort_values("time").reset_index(drop=True)
        return df

    except Exception as e:
        print(f"[WARN] Failed to fetch spot history for {symbol}: {e}")
        return pd.DataFrame()

def detect_directional_bias(candles: pd.DataFrame, fast=9, slow=21):
    """
    Detects market bias using EMAs and RSI from spot OHLC candles.
    Fallback: returns 'neutral' if candles are missing or too short.
    """
    if candles.empty or len(candles) < slow + 1:
        print("[INFO] Insufficient candles for bias calculation. Defaulting to NEUTRAL.")
        return "neutral"

    closes = candles["close"].astype(float)

    # EMA calculation
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()

    # RSI calculation (14-period)
    delta = closes.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Bias decision
    if ema_fast.iloc[-1] > ema_slow.iloc[-1] and rsi.iloc[-1] > 55:
        return "bullish"
    elif ema_fast.iloc[-1] < ema_slow.iloc[-1] and rsi.iloc[-1] < 45:
        return "bearish"
    else:
        return "neutral"

# -----------------------------
# Insider Trader Strategy Selector (JSON -> Python)
# -----------------------------
def select_strangles(
    df: pd.DataFrame, 
    *,
    iv_rank_pct: float | None = None,     # 0-100 (if None, computed from chain IV)
    iv_30d: float | None = None,          # e.g., 0.65 for 65% (optional but enables expected-move constraints)
    rv_7d: float | None = None,           # realized vol (0.xx) - optional
    rv_14d: float | None = None,          # realized vol (0.xx) - optional
    event_risk_level: str = "low",        # "low" | "medium" | "high"
    directional_bias: str = "neutral"     # "bullish" | "bearish" | "neutral"
):
    """
    Returns a dict with the selected strategy and legs, or {"strategy": "no_trade"} if nothing qualifies.
    This function does NOT mutate the data pipeline and only consumes the df produced by to_dataframe().
    """

    result = {"strategy": "no_trade", "reason": "No qualifying trade found."}
    if df.empty:
        result["reason"] = "Empty dataframe"
        return result

    # --- Safety: Clean, compute helpers ---
    df = df.copy()
    # Require core fields, allow one-sided quotes (do not require bid/ask/mid here)
    df = df.dropna(subset=["expiry_date", "side", "strike", "delta", "oi", "volume", "iv", "spot"])

    # If mid is missing, build from whichever side is available
    missing_mid = df["mid"].isna()
    df.loc[missing_mid & df["ask"].notna(), "mid"] = df.loc[missing_mid & df["ask"].notna(), "ask"]
    df.loc[missing_mid & df["bid"].notna(), "mid"] = df.loc[missing_mid & df["bid"].notna(), "bid"]

    # Robust DTE: ceil of fractional days to avoid off-by-one due to intraday time
    exp_ts = pd.to_datetime(df["expiry_date"]).dt.tz_localize(None)
    now = pd.Timestamp.utcnow().tz_localize(None)
    df["dte"] = np.ceil((exp_ts - now) / np.timedelta64(1, "D")).astype(int)

    # Bid-ask spread pct with clamped denominator; penalize one-sided books
    denom = df["mid"].copy()
    denom = denom.mask(denom < 1.0, 1.0)
    df["spread_pct"] = ((df["ask"] - df["bid"]) / denom) * 100
    one_sided = df["bid"].isna() | df["ask"].isna()
    df.loc[one_sided, "spread_pct"] = 999.0

    # --- IV Rank ---
    if iv_rank_pct is None:
        if "iv" in df.columns and not df["iv"].isnull().all():
            df["iv_rank_pct"] = df["iv"].rank(pct=True) * 100
            iv_rank_pct = float(np.clip(df["iv_rank_pct"].mean(), 0, 100))
        else:
            iv_rank_pct = None

    # Spot from chain if not provided
    spot = float(df["spot"].mean())

    # --- Blended RV logic ---
    if rv_7d is not None and rv_14d is not None:
        blended_rv = (rv_7d + rv_14d) / 2.0
    elif rv_7d is not None:
        blended_rv = rv_7d
    elif rv_14d is not None:
        blended_rv = rv_14d
    else:
        # No RV data provided: rely on Global Filters below (avoid strict pre-filter that can wipe chain)
        blended_rv = None

    # --- Global Filters from JSON ---
    GF = {
        "expiry_dte_min": 2,
        "expiry_dte_max": 9,  # widened slightly so near-weekly and next-week survive
        "min_open_interest": 1,
        "min_volume": 1,
        "max_bid_ask_spread_pct": 6,
        "max_slippage_pct_per_leg": 15.0,
    }

    base = df[
        (df["dte"] >= GF["expiry_dte_min"]) &
        (df["dte"] <= GF["expiry_dte_max"]) &
        (df["oi"] >= GF["min_open_interest"]) &
        (df["volume"] >= GF["min_volume"]) &
        (df["spread_pct"] <= GF["max_bid_ask_spread_pct"])
    ].copy()

    if base.empty:
        result["reason"] = f"No contracts left after global filters (len(df)={len(df)})"
        return result

    sides_present = set(base["side"].unique())

    # --- Decision Layer (side-aware, returns only implementable strategies) ---
    def choose_strategy(iv_rank_pct, iv_30d, blended_rv, event_risk_level, directional_bias, sides_present):
        cheap_iv = (iv_rank_pct is not None and iv_rank_pct <= 35)
        high_iv  = (iv_rank_pct is not None and iv_rank_pct >= 50)

        if iv_30d is not None and blended_rv is not None and blended_rv > 0:
            ratio = iv_30d / blended_rv
            iv_vs_rv_ok_long  = (ratio <= 1.0)
            iv_vs_rv_ok_short = (ratio >= 1.2)
        else:
            iv_vs_rv_ok_long  = cheap_iv
            iv_vs_rv_ok_short = high_iv

        have_calls = "CALL" in sides_present
        have_puts  = "PUT" in sides_present

        # Directional bias first, but only return strategies we implement and have legs for
        if directional_bias == "bullish":
            if have_calls and iv_vs_rv_ok_long:
                return "long_call"
            if have_puts and iv_vs_rv_ok_short and event_risk_level == "low":
                return "short_put"
            return "no_trade"

        if directional_bias == "bearish":
            if have_calls and iv_vs_rv_ok_short and event_risk_level == "low":
                return "short_call"
            # (Optional: long_put can be added if desired)
            return "no_trade"

        # Neutral bias
        if have_calls and have_puts:
            if (event_risk_level in ["medium", "high"]) and iv_vs_rv_ok_long:
                return "long_strangle"
            if iv_vs_rv_ok_short and event_risk_level == "low":
                return "short_strangle"
        else:
            # One-sided chain fallbacks
            if have_puts and iv_vs_rv_ok_short and event_risk_level == "low":
                return "short_put"
            if have_calls and iv_vs_rv_ok_short and event_risk_level == "low":
                return "short_call"
            if have_calls and iv_vs_rv_ok_long and event_risk_level in ["medium", "high"]:
                return "long_call"

        return "no_trade"

    strategy = choose_strategy(iv_rank_pct, iv_30d, blended_rv, event_risk_level, directional_bias, sides_present)

    result["decision_metrics"] = {
        "spot": spot,
        "iv_rank_pct": iv_rank_pct,
        "iv_30d": iv_30d,
        "rv_7d": rv_7d,
        "rv_14d": rv_14d,
        "blended_rv": blended_rv,
        "event_risk_level": event_risk_level,
        "directional_bias": directional_bias,
        "chosen": strategy,
    }
    result["sides_present"] = sorted(list(sides_present))

    if strategy == "no_trade":
        result["strategy"] = "no_trade"
        result["reason"] = "Decision layer selected no_trade"
        return result

    # --- Strike selection helpers ---
    def expected_move_band(row):
        """Return True if strike deviation fits expected move band when iv_30d is given."""
        if iv_30d is None:
            return True  # no constraint if IV30 not provided
        em = spot * iv_30d * np.sqrt(max(row["dte"], 1) / 365.0)  # expected move
        dev = abs(row["strike"] - spot)
        return (dev >= em * 1.0) and (dev <= em * 1.5)

    # --- Short strangle selection ---
    if strategy == "short_strangle":
        SS = {
            "delta_abs_min": 0.10,
            "delta_abs_max": 0.18,
            "net_position_delta_limit": 0.07,
            "min_net_credit_usd": 50.0,
        }
        calls = base[(base["side"] == "CALL") &
                     (base["delta"].between(SS["delta_abs_min"], SS["delta_abs_max"]))].copy()
        puts  = base[(base["side"] == "PUT") &
                     (base["delta"].between(-SS["delta_abs_max"], -SS["delta_abs_min"]))].copy()

        if iv_30d is not None:
            calls = calls[calls.apply(expected_move_band, axis=1)]
            puts  = puts[puts.apply(expected_move_band, axis=1)]

        if calls.empty or puts.empty:
            result["strategy"] = "no_trade"
            result["reason"] = "No legs in delta/EM band for short strangle"
            return result

        # Pair by same expiry, enforce net delta cap and score by credit and tight spreads
        pairs = []
        for exp in sorted(set(calls["expiry_date"]).intersection(set(puts["expiry_date"]))):
            c = calls[calls["expiry_date"] == exp]
            p = puts[puts["expiry_date"] == exp]
            if c.empty or p.empty:
                continue
            for _, rc in c.iterrows():
                pp = p.copy()
                pp["net_delta"] = (rc["delta"] + pp["delta"].values)
                candidates = pp[pp["net_delta"].abs() <= SS["net_position_delta_limit"]].copy()
                if candidates.empty:
                    continue
                candidates["credit"] = rc["mid"] + candidates["mid"]
                candidates["score"] = (
                    candidates["credit"]
                    - 0.1 * (rc["spread_pct"] + candidates["spread_pct"])
                    - 10.0 * candidates["net_delta"].abs()
                )
                best = candidates.sort_values("score", ascending=False).head(1)
                if not best.empty:
                    pairs.append((rc, best.iloc[0]))

        if not pairs:
            result["strategy"] = "no_trade"
            result["reason"] = "No call-put pairs satisfy net Δ cap"
            return result

        # Pick best pair with min credit
        best_pair = None
        best_score = -1e9
        for rc, rp in pairs:
            credit = rc["mid"] + rp["mid"]
            if credit < SS["min_net_credit_usd"]:
                continue
            score = credit - 0.1 * (rc["spread_pct"] + rp["spread_pct"])
            if score > best_score:
                best_score = score
                best_pair = (rc, rp, credit)

        if best_pair is None:
            result["strategy"] = "no_trade"
            result["reason"] = "No pair meets minimum net credit"
            return result

        rc, rp, credit = best_pair
        result["strategy"] = "short_strangle"
        result["expiry"] = str(rc["expiry_date"])
        result["legs"] = [
            {"action": "SELL", "type": "CALL", "symbol": rc["symbol"], "strike": float(rc["strike"]),
             "delta": float(rc["delta"]), "bid": float(rc["bid"]) if pd.notna(rc["bid"]) else None,
             "ask": float(rc["ask"]) if pd.notna(rc["ask"]) else None, "mid": float(rc["mid"])},
            {"action": "SELL", "type": "PUT",  "symbol": rp["symbol"], "strike": float(rp["strike"]),
             "delta": float(rp["delta"]), "bid": float(rp["bid"]) if pd.notna(rp["bid"]) else None,
             "ask": float(rp["ask"]) if pd.notna(rp["ask"]) else None, "mid": float(rp["mid"])}
        ]
        result["entry"] = {
            "order_type": "limit",
            "price_basis": "sum_of_mids",
            "net_credit_usd": float(round(credit, 2))
        }
        # OCO exits
        tp_debit = round(credit * 0.50, 2)
        sl_debit = round(credit * 3.00, 2)
        result["exits"] = {
            "take_profit_combo_buy_debit": tp_debit,
            "stop_loss_combo_buy_debit": sl_debit,
            "delta_stop_threshold": 0.20,
            "touch_exit": True,
            "time_exit_hours": 24
        }
        result["used_rows"] = pd.DataFrame([rc, rp])
        return result

    # --- Long strangle selection ---
    if strategy == "long_strangle":
        LS = {
            "delta_abs_min": 0.25,
            "delta_abs_max": 0.35,
        }
        calls = base[(base["side"] == "CALL") &
                     (base["delta"].between(LS["delta_abs_min"], LS["delta_abs_max"]))].copy()
        puts  = base[(base["side"] == "PUT") &
                     (base["delta"].between(-LS["delta_abs_max"], -LS["delta_abs_min"]))].copy()

        if calls.empty or puts.empty:
            result["strategy"] = "no_trade"
            result["reason"] = "No legs in delta band for long strangle"
            return result

        pairs = []
        for exp in sorted(set(calls["expiry_date"]).intersection(set(puts["expiry_date"]))):
            c = calls[calls["expiry_date"] == exp]
            p = puts[puts["expiry_date"] == exp]
            if c.empty or p.empty:
                continue
            for _, rc in c.iterrows():
                cand = p.copy()
                cand["debit"] = rc["mid"] + cand["mid"]
                cand["score"] = ((rc.get("vega", 0) + cand.get("vega", 0)) / cand["debit"].replace(0, np.nan))
                best = cand.sort_values("score", ascending=False).head(1)
                if not best.empty:
                    pairs.append((rc, best.iloc[0]))

        if not pairs:
            result["strategy"] = "no_trade"
            result["reason"] = "No call-put pairs for long strangle"
            return result

        scores = [((rc.get("vega", 0) + rp.get("vega", 0)) / max(rc["mid"] + rp["mid"], 1e-9))
                  for rc, rp in [(p[0], p[1]) for p in pairs]]
        idx = int(np.argmax(scores))
        rc, rp = pairs[idx]
        debit = float(round(rc["mid"] + rp["mid"], 2))

        result["strategy"] = "long_strangle"
        result["expiry"] = str(rc["expiry_date"])
        result["legs"] = [
            {"action": "BUY", "type": "CALL", "symbol": rc["symbol"], "strike": float(rc["strike"]),
             "delta": float(rc["delta"]), "bid": float(rc["bid"]) if pd.notna(rc["bid"]) else None,
             "ask": float(rc["ask"]) if pd.notna(rc["ask"]) else None, "mid": float(rc["mid"])},
            {"action": "BUY",  "type": "PUT",  "symbol": rp["symbol"], "strike": float(rp["strike"]),
             "delta": float(rp["delta"]), "bid": float(rp["bid"]) if pd.notna(rp["bid"]) else None,
             "ask": float(rp["ask"]) if pd.notna(rp["ask"]) else None, "mid": float(rp["mid"])}
        ]
        result["entry"] = {"order_type": "limit", "price_basis": "sum_of_mids", "net_debit_usd": debit}
        result["exits"] = {
            "take_profit_combo_sell_primary": round(debit * 1.50, 2),
            "take_profit_combo_sell_secondary": round(debit * 2.00, 2),
            "take_profit_combo_sell_runner": round(debit * 3.00, 2),
            "stop_loss_combo_sell": round(debit * 0.50, 2),
            "time_exit_hours": 24
        }
        result["used_rows"] = pd.DataFrame([rc, rp])
        return result

    # --- Long call selection ---
    if strategy == "long_call":
        LC = {"call_delta_min": 0.30, "call_delta_max": 0.45}
        calls = base[(base["side"] == "CALL") &
                     (base["delta"].between(LC["call_delta_min"], LC["call_delta_max"]))].copy()
        if calls.empty:
            result["strategy"] = "no_trade"
            result["reason"] = "No calls in delta band for long_call"
            return result

        # prefer high vega / $ and tight spread
        calls["score"] = (calls.get("vega", 0) / calls["mid"].replace(0, np.nan)) - 0.05 * calls["spread_pct"]
        rc = calls.sort_values("score", ascending=False).head(1).iloc[0]
        debit = float(round(rc["mid"], 2))

        result["strategy"] = "long_call"
        result["expiry"] = str(rc["expiry_date"])
        result["legs"] = [
            {"action": "BUY", "type": "CALL", "symbol": rc["symbol"], "strike": float(rc["strike"]),
             "delta": float(rc["delta"]), "bid": float(rc["bid"]) if pd.notna(rc["bid"]) else None,
             "ask": float(rc["ask"]) if pd.notna(rc["ask"]) else None, "mid": float(rc["mid"])}
        ]
        result["entry"] = {"order_type": "limit", "price_basis": "mid", "net_debit_usd": debit}
        result["exits"] = {
            "take_profit_sell_primary": round(debit * 1.50, 2),
            "take_profit_sell_secondary": round(debit * 2.00, 2),
            "stop_loss_sell": round(debit * 0.50, 2),
            "time_exit_hours": 24
        }
        result["used_rows"] = pd.DataFrame([rc])
        return result

    # --- Short put selection (one-sided fallback or bullish/neutral short-vol idea) ---
    if strategy == "short_put":
        SP = {
            "delta_abs_min": 0.12,
            "delta_abs_max": 0.25,
            "min_credit_usd": 10.0,
        }
        puts = base[
            (base["side"] == "PUT") &
            (base["delta"].between(-SP["delta_abs_max"], -SP["delta_abs_min"]))
        ].copy()

        if puts.empty:
            result["strategy"] = "no_trade"
            result["reason"] = "No puts in delta band for short_put"
            return result

        # Prefer higher credit, tighter spreads, healthier OI/volume
        puts["score"] = (
            puts["mid"]
            - 0.10 * puts["spread_pct"]
            + 0.005 * puts["oi"]
            + 0.01 * puts["volume"]
        )
        rp = puts.sort_values("score", ascending=False).head(1).iloc[0]
        credit = float(round(rp["mid"], 2))
        if credit < SP["min_credit_usd"]:
            result["strategy"] = "no_trade"
            result["reason"] = "short_put credit below minimum"
            return result

        result["strategy"] = "short_put"
        result["expiry"] = str(rp["expiry_date"])
        result["legs"] = [{
            "action": "SELL", "type": "PUT", "symbol": rp["symbol"], "strike": float(rp["strike"]),
            "delta": float(rp["delta"]),
            "bid": float(rp["bid"]) if pd.notna(rp["bid"]) else None,
            "ask": float(rp["ask"]) if pd.notna(rp["ask"]) else None,
            "mid": float(rp["mid"])
        }]
        result["entry"] = {
            "order_type": "limit",
            "price_basis": "mid",
            "net_credit_usd": credit
        }
        result["exits"] = {
            "take_profit_buy_debit": round(credit * 0.50, 2),
            "stop_loss_buy_debit": round(credit * 3.00, 2),
            "delta_stop_threshold": 0.30,
            "time_exit_hours": 24
        }
        result["used_rows"] = pd.DataFrame([rp])
        return result

    # --- Short call selection (one-sided fallback or bearish/neutral short-vol idea) ---
    if strategy == "short_call":
        SC = {
            "delta_abs_min": 0.12,
            "delta_abs_max": 0.25,
            "min_credit_usd": 10.0,
        }
        calls = base[
            (base["side"] == "CALL") &
            (base["delta"].between(SC["delta_abs_min"], SC["delta_abs_max"]))
        ].copy()

        if calls.empty:
            result["strategy"] = "no_trade"
            result["reason"] = "No calls in delta band for short_call"
            return result

        calls["score"] = (
            calls["mid"]
            - 0.10 * calls["spread_pct"]
            + 0.005 * calls["oi"]
            + 0.01 * calls["volume"]
        )
        rc = calls.sort_values("score", ascending=False).head(1).iloc[0]
        credit = float(round(rc["mid"], 2))
        if credit < SC["min_credit_usd"]:
            result["strategy"] = "no_trade"
            result["reason"] = "short_call credit below minimum"
            return result

        result["strategy"] = "short_call"
        result["expiry"] = str(rc["expiry_date"])
        result["legs"] = [{
            "action": "SELL", "type": "CALL", "symbol": rc["symbol"], "strike": float(rc["strike"]),
            "delta": float(rc["delta"]),
            "bid": float(rc["bid"]) if pd.notna(rc["bid"]) else None,
            "ask": float(rc["ask"]) if pd.notna(rc["ask"]) else None,
            "mid": float(rc["mid"])
        }]
        result["entry"] = {
            "order_type": "limit",
            "price_basis": "mid",
            "net_credit_usd": credit
        }
        result["exits"] = {
            "take_profit_buy_debit": round(credit * 0.50, 2),
            "stop_loss_buy_debit": round(credit * 3.00, 2),
            "delta_stop_threshold": 0.30,
            "time_exit_hours": 24
        }
        result["used_rows"] = pd.DataFrame([rc])
        return result

    # --- Fallback ---
    result["strategy"] = "no_trade"
    result["reason"] = "Unhandled branch"
    return result

    
# -----------------------------
# Email sending (trade ticket + snapshot table)
# -----------------------------
def send_email_report(df: pd.DataFrame):
    smtp_email = os.environ.get("SMTP_EMAIL")
    smtp_password = os.environ.get("SMTP_PASSWORD")
    if not smtp_email or not smtp_password:
        log.error("SMTP credentials not found.")
        return

    # --- get live directional bias from spot history ---
    candles = fetch_spot_history(symbol="ETHUSD", resolution="1h", lookback_hours=72)
    bias = detect_directional_bias(candles)

    # --- strategy selection ---
    trade = select_strangles(
        df,
        iv_rank_pct=None,
        iv_30d=None,
        rv_7d=None,
        rv_14d=None,
        event_risk_level="low",
        directional_bias=bias   # ✅ inject external bias here
    )

    # --- Filtered snapshot table (debug/market context) ---
    snapshot = filter_options(df)
    snapshot_html = ""
    if not snapshot.empty:
        snapshot_html = snapshot[[
            "symbol", "side", "strike", "expiry_date", "bid", "ask", "mid",
            "iv", "delta", "volume", "oi", "spot"
        ]].to_html(index=False, justify="center",
                   float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x)

    # --- Trade ticket ---
    if trade.get("strategy") == "no_trade":
        subject = f"ETH Options – No Trade ({datetime.utcnow().strftime('%Y-%m-%d')})"
        body = f"""
        <html><body>
        <h3>No Trade Selected</h3>
        <p>Reason: {trade.get('reason','')}</p>
        <pre>{trade.get('decision_metrics')}</pre>
        <h3>Filtered Market Snapshot</h3>
        {snapshot_html}
        </body></html>
        """
    else:
        used = trade.get("used_rows", pd.DataFrame())
        legs_html = ""
        if not used.empty:
            legs_html = used[[
                "symbol", "side", "strike", "expiry_date", "bid", "ask", "mid",
                "delta", "volume", "oi", "spread_pct"
            ]].to_html(index=False, justify="center",
                       float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x)

        dm = trade.get("decision_metrics", {})
        subject = f"ETH {trade['strategy'].replace('_',' ').title()} – {trade.get('expiry','')} ({datetime.utcnow().strftime('%Y-%m-%d')})"
        body = f"""
        <html><body>
          <h2>🎯 Strategy: {trade['strategy'].replace('_',' ').title()}</h2>
          <p><b>Spot</b>: {dm.get('spot')}</p>
          <p><b>Decision Inputs</b>:
             IV Rank %={dm.get('iv_rank_pct')}, IV30={dm.get('iv_30d')}, RV7={dm.get('rv_7d')}, RV14={dm.get('rv_14d')},
             Event={dm.get('event_risk_level')}, Bias={dm.get('directional_bias')}</p>
          <h3>Legs</h3>
          {legs_html}
          <h3>Entry</h3>
          <pre>{trade.get('entry')}</pre>
          <h3>Exits / Risk</h3>
          <pre>{trade.get('exits')}</pre>
          <h3>Filtered Market Snapshot</h3>
          {snapshot_html}
        </body></html>
        """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
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




