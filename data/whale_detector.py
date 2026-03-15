"""
Whale detector — track unusual options activity, volume anomalies, and big money flow.

Scans for:
    - Unusual options volume (volume >> open interest)
    - Large premium sweeps (high dollar value single-leg trades)
    - Put/call ratio extremes
    - Stock volume spikes vs 20-day average
    - Dark pool prints (large block trades inferred from volume patterns)

Public API:
    detect_whales(ticker) -> dict
    scan_whales(tickers)  -> list[dict]

All data sourced from yfinance (free). Cached 10 minutes per ticker.
"""

import logging
import time
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)

_cache: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 600  # 10 minutes


# ── Options flow analysis ────────────────────────────────────────────────────

def _analyze_options_flow(ticker: str) -> dict:
    """
    Scan all near-term expirations for unusual options activity.

    Flags:
        - Contracts where volume > 3x open interest (unusual flow)
        - High-dollar sweeps (volume * mid_price * 100 > $100K)
        - Extreme put/call volume ratios
    """
    result = {
        "unusual_contracts": [],
        "total_call_volume": 0,
        "total_put_volume": 0,
        "put_call_ratio": None,
        "pc_signal": "Neutral",
        "biggest_sweep": None,
        "flow_direction": "Neutral",
        "flow_score": 0,
    }

    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            return result

        # Scan first 3 expirations (near-term = most informative)
        scan_exps = expirations[:3]

        all_unusual = []
        total_call_vol = 0
        total_put_vol = 0
        total_call_premium = 0.0
        total_put_premium = 0.0

        for exp in scan_exps:
            try:
                chain = t.option_chain(exp)
            except Exception:
                continue

            for side, df in [("CALL", chain.calls), ("PUT", chain.puts)]:
                if df is None or df.empty:
                    continue

                for _, row in df.iterrows():
                    raw_vol = row.get("volume")
                    raw_oi = row.get("openInterest")
                    vol = int(raw_vol) if pd.notna(raw_vol) else 0
                    oi = int(raw_oi) if pd.notna(raw_oi) else 0
                    bid = float(row.get("bid") or 0)
                    ask = float(row.get("ask") or 0)
                    strike = float(row.get("strike") or 0)
                    iv = float(row.get("impliedVolatility") or 0)
                    itm = bool(row.get("inTheMoney", False))

                    if vol == 0:
                        continue

                    mid_price = (bid + ask) / 2 if (bid + ask) > 0 else float(row.get("lastPrice") or 0)
                    dollar_value = vol * mid_price * 100  # each contract = 100 shares

                    # Skip deep ITM options — mostly intrinsic value, not speculative flow.
                    # Deep ITM: premium > 20% of strike = the option is mostly intrinsic value,
                    # not a speculative bet. These inflate sweep/premium numbers unrealistically.
                    deep_itm = False
                    if itm and strike > 0 and mid_price > strike * 0.20:
                        deep_itm = True

                    if side == "CALL":
                        total_call_vol += vol
                        if not deep_itm:
                            total_call_premium += dollar_value
                    else:
                        total_put_vol += vol
                        if not deep_itm:
                            total_put_premium += dollar_value

                    # Skip deep ITM for unusual/sweep detection
                    if deep_itm:
                        continue

                    # Flag unusual: volume > 3x open interest, or volume > 1000 with no prior OI
                    is_unusual = False
                    if oi > 0 and vol > oi * 3:
                        is_unusual = True
                    elif oi == 0 and vol > 500:
                        is_unusual = True

                    # Flag big sweeps: > $100K notional
                    is_sweep = dollar_value > 100_000

                    if is_unusual or is_sweep:
                        contract = {
                            "expiration": exp,
                            "side": side,
                            "strike": strike,
                            "volume": vol,
                            "open_interest": oi,
                            "vol_oi_ratio": round(vol / oi, 1) if oi > 0 else None,
                            "mid_price": round(mid_price, 2),
                            "dollar_value": round(dollar_value),
                            "iv": round(iv * 100, 1),
                            "itm": itm,
                            "is_sweep": is_sweep,
                            "is_unusual": is_unusual,
                        }
                        all_unusual.append(contract)

        # Sort by dollar value descending
        all_unusual.sort(key=lambda x: x["dollar_value"], reverse=True)

        # Put/call ratio
        pc_ratio = None
        pc_signal = "Neutral"
        if total_call_vol > 0:
            pc_ratio = round(total_put_vol / total_call_vol, 2)
            if pc_ratio > 1.3:
                pc_signal = "Bearish"  # heavy put buying
            elif pc_ratio < 0.7:
                pc_signal = "Bullish"  # heavy call buying
            elif pc_ratio > 1.0:
                pc_signal = "Slightly Bearish"
            elif pc_ratio < 0.85:
                pc_signal = "Slightly Bullish"

        # Flow direction based on premium
        total_premium = total_call_premium + total_put_premium
        if total_premium > 0:
            call_pct = total_call_premium / total_premium
            if call_pct > 0.55:
                flow_dir = "Bullish"
                flow_score = min(round(call_pct * 100), 100)
            elif call_pct < 0.45:
                flow_dir = "Bearish"
                flow_score = -min(round((1 - call_pct) * 100), 100)
            else:
                flow_dir = "Neutral"
                flow_score = 0
        else:
            flow_dir = "Neutral"
            flow_score = 0

        result.update({
            "unusual_contracts": all_unusual[:15],  # top 15
            "total_call_volume": total_call_vol,
            "total_put_volume": total_put_vol,
            "total_call_premium": round(total_call_premium),
            "total_put_premium": round(total_put_premium),
            "put_call_ratio": pc_ratio,
            "pc_signal": pc_signal,
            "biggest_sweep": all_unusual[0] if all_unusual else None,
            "flow_direction": flow_dir,
            "flow_score": flow_score,
        })

    except Exception as exc:
        logger.debug("Options flow analysis failed for %s: %s", ticker, exc)

    return result


# ── Volume anomaly detection ─────────────────────────────────────────────────

def _analyze_volume(ticker: str) -> dict:
    """
    Detect unusual stock volume patterns:
        - Volume spike vs 20-day average
        - Intraday block trades (large single-bar volume)
        - Accumulation/distribution signal
    """
    result = {
        "volume_ratio": 1.0,
        "volume_signal": "Normal",
        "block_detected": False,
        "accumulation": "Neutral",
        "avg_volume": 0,
        "current_volume": 0,
    }

    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1mo", interval="1d")
        if hist is None or len(hist) < 5:
            return result

        volumes = hist["Volume"].values
        closes = hist["Close"].values
        opens = hist["Open"].values

        current_vol = int(volumes[-1])
        avg_vol = int(volumes[:-1].mean()) if len(volumes) > 1 else int(volumes[0])
        ratio = round(current_vol / avg_vol, 2) if avg_vol > 0 else 1.0

        # Volume signal
        if ratio >= 5.0:
            vol_signal = "Extreme"
        elif ratio >= 3.0:
            vol_signal = "Very High"
        elif ratio >= 2.0:
            vol_signal = "High"
        elif ratio >= 1.5:
            vol_signal = "Elevated"
        else:
            vol_signal = "Normal"

        # Block trade detection: any single day > 5x average in last 5 days
        block = any(v > avg_vol * 5 for v in volumes[-5:])

        # Simple accumulation/distribution: price up on high volume = accumulation
        if len(closes) >= 2:
            price_change = closes[-1] - closes[-2]
            if ratio >= 1.5 and price_change > 0:
                accum = "Accumulation"
            elif ratio >= 1.5 and price_change < 0:
                accum = "Distribution"
            else:
                accum = "Neutral"
        else:
            accum = "Neutral"

        result.update({
            "volume_ratio": ratio,
            "volume_signal": vol_signal,
            "block_detected": block,
            "accumulation": accum,
            "avg_volume": avg_vol,
            "current_volume": current_vol,
        })

    except Exception as exc:
        logger.debug("Volume analysis failed for %s: %s", ticker, exc)

    return result


# ── Dark pool / institutional inference ──────────────────────────────────────

def _infer_institutional(ticker: str) -> dict:
    """
    Infer institutional activity from public data:
        - Large holder changes (yfinance major_holders / institutional_holders)
        - Recent 13F filings activity
    """
    result = {
        "institutional_pct": None,
        "insider_pct": None,
        "recent_institutional_change": "Unknown",
    }

    try:
        t = yf.Ticker(ticker)

        # Major holders
        mh = getattr(t, "major_holders", None)
        if mh is not None and isinstance(mh, pd.DataFrame) and not mh.empty:
            for _, row in mh.iterrows():
                label = str(row.iloc[-1]).lower() if len(row) > 1 else ""
                value = str(row.iloc[0]).replace("%", "").strip()
                try:
                    pct = float(value)
                except ValueError:
                    continue
                if "insider" in label:
                    result["insider_pct"] = pct
                elif "institution" in label:
                    result["institutional_pct"] = pct

        # Institutional holders — check for recent buys/sells
        ih = getattr(t, "institutional_holders", None)
        if ih is not None and isinstance(ih, pd.DataFrame) and not ih.empty:
            cutoff = datetime.now() - timedelta(days=90)
            recent_count = 0
            for _, row in ih.head(10).iterrows():
                date_held = row.get("Date Reported")
                if hasattr(date_held, "date"):
                    if date_held.date() >= cutoff.date():
                        recent_count += 1
            if recent_count >= 3:
                result["recent_institutional_change"] = "Active"
            elif recent_count >= 1:
                result["recent_institutional_change"] = "Some"
            else:
                result["recent_institutional_change"] = "Quiet"

    except Exception as exc:
        logger.debug("Institutional inference failed for %s: %s", ticker, exc)

    return result


# ── Squeeze detection ────────────────────────────────────────────────────────

def _detect_squeeze(ticker: str, options_flow: dict) -> dict:
    """
    Detect potential gamma squeeze and short squeeze setups.

    Gamma squeeze: massive call OI near current price + rising call volume
        = market makers forced to buy shares to hedge = price rockets up.

    Short squeeze: high short interest + price rising + high volume
        = shorts forced to cover = rapid price increase.
    """
    result = {
        "gamma_squeeze_risk": "None",
        "short_squeeze_risk": "None",
        "gamma_score": 0,
        "short_score": 0,
        "squeeze_signals": [],
    }

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        price = info.get("currentPrice") or info.get("regularMarketPrice") or 0

        # ── Short squeeze detection ──────────────────────────────────────────
        short_pct = info.get("shortPercentOfFloat") or 0
        short_ratio = info.get("shortRatio") or 0  # days to cover

        short_score = 0
        if short_pct > 0.20:  # > 20% short interest
            short_score += 40
            result["squeeze_signals"].append(
                f"Short interest {short_pct*100:.1f}% of float — very high"
            )
        elif short_pct > 0.10:
            short_score += 20
            result["squeeze_signals"].append(
                f"Short interest {short_pct*100:.1f}% of float — elevated"
            )

        if short_ratio > 5:  # > 5 days to cover
            short_score += 25
            result["squeeze_signals"].append(
                f"Days to cover: {short_ratio:.1f} — shorts would struggle to exit"
            )
        elif short_ratio > 3:
            short_score += 10
            result["squeeze_signals"].append(
                f"Days to cover: {short_ratio:.1f}"
            )

        # Price momentum + volume = squeeze pressure
        hist = t.history(period="10d", interval="1d")
        if hist is not None and len(hist) >= 5:
            closes = hist["Close"].values
            volumes = hist["Volume"].values
            avg_vol = volumes[:-1].mean() if len(volumes) > 1 else volumes[0]
            cur_vol = volumes[-1]

            # 5-day return
            five_day_ret = (closes[-1] - closes[-5]) / closes[-5] * 100 if closes[-5] > 0 else 0
            if five_day_ret > 5 and short_pct > 0.10:
                short_score += 15
                result["squeeze_signals"].append(
                    f"Price up {five_day_ret:.1f}% in 5 days while heavily shorted"
                )

            if avg_vol > 0 and cur_vol > avg_vol * 2 and short_pct > 0.10:
                short_score += 10
                result["squeeze_signals"].append(
                    "Volume surge with high short interest — potential covering"
                )

        short_score = min(short_score, 100)
        if short_score >= 60:
            result["short_squeeze_risk"] = "High"
        elif short_score >= 35:
            result["short_squeeze_risk"] = "Moderate"
        elif short_score >= 15:
            result["short_squeeze_risk"] = "Low"
        result["short_score"] = short_score

        # ── Gamma squeeze detection ──────────────────────────────────────────
        # Look for massive call OI clustering near current price
        gamma_score = 0
        unusual = options_flow.get("unusual_contracts", [])
        call_oi_near_price = 0
        total_call_sweep_value = 0

        exps = t.options[:2] if t.options else []
        for exp in exps:
            try:
                chain = t.option_chain(exp)
                calls = chain.calls
                if calls is None or calls.empty or price <= 0:
                    continue

                # OI within 5% of current price
                near = calls[
                    (calls["strike"] >= price * 0.95) &
                    (calls["strike"] <= price * 1.05)
                ]
                if not near.empty:
                    oi_sum = near["openInterest"].fillna(0).sum()
                    call_oi_near_price += int(oi_sum)
            except Exception:
                continue

        # Large call OI near price = gamma wall
        if call_oi_near_price > 50_000:
            gamma_score += 40
            result["squeeze_signals"].append(
                f"Gamma wall: {call_oi_near_price:,} call OI within 5% of price — "
                "market makers must hedge aggressively"
            )
        elif call_oi_near_price > 20_000:
            gamma_score += 20
            result["squeeze_signals"].append(
                f"Elevated call OI near price: {call_oi_near_price:,} contracts"
            )

        # Large call sweeps = someone positioning for gamma
        call_sweeps = [c for c in unusual if c["side"] == "CALL" and c.get("is_sweep")]
        for cs in call_sweeps[:3]:
            total_call_sweep_value += cs["dollar_value"]

        if total_call_sweep_value > 10_000_000:  # > $10M in call sweeps
            gamma_score += 30
            result["squeeze_signals"].append(
                f"${total_call_sweep_value/1e6:.0f}M in call sweeps — aggressive positioning"
            )
        elif total_call_sweep_value > 1_000_000:
            gamma_score += 15
            result["squeeze_signals"].append(
                f"${total_call_sweep_value/1e6:.1f}M in call sweeps"
            )

        # Call/put ratio extreme = gamma pressure
        pc = options_flow.get("put_call_ratio")
        if pc is not None and pc < 0.5:
            gamma_score += 15
            result["squeeze_signals"].append(
                "Extreme call dominance (P/C < 0.5) — gamma acceleration risk"
            )

        gamma_score = min(gamma_score, 100)
        if gamma_score >= 60:
            result["gamma_squeeze_risk"] = "High"
        elif gamma_score >= 30:
            result["gamma_squeeze_risk"] = "Moderate"
        elif gamma_score >= 10:
            result["gamma_squeeze_risk"] = "Low"
        result["gamma_score"] = gamma_score

    except Exception as exc:
        logger.debug("Squeeze detection failed for %s: %s", ticker, exc)

    return result


# ── Public API ───────────────────────────────────────────────────────────────

def detect_whales(ticker: str) -> dict:
    """
    Full whale detection scan for a single ticker.

    Returns:
        {
            "ticker": str,
            "options_flow": dict,       # unusual contracts, P/C ratio, sweeps
            "volume": dict,             # volume spikes, blocks, accumulation
            "institutional": dict,      # holder data, recent changes
            "whale_score": int,         # -100 to +100 composite
            "whale_direction": str,     # Bullish / Bearish / Neutral
            "whale_signals": list[str], # plain English alerts
            "alert_level": str,         # "None" / "Watch" / "Alert" / "Whale Alert"
        }
    """
    ticker = ticker.upper().strip()

    # Check cache
    now = time.time()
    if ticker in _cache:
        ts, cached = _cache[ticker]
        if now - ts < _CACHE_TTL:
            return cached

    options = _analyze_options_flow(ticker)
    volume = _analyze_volume(ticker)
    institutional = _infer_institutional(ticker)
    squeeze = _detect_squeeze(ticker, options)

    # Build composite score and signals
    signals = []
    score = 0

    # Options flow signals
    flow_dir = options.get("flow_direction", "Neutral")
    if flow_dir == "Bullish":
        score += 25
        signals.append(f"Bullish options flow — {options['total_call_volume']:,} calls vs {options['total_put_volume']:,} puts")
    elif flow_dir == "Bearish":
        score -= 25
        signals.append(f"Bearish options flow — {options['total_put_volume']:,} puts vs {options['total_call_volume']:,} calls")

    # Unusual contracts
    n_unusual = len([c for c in options.get("unusual_contracts", []) if c.get("is_unusual")])
    if n_unusual >= 5:
        score += 20 if flow_dir == "Bullish" else -20
        signals.append(f"{n_unusual} contracts with unusual volume/OI ratio")
    elif n_unusual >= 2:
        score += 10 if flow_dir == "Bullish" else -10
        signals.append(f"{n_unusual} contracts with unusual activity")

    # Big sweeps
    sweeps = [c for c in options.get("unusual_contracts", []) if c.get("is_sweep")]
    if sweeps:
        biggest = sweeps[0]
        score += 15 if biggest["side"] == "CALL" else -15
        signals.append(f"${biggest['dollar_value']:,} sweep on {biggest['strike']} {biggest['side']} exp {biggest['expiration']}")

    # Put/call ratio
    pc = options.get("put_call_ratio")
    if pc is not None:
        if pc > 2.0:
            score -= 15
            signals.append(f"Put/Call ratio {pc:.2f} — heavy put activity (hedging or bearish)")
        elif pc < 0.3:
            score += 15
            signals.append(f"Put/Call ratio {pc:.2f} — heavy call activity (bullish conviction)")

    # Volume signals
    vol_ratio = volume.get("volume_ratio", 1.0)
    if vol_ratio >= 3.0:
        accum = volume.get("accumulation", "Neutral")
        if accum == "Accumulation":
            score += 15
            signals.append(f"Volume {vol_ratio:.1f}x average with price UP — accumulation pattern")
        elif accum == "Distribution":
            score -= 15
            signals.append(f"Volume {vol_ratio:.1f}x average with price DOWN — distribution pattern")
        else:
            signals.append(f"Volume spike: {vol_ratio:.1f}x average — watch for direction")
    elif vol_ratio >= 2.0:
        signals.append(f"Elevated volume: {vol_ratio:.1f}x average")

    if volume.get("block_detected"):
        score += 5
        signals.append("Block trade detected in recent sessions")

    # Institutional
    inst_pct = institutional.get("institutional_pct")
    if inst_pct and inst_pct > 80:
        signals.append(f"Institutional ownership: {inst_pct:.0f}% — heavily held")
    inst_change = institutional.get("recent_institutional_change", "Unknown")
    if inst_change == "Active":
        score += 5
        signals.append("Active institutional filing changes in last 90 days")

    # Squeeze signals
    if squeeze["gamma_squeeze_risk"] in ("High", "Moderate"):
        score += squeeze["gamma_score"] // 3
        signals.append(f"GAMMA SQUEEZE risk: {squeeze['gamma_squeeze_risk']}")
    if squeeze["short_squeeze_risk"] in ("High", "Moderate"):
        score += squeeze["short_score"] // 3
        signals.append(f"SHORT SQUEEZE risk: {squeeze['short_squeeze_risk']}")
    for ss in squeeze["squeeze_signals"]:
        signals.append(f"  -> {ss}")

    # Clamp score
    score = max(-100, min(100, score))

    # Direction and alert level
    if score > 25:
        direction = "Bullish"
    elif score < -25:
        direction = "Bearish"
    else:
        direction = "Neutral"

    abs_score = abs(score)
    if abs_score >= 60:
        alert = "Whale Alert"
    elif abs_score >= 40:
        alert = "Alert"
    elif abs_score >= 20:
        alert = "Watch"
    else:
        alert = "None"

    result = {
        "ticker": ticker,
        "options_flow": options,
        "volume": volume,
        "institutional": institutional,
        "squeeze": squeeze,
        "whale_score": score,
        "whale_direction": direction,
        "whale_signals": signals,
        "alert_level": alert,
    }

    _cache[ticker] = (now, result)
    return result


def scan_whales(tickers: list[str]) -> list[dict]:
    """
    Scan multiple tickers for whale activity.
    Returns only tickers with signals, sorted by |whale_score| descending.
    """
    results = []
    for ticker in tickers:
        try:
            result = detect_whales(ticker)
            if result["whale_signals"]:  # only include tickers with activity
                results.append(result)
        except Exception as exc:
            logger.debug("Whale scan failed for %s: %s", ticker, exc)
            continue

    results.sort(key=lambda r: abs(r["whale_score"]), reverse=True)
    return results
