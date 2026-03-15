"""
Fundamental analysis layer — valuation, growth, financial health,
institutional flow, and short squeeze detection.

Public API:
    get_fundamentals(ticker: str) -> dict
    score_fundamentals(fund_dict: dict) -> dict
    get_short_squeeze_score(fund_dict: dict) -> dict
"""

import logging

import yfinance as yf

logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def get_fundamentals(ticker: str) -> dict:
    """
    Fetch fundamental data for a ticker from yfinance.
    All sub-calls are individually wrapped in try/except — never raises.

    Returns a dict with all fields; value is None if unavailable.
    """
    ticker = ticker.upper()
    result = {
        "ticker":             ticker,
        "pe_ratio":           None,
        "forward_pe":         None,
        "peg_ratio":          None,
        "ps_ratio":           None,
        "pb_ratio":           None,
        "ev_ebitda":          None,
        "revenue_growth":     None,
        "earnings_growth":    None,
        "debt_to_equity":     None,
        "current_ratio":      None,
        "free_cashflow":      None,
        "profit_margins":     None,
        "roe":                None,
        "inst_ownership_pct": None,
        "short_ratio":        None,
        "short_pct_float":    None,
        "recommendation":     None,
        "target_price":       None,
        "earnings_beat_rate": None,
        "recent_upgrade":     False,
        "recent_downgrade":   False,
        "insider_buy_recent": False,
        "insider_sell_recent": False,
        "market_cap":         None,
        "sector":             None,
        "industry":           None,
        "error":              None,
    }

    try:
        t = yf.Ticker(ticker)
    except Exception as exc:
        result["error"] = str(exc)
        return result

    # ── .info — all fields in one call ────────────────────────────────────────
    try:
        info = t.info or {}
        result["pe_ratio"]           = _safe(info.get("trailingPE"))
        result["forward_pe"]         = _safe(info.get("forwardPE"))
        result["peg_ratio"]          = _safe(info.get("pegRatio"))
        result["ps_ratio"]           = _safe(info.get("priceToSalesTrailingTwelveMonths"))
        result["pb_ratio"]           = _safe(info.get("priceToBook"))
        result["ev_ebitda"]          = _safe(info.get("enterpriseToEbitda"))
        result["revenue_growth"]     = _safe(info.get("revenueGrowth"))
        result["earnings_growth"]    = _safe(info.get("earningsGrowth"))
        result["debt_to_equity"]     = _safe(info.get("debtToEquity"))
        result["current_ratio"]      = _safe(info.get("currentRatio"))
        result["free_cashflow"]      = _safe(info.get("freeCashflow"))
        result["profit_margins"]     = _safe(info.get("profitMargins"))
        result["roe"]                = _safe(info.get("returnOnEquity"))
        result["inst_ownership_pct"] = _safe(info.get("heldPercentInstitutions"))
        result["short_ratio"]        = _safe(info.get("shortRatio"))
        result["short_pct_float"]    = _safe(info.get("shortPercentOfFloat"))
        result["recommendation"]     = info.get("recommendationKey")
        result["target_price"]       = _safe(info.get("targetMeanPrice"))
        result["market_cap"]         = _safe(info.get("marketCap"))
        result["sector"]             = info.get("sector")
        result["industry"]           = info.get("industry")
    except Exception as exc:
        logger.debug("get_fundamentals info failed for %s: %s", ticker, exc)

    # ── Earnings history — beat rate ───────────────────────────────────────────
    try:
        eh = t.earnings_history
        if eh is not None and not eh.empty:
            # Look for epsEstimate and epsActual columns
            est_col = None
            act_col = None
            for c in eh.columns:
                cl = c.lower()
                if "estimate" in cl or "epsestimate" in cl:
                    est_col = c
                if "actual" in cl or "epsactual" in cl:
                    act_col = c
            if est_col and act_col:
                last4 = eh.tail(4)
                valid = last4.dropna(subset=[est_col, act_col])
                if len(valid) > 0:
                    beats = (valid[act_col] > valid[est_col]).sum()
                    result["earnings_beat_rate"] = round(float(beats) / len(valid), 2)
    except Exception as exc:
        logger.debug("get_fundamentals earnings_history failed for %s: %s", ticker, exc)

    # ── Upgrades / downgrades — most recent action ────────────────────────────
    try:
        ud = t.upgrades_downgrades
        if ud is not None and not ud.empty:
            # Sort by date descending and look at last 90 days
            import pandas as pd
            ud_sorted = ud.sort_index(ascending=False)
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=90)
            recent = ud_sorted[ud_sorted.index >= cutoff] if ud_sorted.index.tz else ud_sorted.head(5)
            if not recent.empty:
                action_col = None
                for c in recent.columns:
                    if "action" in c.lower() or "tograde" in c.lower() or "grade" in c.lower():
                        action_col = c
                        break
                if action_col:
                    actions = recent[action_col].astype(str).str.lower()
                    result["recent_upgrade"]   = any("upgrade" in a or "buy" in a or "outperform" in a for a in actions)
                    result["recent_downgrade"] = any("downgrade" in a or "sell" in a or "underperform" in a for a in actions)
    except Exception as exc:
        logger.debug("get_fundamentals upgrades_downgrades failed for %s: %s", ticker, exc)

    # ── Insider transactions — most recent buy/sell ────────────────────────────
    try:
        ins = t.insider_transactions
        if ins is not None and not ins.empty:
            import pandas as pd
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
            # normalize dates
            date_col = None
            for c in ins.columns:
                if "date" in c.lower() or "start" in c.lower():
                    date_col = c
                    break
            text_col = None
            for c in ins.columns:
                if "text" in c.lower() or "transaction" in c.lower() or "type" in c.lower():
                    text_col = c
                    break
            if date_col:
                try:
                    dates = pd.to_datetime(ins[date_col], errors="coerce")
                    recent_mask = dates >= cutoff
                    recent_ins = ins[recent_mask]
                except Exception:
                    recent_ins = ins.head(5)
            else:
                recent_ins = ins.head(5)

            if not recent_ins.empty and text_col:
                txn_text = recent_ins[text_col].astype(str).str.lower()
                result["insider_buy_recent"]  = any("buy" in t or "purchase" in t for t in txn_text)
                result["insider_sell_recent"] = any("sell" in t or "sale" in t for t in txn_text)
    except Exception as exc:
        logger.debug("get_fundamentals insider_transactions failed for %s: %s", ticker, exc)

    return result


def score_fundamentals(fund_dict: dict) -> dict:
    """
    Score a fund_dict (from get_fundamentals) on a -100 to +100 scale.

    Returns:
        {
            "fundamental_score":   int,       # -100 to +100
            "fundamental_signal":  str,       # "Bullish", "Bearish", "Neutral"
            "fundamental_reasons": list[str], # up to 5 signal labels triggered
        }
    """
    score   = 0
    reasons = []

    # ── Valuation signals (max ±30) ──────────────────────────────────────────
    peg = fund_dict.get("peg_ratio")
    if peg is not None:
        if peg < 1.0:
            score += 10; reasons.append(f"Low PEG ({peg:.1f})")
        elif peg > 3.0:
            score -= 10; reasons.append(f"High PEG ({peg:.1f})")

    fwd_pe = fund_dict.get("forward_pe")
    if fwd_pe is not None:
        if fwd_pe < 15:
            score += 8; reasons.append(f"Low Fwd P/E ({fwd_pe:.1f})")
        elif fwd_pe > 40:
            score -= 8; reasons.append(f"High Fwd P/E ({fwd_pe:.1f})")

    ev_ebitda = fund_dict.get("ev_ebitda")
    if ev_ebitda is not None:
        if ev_ebitda < 10:
            score += 7; reasons.append(f"Low EV/EBITDA ({ev_ebitda:.1f})")
        elif ev_ebitda > 25:
            score -= 7; reasons.append(f"High EV/EBITDA ({ev_ebitda:.1f})")

    pb = fund_dict.get("pb_ratio")
    if pb is not None:
        if pb < 1.5:
            score += 5; reasons.append(f"Low P/B ({pb:.1f})")
        elif pb > 5:
            score -= 5; reasons.append(f"High P/B ({pb:.1f})")

    # ── Growth signals (max ±25) ─────────────────────────────────────────────
    rev_growth = fund_dict.get("revenue_growth")
    if rev_growth is not None:
        pct = rev_growth * 100
        if pct > 20:
            score += 12; reasons.append(f"Revenue Growth +{pct:.0f}%")
        elif pct < 0:
            score -= 12; reasons.append(f"Revenue Decline {pct:.0f}%")

    earn_growth = fund_dict.get("earnings_growth")
    if earn_growth is not None:
        pct = earn_growth * 100
        if pct > 20:
            score += 10; reasons.append(f"Earnings Growth +{pct:.0f}%")
        elif pct < 0:
            score -= 10; reasons.append(f"Earnings Decline {pct:.0f}%")

    beat_rate = fund_dict.get("earnings_beat_rate")
    if beat_rate is not None:
        if beat_rate >= 0.75:
            score += 3; reasons.append(f"EPS Beat Rate {beat_rate*100:.0f}%")
        elif beat_rate <= 0.25:
            score -= 3; reasons.append(f"EPS Miss Rate {(1-beat_rate)*100:.0f}%")

    # ── Financial health (max ±25) ───────────────────────────────────────────
    cur_ratio = fund_dict.get("current_ratio")
    if cur_ratio is not None:
        if cur_ratio > 2.0:
            score += 8; reasons.append(f"Strong Current Ratio ({cur_ratio:.1f})")
        elif cur_ratio < 1.0:
            score -= 8; reasons.append(f"Weak Current Ratio ({cur_ratio:.1f})")

    dte = fund_dict.get("debt_to_equity")
    if dte is not None:
        if dte < 0.5:
            score += 7; reasons.append(f"Low D/E ({dte:.2f})")
        elif dte > 2.0:
            score -= 7; reasons.append(f"High D/E ({dte:.2f})")

    roe = fund_dict.get("roe")
    if roe is not None:
        pct = roe * 100
        if pct > 15:
            score += 6; reasons.append(f"High ROE ({pct:.0f}%)")
        elif pct < 0:
            score -= 6; reasons.append(f"Negative ROE ({pct:.0f}%)")

    fcf = fund_dict.get("free_cashflow")
    if fcf is not None:
        if fcf > 0:
            score += 4; reasons.append("Positive Free Cash Flow")
        elif fcf < 0:
            score -= 4; reasons.append("Negative Free Cash Flow")

    # ── Institutional / analyst signals (max ±20) ────────────────────────────
    inst_pct = fund_dict.get("inst_ownership_pct")
    if inst_pct is not None and inst_pct > 0.70:
        score += 5; reasons.append(f"High Inst. Ownership ({inst_pct*100:.0f}%)")

    if fund_dict.get("recent_upgrade"):
        score += 6; reasons.append("Recent Upgrade")
    if fund_dict.get("recent_downgrade"):
        score -= 6; reasons.append("Recent Downgrade")

    if fund_dict.get("insider_buy_recent"):
        score += 5; reasons.append("Insider Buy")
    if fund_dict.get("insider_sell_recent"):
        score -= 3; reasons.append("Insider Sell")

    rec = (fund_dict.get("recommendation") or "").lower()
    if rec in ("buy", "strongbuy", "strong_buy"):
        score += 4; reasons.append(f"Analyst: {rec.title()}")
    elif rec in ("sell", "strongsell", "strong_sell", "underperform"):
        score -= 4; reasons.append(f"Analyst: {rec.title()}")

    # Clamp and derive signal label
    score = max(-100, min(100, score))

    if score > 20:
        signal = "Bullish"
    elif score < -20:
        signal = "Bearish"
    else:
        signal = "Neutral"

    return {
        "fundamental_score":   score,
        "fundamental_signal":  signal,
        "fundamental_reasons": reasons[:5],
    }


def get_short_squeeze_score(fund_dict: dict) -> dict:
    """
    Compute a short squeeze likelihood score (0-100) from a pre-fetched fund_dict.

    Args:
        fund_dict: dict returned by get_fundamentals()

    Returns:
        {"squeeze_score": int, "is_squeeze_candidate": bool}
    """
    score = 0

    short_float = fund_dict.get("short_pct_float")
    if short_float is not None:
        spf = short_float * 100 if short_float < 1.0 else short_float  # normalise fraction vs percent
        if spf >= 20:
            score += 40
        elif spf >= 15:
            score += 25
        elif spf >= 10:
            score += 10

    short_ratio = fund_dict.get("short_ratio")
    if short_ratio is not None:
        if short_ratio >= 10:
            score += 30
        elif short_ratio >= 5:
            score += 15

    if fund_dict.get("insider_buy_recent"):
        score += 10

    if fund_dict.get("recent_upgrade"):
        score += 10

    inst_pct = fund_dict.get("inst_ownership_pct")
    if inst_pct is not None and inst_pct < 0.30:
        score += 10  # retail-dominated float

    score = min(100, score)

    # Squeeze candidate: meaningful short float AND high squeeze score
    spf_val = short_float
    if spf_val is not None and spf_val < 1.0:
        spf_val = spf_val * 100  # fraction → percent
    is_candidate = bool(
        spf_val is not None and spf_val >= 15 and score >= 40
    )

    return {"squeeze_score": score, "is_squeeze_candidate": is_candidate}


# ── Private helpers ───────────────────────────────────────────────────────────

def _safe(val):
    """Return numeric val as float, or None if invalid/infinite."""
    try:
        v = float(val)
        if v != v:   # NaN
            return None
        if abs(v) == float("inf"):
            return None
        return v
    except (TypeError, ValueError):
        return None
