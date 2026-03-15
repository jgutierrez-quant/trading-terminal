"""
Quantitative Factor Model — Stage 11.

7-factor scoring engine based on academically validated anomalies.
Each factor is independently scored and normalized to a percentile (0-100)
against a peer universe. Composite score drives Long/Short/Neutral signals.

Factors and weights:
    1. Price Momentum      (18%)  — Jegadeesh & Titman 12-1 month
    2. Earnings Momentum   (18%)  — Ball & Brown SUE + PEAD
    3. Value               (14%)  — EV/EBITDA, P/E, P/B, FCF yield
    4. Quality             (18%)  — ROE, margins, leverage, stability
    5. Short Interest       (9%)  — Short float %, days-to-cover, squeeze
    6. Institutional Flow  (13%)  — Ownership %, insider ratio, upgrades
    7. DCF Intrinsic Value (10%)  — 5-year DCF margin of safety

Composite:
    Weighted average of available factor percentiles.
    Missing factors are excluded and weights renormalized.
    Score > 70 → Long candidate
    Score < 30 → Short candidate
    30-70      → Neutral

Universe normalization:
    Price momentum: computed from 50-stock S&P 500 sample via bulk download.
    Other factors: calibrated distribution priors from empirical S&P 500 data.
    Universe stats cached at .cache/universe_stats.json (TTL: 24 h).

Public API:
    compute_factor_model(ticker, fund_dict=None) -> dict
    get_universe_stats(force_refresh=False)      -> dict
"""

import json
import logging
import math
import os
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from data.dcf import compute_dcf
from data.fundamentals import get_fundamentals

logger = logging.getLogger(__name__)

# ── Weights ───────────────────────────────────────────────────────────────────
FACTOR_WEIGHTS = {
    "momentum":          0.18,
    "earnings_momentum": 0.18,
    "value":             0.14,
    "quality":           0.18,
    "short_interest":    0.09,
    "institutional":     0.13,
    "dcf":               0.10,
}

# ── Calibrated distribution priors (S&P 500 empirical) ───────────────────────
# Used when live universe stats are unavailable.
# Source: academic literature + long-run S&P 500 characteristics.
PRIORS = {
    "momentum_12_1m":  {"mean":  0.08,  "std": 0.30},   # 8% avg, 30% σ
    "momentum_1m":     {"mean":  0.007, "std": 0.07},
    "ev_ebitda":       {"mean": 14.0,   "std": 8.0},    # S&P 500 median ~14x
    "forward_pe":      {"mean": 18.0,   "std": 8.0},    # forward P/E ~18x
    "pb_ratio":        {"mean":  3.5,   "std": 2.5},
    "fcf_yield":       {"mean":  0.04,  "std": 0.03},   # 4% FCF yield
    "roe":             {"mean":  0.15,  "std": 0.12},
    "profit_margin":   {"mean":  0.10,  "std": 0.10},
    "revenue_growth":  {"mean":  0.08,  "std": 0.12},
    "debt_to_equity":  {"mean":  0.80,  "std": 0.80},
    "current_ratio":   {"mean":  1.5,   "std": 0.6},
    "short_pct_float": {"mean":  0.04,  "std": 0.04},
    "short_ratio":     {"mean":  3.5,   "std": 2.5},
    "inst_ownership":  {"mean":  0.70,  "std": 0.15},
    "dcf_margin_of_safety": {"mean": 0.0, "std": 0.25},
}

# ── Universe sample for z-score normalization ─────────────────────────────────
_UNIVERSE_SAMPLE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "JPM",
    "UNH", "XOM", "LLY", "JNJ", "V", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "AVGO", "COST", "PEP", "KO", "WMT", "MCD", "CRM", "NFLX",
    "ACN", "TMO", "LIN", "ABT", "ORCL", "DHR", "BAC", "AMD", "TXN",
    "QCOM", "NEE", "PM", "RTX", "HON", "LOW", "IBM", "CAT", "GE",
    "PYPL", "AMGN", "GILD", "MDT", "BSX",
]

_CACHE_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache")
_CACHE_FILE = os.path.join(_CACHE_DIR, "universe_stats.json")
_CACHE_TTL  = 86_400  # 24 hours


# ── Public entry point ────────────────────────────────────────────────────────

def compute_factor_model(ticker: str, fund_dict: dict = None) -> dict:
    """
    Compute all 6 factors and return a composite score.

    Args:
        ticker:    Uppercase ticker string.
        fund_dict: Pre-fetched fundamentals dict (avoids duplicate yfinance call).
                   If None, fetched internally.

    Returns dict with:
        composite_score    float 0-100 (percentile; 50 = market avg)
        composite_signal   "Long" | "Short" | "Neutral"
        factors            dict of 6 factor result dicts
        weights_used       dict of actual weights (renormalized when factors missing)
        data_completeness  float 0-1 (fraction of factors available)
        pead_candidate     bool — True if active PEAD long opportunity
    """
    ticker = ticker.upper()

    # Fetch fundamentals once if not provided
    if fund_dict is None:
        try:
            fund_dict = get_fundamentals(ticker)
        except Exception:
            fund_dict = {}

    # Compute all 7 factors independently
    factors = {
        "momentum":          _factor_momentum(ticker),
        "earnings_momentum": _factor_earnings_momentum(ticker),
        "value":             _factor_value(fund_dict),
        "quality":           _factor_quality(fund_dict),
        "short_interest":    _factor_short_interest(fund_dict),
        "institutional":     _factor_institutional(fund_dict),
        "dcf":               _factor_dcf(ticker, fund_dict),
    }

    # Build composite from available factors only
    available = {k: v for k, v in factors.items() if v.get("available")}
    data_completeness = len(available) / len(factors)

    if not available:
        return {
            "ticker":             ticker,
            "composite_score":    50.0,
            "composite_signal":   "Neutral",
            "factors":            factors,
            "weights_used":       {},
            "data_completeness":  0.0,
            "pead_candidate":     False,
            "error":              "No factors available",
        }

    # Renormalize weights to available factors
    raw_weights = {k: FACTOR_WEIGHTS[k] for k in available}
    total_w = sum(raw_weights.values())
    weights_used = {k: round(v / total_w, 4) for k, v in raw_weights.items()}

    composite = sum(
        factors[k]["percentile"] * weights_used[k]
        for k in available
    )
    composite = round(float(np.clip(composite, 0, 100)), 1)

    if composite > 70:
        signal = "Long"
    elif composite < 30:
        signal = "Short"
    else:
        signal = "Neutral"

    pead = bool(
        factors["earnings_momentum"].get("available") and
        factors["earnings_momentum"].get("details", {}).get("pead_candidate")
    )

    return {
        "ticker":            ticker,
        "composite_score":   composite,
        "composite_signal":  signal,
        "factors":           factors,
        "weights_used":      weights_used,
        "data_completeness": round(data_completeness, 2),
        "pead_candidate":    pead,
        "error":             None,
    }


# ── Universe stats cache ──────────────────────────────────────────────────────

def get_universe_stats(force_refresh: bool = False) -> dict:
    """
    Return cached universe factor statistics for z-score normalization.

    Only price momentum stats are computed from live data (fast bulk download).
    Other factors use calibrated PRIORS.

    Cache stored at .cache/universe_stats.json, TTL = 24 hours.
    Falls back to PRIORS silently if computation fails.
    """
    if not force_refresh and _cache_is_fresh():
        try:
            with open(_CACHE_FILE, encoding="utf-8") as f:
                return json.load(f).get("stats", PRIORS)
        except Exception:
            pass

    # Try to compute fresh universe stats
    try:
        stats = _compute_universe_stats()
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"computed_at": time.time(), "stats": stats}, f)
        return stats
    except Exception as exc:
        logger.debug("Universe stats computation failed, using priors: %s", exc)
        return PRIORS


def _cache_is_fresh() -> bool:
    try:
        if not os.path.exists(_CACHE_FILE):
            return False
        with open(_CACHE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        age = time.time() - data.get("computed_at", 0)
        return age < _CACHE_TTL
    except Exception:
        return False


def _compute_universe_stats() -> dict:
    """
    Compute live price momentum stats from universe sample.
    Uses yfinance bulk download (single API call for all tickers).
    """
    try:
        hist = yf.download(
            _UNIVERSE_SAMPLE, period="14mo", interval="1mo",
            auto_adjust=True, progress=False, group_by="ticker",
        )
        mom_12_1_list = []
        mom_1m_list   = []

        for t in _UNIVERSE_SAMPLE:
            try:
                if isinstance(hist.columns, pd.MultiIndex):
                    closes = hist[t]["Close"].dropna()
                else:
                    closes = hist["Close"].dropna()
                if len(closes) >= 13:
                    mom_12_1_list.append(float(closes.iloc[-2] / closes.iloc[-13]) - 1)
                if len(closes) >= 2:
                    mom_1m_list.append(float(closes.iloc[-1] / closes.iloc[-2]) - 1)
            except Exception:
                continue

        stats = dict(PRIORS)  # start with priors
        if len(mom_12_1_list) >= 10:
            stats["momentum_12_1m"] = {
                "mean": float(np.mean(mom_12_1_list)),
                "std":  float(np.std(mom_12_1_list)),
                "n":    len(mom_12_1_list),
            }
        if len(mom_1m_list) >= 10:
            stats["momentum_1m"] = {
                "mean": float(np.mean(mom_1m_list)),
                "std":  float(np.std(mom_1m_list)),
                "n":    len(mom_1m_list),
            }
        return stats
    except Exception as exc:
        logger.debug("_compute_universe_stats failed: %s", exc)
        return PRIORS


# ── Factor 1: Price Momentum ──────────────────────────────────────────────────

def _factor_momentum(ticker: str) -> dict:
    """
    Jegadeesh & Titman (1993) — 12-1 month price momentum.

    12-1m: return from 13 months ago to 1 month ago (skips most recent month
    to avoid short-term reversal contamination).
    1m: most recent month return (separate signal).
    """
    try:
        hist = yf.Ticker(ticker).history(period="14mo", interval="1mo")
        if hist is None or len(hist) < 3:
            return _unavailable("Insufficient price history")

        closes = hist["Close"].dropna()

        mom_12_1 = None
        mom_1m   = None
        if len(closes) >= 13:
            mom_12_1 = float(closes.iloc[-2] / closes.iloc[-13]) - 1
        if len(closes) >= 2:
            mom_1m = float(closes.iloc[-1] / closes.iloc[-2]) - 1

        if mom_12_1 is None:
            return _unavailable("Not enough price history for 12-1m momentum")

        stats  = get_universe_stats().get("momentum_12_1m", PRIORS["momentum_12_1m"])
        z      = _to_z(mom_12_1, stats["mean"], stats["std"])
        pct    = _z_to_pct(z)
        signal = "Bullish" if pct > 65 else ("Bearish" if pct < 35 else "Neutral")

        return _factor_result(
            raw=round(mom_12_1, 4),
            z_score=round(z, 3),
            percentile=round(pct, 1),
            signal=signal,
            details={
                "momentum_12_1m_pct": round(mom_12_1 * 100, 2),
                "momentum_1m_pct":    round(mom_1m * 100, 2) if mom_1m is not None else None,
                "months_of_data":     len(closes),
            },
        )
    except Exception as exc:
        logger.debug("Factor 1 (momentum) failed for %s: %s", ticker, exc)
        return _unavailable(str(exc))


# ── Factor 2: Earnings Momentum / PEAD ───────────────────────────────────────

def _factor_earnings_momentum(ticker: str) -> dict:
    """
    Ball & Brown (1968) / Bernard & Thomas (1989).

    SUE (Standardized Unexpected Earnings):
        SUE = (actual_EPS - estimated_EPS) / std(recent_surprises_last_4Q)

    PEAD flag: positive SUE + earnings within last 28 calendar days.
    Academic basis: post-earnings announcement drift — stocks with positive
    surprises continue to drift upward for ~60 trading days.
    """
    try:
        t  = yf.Ticker(ticker)
        eh = t.earnings_history
        if eh is None or eh.empty:
            return _unavailable("No earnings history available")

        est_col = act_col = None
        for c in eh.columns:
            cl = c.lower()
            if "estimate" in cl:
                est_col = c
            if "actual" in cl:
                act_col = c
        if not est_col or not act_col:
            return _unavailable("EPS estimate/actual columns not found")

        last4 = eh.tail(4).dropna(subset=[est_col, act_col])
        if last4.empty:
            return _unavailable("No clean EPS data in last 4 quarters")

        surprises     = (last4[act_col] - last4[est_col]).values.astype(float)
        last_surprise = float(surprises[-1])
        std_surprises = float(np.std(surprises)) if len(surprises) > 1 else None

        sue = None
        if std_surprises and std_surprises > 0:
            sue = last_surprise / std_surprises
        elif last_surprise != 0:
            sue = 1.0 if last_surprise > 0 else -1.0

        # PEAD check
        pead_flag = False
        days_since = None
        last_earn_date = None
        try:
            last_idx = eh.index[-1]
            if hasattr(last_idx, "date"):
                last_earn_date = last_idx.date()
            else:
                last_earn_date = pd.Timestamp(last_idx).date()
            days_since = (date.today() - last_earn_date).days
            pead_flag = bool(
                days_since is not None and days_since <= 28 and last_surprise > 0
            )
        except Exception:
            pass

        beat_rate = float((last4[act_col] > last4[est_col]).mean())

        # Percentile from SUE — SUE ≈ standard normal by construction
        if sue is not None:
            z   = float(np.clip(sue, -3.5, 3.5))
            pct = _z_to_pct(z)
        else:
            z   = 0.0
            pct = 50.0 + (25.0 if last_surprise > 0 else -25.0)

        # PEAD boosts signal
        signal = "Bullish" if pct > 65 or pead_flag else ("Bearish" if pct < 35 else "Neutral")

        return _factor_result(
            raw=round(sue, 3) if sue is not None else None,
            z_score=round(z, 3),
            percentile=round(pct, 1),
            signal=signal,
            details={
                "sue_score":          round(sue, 3) if sue is not None else None,
                "last_eps_surprise":  round(last_surprise, 4),
                "earnings_beat_rate": round(beat_rate, 2),
                "pead_candidate":     pead_flag,
                "last_earnings_date": str(last_earn_date) if last_earn_date else None,
                "days_since_earnings": days_since,
            },
        )
    except Exception as exc:
        logger.debug("Factor 2 (earnings momentum) failed for %s: %s", ticker, exc)
        return _unavailable(str(exc))


# ── Factor 3: Value ───────────────────────────────────────────────────────────

def _factor_value(fund_dict: dict) -> dict:
    """
    Fama & French (1992) value factor — composite of low-valuation metrics.

    EV/EBITDA, Forward P/E, P/B, FCF Yield — each scored as percentile
    relative to calibrated S&P 500 distribution priors.
    Low relative valuation = high value score.
    """
    try:
        sub_scores = []
        details    = {}

        ev_ebitda = _sf(fund_dict.get("ev_ebitda"))
        fwd_pe    = _sf(fund_dict.get("forward_pe") or fund_dict.get("pe_ratio"))
        pb_ratio  = _sf(fund_dict.get("pb_ratio"))
        fcf       = _sf(fund_dict.get("free_cashflow"))
        mkt_cap   = _sf(fund_dict.get("market_cap"))

        if ev_ebitda is not None and 0 < ev_ebitda < 100:
            pct = _z_to_pct(-_to_z(ev_ebitda, PRIORS["ev_ebitda"]["mean"],
                                               PRIORS["ev_ebitda"]["std"]))
            sub_scores.append(pct)
            details["ev_ebitda"]     = round(ev_ebitda, 1)
            details["ev_ebitda_pct"] = round(pct, 1)

        if fwd_pe is not None and 0 < fwd_pe < 150:
            pct = _z_to_pct(-_to_z(fwd_pe, PRIORS["forward_pe"]["mean"],
                                            PRIORS["forward_pe"]["std"]))
            sub_scores.append(pct)
            details["forward_pe"]     = round(fwd_pe, 1)
            details["forward_pe_pct"] = round(pct, 1)

        if pb_ratio is not None and 0 < pb_ratio < 50:
            pct = _z_to_pct(-_to_z(pb_ratio, PRIORS["pb_ratio"]["mean"],
                                              PRIORS["pb_ratio"]["std"]))
            sub_scores.append(pct)
            details["pb_ratio"]     = round(pb_ratio, 1)
            details["pb_ratio_pct"] = round(pct, 1)

        if fcf is not None and mkt_cap and mkt_cap > 0:
            fcf_yield = fcf / mkt_cap
            pct = _z_to_pct(_to_z(fcf_yield, PRIORS["fcf_yield"]["mean"],
                                              PRIORS["fcf_yield"]["std"]))
            sub_scores.append(pct)
            details["fcf_yield_pct_val"] = round(fcf_yield * 100, 2)
            details["fcf_yield_pct"]     = round(pct, 1)

        if not sub_scores:
            return _unavailable("No valuation metrics available")

        composite_pct = float(np.mean(sub_scores))
        signal = "Bullish" if composite_pct > 65 else ("Bearish" if composite_pct < 35 else "Neutral")

        return _factor_result(
            raw=round(composite_pct / 100, 3),
            z_score=round(_pct_to_z(composite_pct), 3),
            percentile=round(composite_pct, 1),
            signal=signal,
            details=details,
        )
    except Exception as exc:
        logger.debug("Factor 3 (value) failed: %s", exc)
        return _unavailable(str(exc))


# ── Factor 4: Quality ─────────────────────────────────────────────────────────

def _factor_quality(fund_dict: dict) -> dict:
    """
    Novy-Marx (2013) gross profitability / Asness et al. quality factor.

    Composite of ROE, profit margins, revenue growth, leverage (D/E inverted),
    current ratio, and EPS beat consistency.
    """
    try:
        sub_scores = []
        details    = {}

        roe     = _sf(fund_dict.get("roe"))
        margin  = _sf(fund_dict.get("profit_margins"))
        rev_g   = _sf(fund_dict.get("revenue_growth"))
        dte     = _sf(fund_dict.get("debt_to_equity"))
        cur_r   = _sf(fund_dict.get("current_ratio"))
        beat_r  = _sf(fund_dict.get("earnings_beat_rate"))

        if roe is not None:
            pct = _z_to_pct(_to_z(roe, PRIORS["roe"]["mean"], PRIORS["roe"]["std"]))
            sub_scores.append(pct)
            details["roe_pct_val"] = round(roe * 100, 1)
            details["roe_pct"]     = round(pct, 1)

        if margin is not None:
            pct = _z_to_pct(_to_z(margin, PRIORS["profit_margin"]["mean"],
                                           PRIORS["profit_margin"]["std"]))
            sub_scores.append(pct)
            details["profit_margin_pct_val"] = round(margin * 100, 1)
            details["profit_margin_pct"]     = round(pct, 1)

        if rev_g is not None:
            pct = _z_to_pct(_to_z(rev_g, PRIORS["revenue_growth"]["mean"],
                                          PRIORS["revenue_growth"]["std"]))
            sub_scores.append(pct)
            details["revenue_growth_pct_val"] = round(rev_g * 100, 1)
            details["revenue_growth_pct"]     = round(pct, 1)

        if dte is not None:
            # Low D/E is good — invert
            pct = _z_to_pct(-_to_z(dte, PRIORS["debt_to_equity"]["mean"],
                                        PRIORS["debt_to_equity"]["std"]))
            sub_scores.append(pct)
            details["debt_to_equity"]     = round(dte, 2)
            details["debt_to_equity_pct"] = round(pct, 1)

        if cur_r is not None:
            pct = _z_to_pct(_to_z(cur_r, PRIORS["current_ratio"]["mean"],
                                          PRIORS["current_ratio"]["std"]))
            sub_scores.append(pct)
            details["current_ratio"]     = round(cur_r, 2)
            details["current_ratio_pct"] = round(pct, 1)

        if beat_r is not None:
            # 0-1 → 0-100 percentile linearly (0.65 = average → 50th)
            pct = float(np.clip((beat_r - 0.25) / 0.75 * 100, 0, 100))
            sub_scores.append(pct)
            details["eps_beat_rate"]     = round(beat_r, 2)
            details["eps_beat_rate_pct"] = round(pct, 1)

        if not sub_scores:
            return _unavailable("No quality metrics available")

        composite_pct = float(np.mean(sub_scores))
        signal = "Bullish" if composite_pct > 65 else ("Bearish" if composite_pct < 35 else "Neutral")

        return _factor_result(
            raw=round(composite_pct / 100, 3),
            z_score=round(_pct_to_z(composite_pct), 3),
            percentile=round(composite_pct, 1),
            signal=signal,
            details=details,
        )
    except Exception as exc:
        logger.debug("Factor 4 (quality) failed: %s", exc)
        return _unavailable(str(exc))


# ── Factor 5: Short Interest / Squeeze Potential ──────────────────────────────

def _factor_short_interest(fund_dict: dict) -> dict:
    """
    Asquith, Pathak & Ritter short interest studies.

    High short interest is bearish by itself, but combined with positive
    momentum signals a potential short squeeze.

    Momentum context from same fund_dict (uses recent_upgrade as proxy).
    """
    try:
        spf  = _sf(fund_dict.get("short_pct_float"))
        sr   = _sf(fund_dict.get("short_ratio"))

        if spf is None and sr is None:
            return _unavailable("No short interest data")

        # Normalize short float % (high short % = bearish signal)
        details = {}
        spf_val = spf * 100 if (spf is not None and spf < 1.0) else spf  # fraction → pct

        squeeze_score = 0
        bearish_score = 0

        if spf_val is not None:
            details["short_pct_float"] = round(spf_val, 2)
            if spf_val >= 20:
                squeeze_score += 40; bearish_score += 40
            elif spf_val >= 15:
                squeeze_score += 25; bearish_score += 25
            elif spf_val >= 10:
                squeeze_score += 10; bearish_score += 10
            elif spf_val >= 5:
                bearish_score += 5

        if sr is not None:
            details["short_ratio_days"] = round(sr, 1)
            if sr >= 10:
                squeeze_score += 30; bearish_score += 20
            elif sr >= 5:
                squeeze_score += 15; bearish_score += 10

        has_positive_momentum = bool(
            fund_dict.get("recent_upgrade") or
            fund_dict.get("insider_buy_recent")
        )
        is_squeeze_candidate = bool(
            spf_val is not None and spf_val >= 15 and squeeze_score >= 40
        )
        details["is_squeeze_candidate"] = is_squeeze_candidate

        if is_squeeze_candidate and has_positive_momentum:
            # Squeeze: high short + positive momentum = bullish signal
            pct    = min(50 + squeeze_score, 90)
            signal = "Bullish"
        elif bearish_score >= 20:
            # High short without positive momentum = bearish
            pct    = max(50 - bearish_score * 0.6, 15)
            signal = "Bearish"
        else:
            pct    = 50.0
            signal = "Neutral"

        return _factor_result(
            raw=spf_val,
            z_score=round(_pct_to_z(pct), 3),
            percentile=round(float(pct), 1),
            signal=signal,
            details=details,
        )
    except Exception as exc:
        logger.debug("Factor 5 (short interest) failed: %s", exc)
        return _unavailable(str(exc))


# ── Factor 6: Institutional Flow ──────────────────────────────────────────────

def _factor_institutional(fund_dict: dict) -> dict:
    """
    Nofsinger & Sias (1999) institutional momentum.

    Uses current institutional ownership %, insider buy/sell signal,
    and analyst upgrade/downgrade direction as proxies for flow.

    Limitation: yfinance does not provide quarterly ownership change
    (13F delta). Current ownership % and insider signals are used as
    available proxies.
    """
    try:
        inst_pct  = _sf(fund_dict.get("inst_ownership_pct"))
        upgrade   = bool(fund_dict.get("recent_upgrade"))
        downgrade = bool(fund_dict.get("recent_downgrade"))
        ins_buy   = bool(fund_dict.get("insider_buy_recent"))
        ins_sell  = bool(fund_dict.get("insider_sell_recent"))
        rec       = (fund_dict.get("recommendation") or "").lower()

        if inst_pct is None and not any([upgrade, downgrade, ins_buy, ins_sell]):
            return _unavailable("No institutional flow data")

        sub_scores = []
        details    = {}

        if inst_pct is not None:
            pct = _z_to_pct(_to_z(inst_pct, PRIORS["inst_ownership"]["mean"],
                                             PRIORS["inst_ownership"]["std"]))
            sub_scores.append(pct)
            details["inst_ownership_pct_val"] = round(inst_pct * 100, 1)
            details["inst_ownership_pct"]     = round(pct, 1)

        # Analyst revisions (+/- 10 percentile points each)
        analyst_adj = 0
        if upgrade:
            analyst_adj += 10
            details["recent_upgrade"] = True
        if downgrade:
            analyst_adj -= 10
            details["recent_downgrade"] = True
        if rec in ("buy", "strongbuy", "strong_buy"):
            analyst_adj += 7
        elif rec in ("sell", "strongsell", "strong_sell", "underperform"):
            analyst_adj -= 7
        details["recommendation"] = rec or None

        # Insider transactions (+/- 8 percentile points)
        insider_adj = 0
        if ins_buy:
            insider_adj += 8
            details["insider_buy_recent"] = True
        if ins_sell:
            insider_adj -= 5
            details["insider_sell_recent"] = True

        base_pct = float(np.mean(sub_scores)) if sub_scores else 50.0
        composite_pct = float(np.clip(base_pct + analyst_adj + insider_adj, 5, 95))
        signal = "Bullish" if composite_pct > 65 else ("Bearish" if composite_pct < 35 else "Neutral")

        return _factor_result(
            raw=round(composite_pct / 100, 3),
            z_score=round(_pct_to_z(composite_pct), 3),
            percentile=round(composite_pct, 1),
            signal=signal,
            details=details,
        )
    except Exception as exc:
        logger.debug("Factor 6 (institutional) failed: %s", exc)
        return _unavailable(str(exc))


# ── Factor 7: DCF Intrinsic Value ────────────────────────────────────────────

def _factor_dcf(ticker: str, fund_dict: dict) -> dict:
    """
    DCF-based valuation factor.

    Positive margin of safety (undervalued) → Bullish percentile.
    Negative margin of safety (overvalued)  → Bearish percentile.
    """
    try:
        dcf_result = compute_dcf(ticker, fund_dict)

        if dcf_result.get("error") or dcf_result.get("margin_of_safety") is None:
            return _unavailable(dcf_result.get("error") or "DCF computation unavailable")

        mos = dcf_result["margin_of_safety"]

        # Convert margin_of_safety to z-score using prior distribution
        prior = PRIORS["dcf_margin_of_safety"]
        z = _to_z(mos, prior["mean"], prior["std"])
        pct = _z_to_pct(z)

        signal_str = dcf_result.get("signal", "Neutral")
        if signal_str == "Undervalued":
            signal = "Bullish"
        elif signal_str == "Overvalued":
            signal = "Bearish"
        else:
            signal = "Neutral"

        return _factor_result(
            raw=round(mos, 4),
            z_score=round(z, 3),
            percentile=round(pct, 1),
            signal=signal,
            details={
                "intrinsic_value": dcf_result.get("intrinsic_value"),
                "current_price": dcf_result.get("current_price"),
                "margin_of_safety_pct": round(mos * 100, 1),
                "dcf_signal": signal_str,
                "scenarios": dcf_result.get("scenarios"),
            },
        )
    except Exception as exc:
        logger.debug("Factor 7 (DCF) failed for %s: %s", ticker, exc)
        return _unavailable(str(exc))


# ── Private helpers ───────────────────────────────────────────────────────────

def _factor_result(raw, z_score, percentile, signal, details) -> dict:
    return {
        "raw":        raw,
        "z_score":    z_score,
        "percentile": float(np.clip(percentile, 0, 100)),
        "signal":     signal,
        "details":    details,
        "available":  True,
    }


def _unavailable(reason: str) -> dict:
    return {
        "raw":        None,
        "z_score":    None,
        "percentile": 50.0,   # neutral when unavailable
        "signal":     "Neutral",
        "details":    {"reason": reason},
        "available":  False,
    }


def _z_to_pct(z: float) -> float:
    """Standard normal CDF: converts z-score to percentile (0-100)."""
    try:
        return float(0.5 * (1.0 + math.erf(z / math.sqrt(2)))) * 100.0
    except Exception:
        return 50.0


def _pct_to_z(pct: float) -> float:
    """Approximate inverse normal CDF: percentile (0-100) → z-score."""
    p = max(0.001, min(0.999, pct / 100.0))
    # Rational approximation (Beasley-Springer-Moro)
    a = [0, -3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [0, -5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [0, -7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [0,  7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    try:
        if p < 0.02425:
            q = math.sqrt(-2 * math.log(p))
            return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) / \
                   ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
        elif p <= 0.97575:
            q = p - 0.5
            r = q * q
            return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6])*q / \
                   (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1)
        else:
            q = math.sqrt(-2 * math.log(1 - p))
            return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) / \
                    ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    except Exception:
        return (pct - 50.0) / 25.0


def _to_z(raw: float, mean: float, std: float) -> float:
    """Compute z-score; returns 0 if std is 0."""
    if not std or std == 0:
        return 0.0
    return float((raw - mean) / std)


def _sf(val):
    """Safe float conversion — returns None for invalid/infinite values."""
    try:
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


