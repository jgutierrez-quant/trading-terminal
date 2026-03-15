"""
DCF (Discounted Cash Flow) Valuation Module.

5-year DCF with base/bull/bear scenarios, dual terminal value methods
(perpetuity growth + EV/EBITDA exit multiple), and margin-of-safety signal.

Public API:
    compute_dcf(ticker, fund_dict=None) -> dict
"""

import logging
import math

import yfinance as yf

from data.fundamentals import get_fundamentals

logger = logging.getLogger(__name__)

# ── Defaults ─────────────────────────────────────────────────────────────────
_RISK_FREE_RATE     = 0.045   # 10Y Treasury proxy
_EQUITY_RISK_PREMIUM = 0.055  # Damodaran long-run average
_DEFAULT_COST_DEBT  = 0.05
_DEFAULT_TAX_RATE   = 0.21
_DEFAULT_TERMINAL_G = 0.025   # 2.5% perpetuity growth
_DEFAULT_EXIT_MULT  = 15.0    # EV/EBITDA exit multiple
_PROJECTION_YEARS   = 5


def compute_dcf(ticker: str, fund_dict: dict = None) -> dict:
    """
    5-year DCF valuation with base/bull/bear scenarios.

    Args:
        ticker:    Uppercase ticker symbol.
        fund_dict: Pre-fetched fundamentals dict (avoids duplicate yfinance call).

    Returns dict with intrinsic_value, margin_of_safety, signal, scenarios, etc.
    """
    ticker = ticker.upper()
    result = {
        "ticker": ticker,
        "intrinsic_value": None,
        "current_price": None,
        "margin_of_safety": None,
        "signal": None,
        "scenarios": {},
        "method_details": {},
        "inputs": {},
        "error": None,
    }

    try:
        # Fetch fundamentals if not provided
        if fund_dict is None:
            fund_dict = get_fundamentals(ticker)

        if fund_dict.get("error"):
            result["error"] = f"Fundamentals error: {fund_dict['error']}"
            return result

        # Pull additional data from yfinance .info
        info = yf.Ticker(ticker).info or {}

        fcf = _safe(fund_dict.get("free_cashflow")) or _safe(info.get("freeCashflow"))
        if not fcf or fcf <= 0:
            result["error"] = "No positive free cash flow available for DCF"
            return result

        revenue_growth = _safe(fund_dict.get("revenue_growth")) or _safe(info.get("revenueGrowth")) or 0.05
        market_cap = _safe(fund_dict.get("market_cap")) or _safe(info.get("marketCap"))
        shares = _safe(info.get("sharesOutstanding"))
        beta = _safe(info.get("beta")) or 1.0
        total_debt = _safe(info.get("totalDebt")) or 0
        interest_expense = _safe(info.get("interestExpense"))
        current_price = _safe(info.get("currentPrice")) or _safe(info.get("regularMarketPrice"))
        ev_ebitda = _safe(fund_dict.get("ev_ebitda")) or _safe(info.get("enterpriseToEbitda"))

        if not shares or shares <= 0:
            result["error"] = "Shares outstanding unavailable"
            return result

        if not current_price:
            result["error"] = "Current price unavailable"
            return result

        result["current_price"] = round(current_price, 2)

        # ── WACC ─────────────────────────────────────────────────────────
        cost_equity = _RISK_FREE_RATE + beta * _EQUITY_RISK_PREMIUM
        cost_debt = (abs(interest_expense) / total_debt) if (interest_expense and total_debt and total_debt > 0) else _DEFAULT_COST_DEBT

        equity_value = market_cap if market_cap else current_price * shares
        total_capital = equity_value + total_debt
        weight_equity = equity_value / total_capital if total_capital > 0 else 1.0
        weight_debt = total_debt / total_capital if total_capital > 0 else 0.0

        wacc = weight_equity * cost_equity + weight_debt * cost_debt * (1 - _DEFAULT_TAX_RATE)
        wacc = max(wacc, 0.04)  # floor at 4%

        # ── Exit multiple ────────────────────────────────────────────────
        exit_multiple = ev_ebitda if (ev_ebitda and 5 < ev_ebitda < 50) else _DEFAULT_EXIT_MULT

        # ── Scenarios ────────────────────────────────────────────────────
        scenarios = {
            "base": {
                "growth_multiplier": 1.0,
                "terminal_growth": _DEFAULT_TERMINAL_G,
            },
            "bull": {
                "growth_multiplier": 1.3,
                "terminal_growth": 0.03,
            },
            "bear": {
                "growth_multiplier": 0.5,
                "terminal_growth": 0.02,
            },
        }

        scenario_results = {}
        for name, params in scenarios.items():
            iv = _run_dcf(
                fcf=fcf,
                revenue_growth=revenue_growth * params["growth_multiplier"],
                wacc=wacc,
                terminal_growth=params["terminal_growth"],
                exit_multiple=exit_multiple,
                shares=shares,
            )
            scenario_results[name] = {
                "intrinsic": round(iv["blended_per_share"], 2),
                "assumptions": {
                    "revenue_growth": round(revenue_growth * params["growth_multiplier"], 4),
                    "terminal_growth": params["terminal_growth"],
                    "wacc": round(wacc, 4),
                },
            }

        base_iv = scenario_results["base"]["intrinsic"]
        result["intrinsic_value"] = base_iv
        result["scenarios"] = scenario_results

        # Method details from base case
        base_run = _run_dcf(fcf, revenue_growth, wacc, _DEFAULT_TERMINAL_G, exit_multiple, shares)
        result["method_details"] = {
            "perpetuity_growth": round(base_run["perp_per_share"], 2),
            "exit_multiple": round(base_run["exit_per_share"], 2),
            "blended": round(base_run["blended_per_share"], 2),
        }

        # Margin of safety
        if base_iv and current_price and current_price > 0:
            mos = (base_iv - current_price) / current_price
            result["margin_of_safety"] = round(mos, 4)

            if mos > 0.20:
                result["signal"] = "Undervalued"
            elif mos < -0.20:
                result["signal"] = "Overvalued"
            else:
                result["signal"] = "Fairly Valued"

        result["inputs"] = {
            "fcf_ttm": round(fcf, 0),
            "revenue_growth": round(revenue_growth, 4),
            "wacc": round(wacc, 4),
            "terminal_growth": _DEFAULT_TERMINAL_G,
            "shares_outstanding": shares,
        }

    except Exception as exc:
        logger.error("compute_dcf failed for %s: %s", ticker, exc)
        result["error"] = str(exc)

    return result


def _run_dcf(fcf, revenue_growth, wacc, terminal_growth, exit_multiple, shares):
    """
    Run a single DCF scenario.

    Projects FCF for 5 years with growth fading linearly to terminal_growth.
    Computes terminal value via both perpetuity growth and exit multiple methods.
    Returns per-share intrinsic values for each method and blended average.
    """
    projected_fcfs = []
    current_fcf = fcf

    for year in range(1, _PROJECTION_YEARS + 1):
        # Linear fade from revenue_growth to terminal_growth over 5 years
        fade = (year - 1) / (_PROJECTION_YEARS - 1) if _PROJECTION_YEARS > 1 else 1
        growth = revenue_growth * (1 - fade) + terminal_growth * fade
        current_fcf = current_fcf * (1 + growth)
        projected_fcfs.append(current_fcf)

    # Present value of projected FCFs
    pv_fcfs = sum(
        fcf_y / (1 + wacc) ** year
        for year, fcf_y in enumerate(projected_fcfs, 1)
    )

    fcf_terminal = projected_fcfs[-1]

    # Terminal value — perpetuity growth (Gordon Growth Model)
    if wacc > terminal_growth:
        tv_perp = fcf_terminal * (1 + terminal_growth) / (wacc - terminal_growth)
    else:
        tv_perp = fcf_terminal * 25  # fallback: 25x terminal FCF

    # Terminal value — exit multiple
    tv_exit = fcf_terminal * exit_multiple

    pv_tv_perp = tv_perp / (1 + wacc) ** _PROJECTION_YEARS
    pv_tv_exit = tv_exit / (1 + wacc) ** _PROJECTION_YEARS

    ev_perp = pv_fcfs + pv_tv_perp
    ev_exit = pv_fcfs + pv_tv_exit
    ev_blended = (ev_perp + ev_exit) / 2

    return {
        "perp_per_share": ev_perp / shares if shares else 0,
        "exit_per_share": ev_exit / shares if shares else 0,
        "blended_per_share": ev_blended / shares if shares else 0,
    }


def _safe(val):
    """Safe float conversion — returns None for invalid values."""
    try:
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None
