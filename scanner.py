#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Options parity scanner (Tradier) — hardened with safe borrow handling + audit CSV fix

- Safe borrow handling:
  * get_borrow_rate(sym) -> float or None
  * norm_borrow_rate(None) -> 0.0 (when REQUIRE_BORROW_FOR_REVERSAL=False)
  * Enforce BORROW_RATE_CAP only when a real rate is known
  * Subtract borrow cost only for reversals (gap < 0)
- Audit CSV writer filters to a fixed header so extra fields never crash DictWriter
"""

import os
import csv
import time
import math
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from statistics import median

import requests
from dotenv import load_dotenv

# ===================== CONFIG =====================

load_dotenv()

def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None: return default
    return str(v).strip().lower() in ("1","true","yes","y","on")

USE_SANDBOX   = False
TRADIER_HOST  = "https://sandbox.tradier.com" if USE_SANDBOX else "https://api.tradier.com"
TRADIER_TOKEN = os.getenv("TRADIER_TOKEN")
if not TRADIER_TOKEN:
    raise SystemExit("Missing TRADIER_TOKEN in environment (.env)")

DIVIDENDS_ENABLED = _env_bool("DIVIDENDS_ENABLED", True)
EARNINGS_ENABLED  = _env_bool("EARNINGS_ENABLED", True)

# ----- Universe / discovery -----
MAX_UNDERLYINGS     = 35
MAX_EXPIRIES        = 3
MAX_DTE             = 6           # (tighten later to 4 if you want)
SEED_UNDERLYINGS    = ["SPY","QQQ","AAPL","MSFT","TSLA","NVDA","AMD","META","GOOGL","AMZN"]
USE_SEED_WHEN_EMPTY = True

# ----- Cadence / output -----
REFRESH_SECONDS = 15
TOP_N           = 5
WRITE_CSV       = True
CSV_FILE        = "options_parity_hits.csv"

# ----- Audit (skips & reasons) -----
AUDIT_ENABLED   = True
AUDIT_FILE      = "options_parity_audit.csv"
AUDIT_BUFFER: List[Dict] = []

def log_skip(sym, exp, K, **kw):
    if not AUDIT_ENABLED: return
    row = {
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "underlying": sym, "expiry": exp, "strike": K,
        **kw
    }
    AUDIT_BUFFER.append(row)

def write_audit_csv():
    if not AUDIT_ENABLED or not AUDIT_BUFFER:
        return
    newfile = not os.path.exists(AUDIT_FILE)
    with open(AUDIT_FILE, "a", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp","underlying","expiry","strike","reason",
                "spot","call_mid","put_mid","call_spread","put_spread",
                "spread_frac","oi","vol","quote_age_sec","borrow_rate",
                "div_pv","earnings_date","gap","dte_days","max_dte",
                "borrow_required","err"
            ]
        )
        if newfile:
            w.writeheader()
        for r in AUDIT_BUFFER:
            filtered = {k: r.get(k, "") for k in w.fieldnames}
            w.writerow(filtered)
    AUDIT_BUFFER.clear()

# ----- Model / filters -----
RISK_FREE         = 0.045         # annual risk free for PV(K)
PARITY_THRESH     = 0.01          # |gap| >= $0.01/share
SPREAD_MAX        = 0.35          # max spread fraction vs mid
BAND_PCT          = 0.50          # ± band around spot for strikes

# Liquidity depth (per leg)
MIN_OI            = 200
MIN_VOL           = 10

# Quote staleness
QUOTE_MAX_AGE_SEC = 6             # (ignored if no timestamps)

# Haircuts / costs (simple model; adjust for your broker)
SLIP_FRAC_OF_SPREAD = 0.35        # of quoted spread/leg as slippage
STOCK_SLIP_CENTS    = 0.01        # $/share stock slip
OPTION_FEE_PER_CT   = 0.00
STOCK_FEE_PER_TRADE = 0.00

# ----- Borrow handling (safe defaults for Tradier discovery) -----
BORROW_RATE_CAP             = 0.06   # reject reversals if borrow > 6%/yr (only if known)
REQUIRE_BORROW_FOR_REVERSAL = False  # allow reversals when borrow is unknown

# Stub borrow rates (annualized) until IBKR shortable feed is wired in
PER_SYMBOL_BORROW: Dict[str, float] = {
    # Broad ETFs
    "SPY": 0.002, "QQQ": 0.002, "IWM": 0.003, "DIA": 0.002,
    # Mega-cap tech
    "AAPL": 0.010, "MSFT": 0.010, "GOOGL": 0.008, "AMZN": 0.010,
    "META": 0.010, "NVDA": 0.020, "AMD":  0.020, "TSLA": 0.080,
    # Financials / staples / energy
    "JPM":  0.010, "BAC": 0.010, "GS": 0.012, "BRK.B": 0.006,
    "XOM":  0.005, "CVX": 0.005, "KO": 0.008,  "WMT": 0.010,
}

# Capital model (per-contract)
MARGIN_FRACTION   = 0.50          # 50% Reg-T initial requirement
MARGIN_SAFETY     = 1.00          # no extra multiplier
MARGIN_RATE       = 0.08          # (not used here)

# ================= OUTPUT & SIZING =================
VERBOSE_OUTPUT      = False
DISPLAY_ROI_IN_BPS  = True
ASSUMED_CYCLE_DAYS  = 3

# Legacy single-budget (only used if USE_DUAL_CAPS=False)
USE_MARGIN_FOR_SIZE = True
ACCOUNT_BUDGET_USD  = 50_000

# ----- Dual capital (recommended) -----
USE_DUAL_CAPS   = True
CASH_CAP_USD    = 0                # conversions use cash → reversals favored
MARGIN_CAP_USD  = 50_000           # reversals use margin (your BP)

# ----- Targeting & quality gates -----
TARGET_TIERS        = [1, 2, 3]
ONLY_PROFITABLE     = True
MIN_FILL_PROB       = 0.60
MIN_ROI_BPS         = 0
MIN_TOTAL_NET_USD   = 0.0

# -------------------- Static TIER CONFIG (fallback) --------------------
TIER_MAP = {
    "SPY": 1, "QQQ": 1, "AAPL": 1, "MSFT": 1,
    "NVDA": 2, "AMZN": 2, "META": 2,
    "TSLA": 3, "AMD": 3, "GOOGL": 3,
    "IWM": 4, "DIA": 4, "NFLX": 4, "BABA": 4, "INTC": 4,
}
TIER_SAFETY_SCORE = {1: 1.00, 2: 0.80, 3: 0.60, 4: 0.40}
PRIORITY_WEIGHTS  = {"roi": 0.50, "total_net": 0.30, "tier": 0.20}

# Fill-probability model
FILL_WEIGHTS   = {"tier": 0.40, "edge_vs_spread": 0.35, "dte": 0.15, "spread_frac": 0.10}
TIER_BASE_PROB = {1: 1.00, 2: 0.85, 3: 0.65, 4: 0.40}

# ===== Static Tier-1 anchors (always Tier 1) =====
TIER1_ANCHORS = [
    "SPY","QQQ","IWM","DIA","XLF",
    "AAPL","MSFT","NVDA","AMZN","META","TSLA","AMD","GOOGL",
    "JPM","BAC","GS","BRK.B",
    "XOM","CVX",
    "UNH","JNJ","PFE",
    "WMT","KO",
]

# ===== Dynamic tiering =====
DYNAMIC_TIERING        = True
TIER_CACHE_TTL_SEC     = 3600
TIERING_MAX_EXPIRIES   = 2
TIERING_MAX_DTE        = 10
TIERING_STRIKE_BAND    = 0.20
TIERING_MIN_OBS        = 6
TIERING_UNIVERSE = [
    "SPY","QQQ","IWM","DIA","XLK","XLF","XLV","SMH","XLE",
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO","COST","ORCL",
    "NFLX","AMD","INTC","AMAT","CRM","CSCO","QCOM","BKNG","ADBE","LIN",
    "WMT","HD","MCD","PG","PEP","KO","PFE","JNJ","UNH","MRK",
    "BA","CAT","GE","UPS","CVX","XOM","PM","TMO","NKE","ABBV",
]

# ===== Market-hours + holiday/early close guard =====
from datetime import datetime, time as dtime
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

MARKET_HOURS_ONLY  = False
MARKET_TZ          = "America/New_York"
MARKET_OPEN_ET     = dtime(9, 30)
MARKET_CLOSE_ET    = dtime(16, 0)
PREOPEN_WARMUP_MIN = 10

# Prefer real exchange calendar if available
try:
    import pandas_market_calendars as mcal
    _NY_CAL = mcal.get_calendar("XNYS")
except Exception:
    _NY_CAL = None

def _now_et():
    if ZoneInfo:
        return datetime.now(ZoneInfo(MARKET_TZ))
    return datetime.now()

def _is_market_session(now_et: datetime) -> bool:
    if _NY_CAL is not None:
        sched = _NY_CAL.schedule(start_date=now_et.date(), end_date=now_et.date())
        if sched.empty: return False
        o = sched.iloc[0]["market_open"].tz_convert(MARKET_TZ)
        c = sched.iloc[0]["market_close"].tz_convert(MARKET_TZ)
        preopen = (o - dt.timedelta(minutes=PREOPEN_WARMUP_MIN))
        return preopen <= now_et <= c
    if now_et.weekday() > 4: return False
    t = now_et.time()
    preopen_dt = dt.datetime.combine(now_et.date(), MARKET_OPEN_ET) - dt.timedelta(minutes=PREOPEN_WARMUP_MIN)
    return (t >= preopen_dt.time()) and (t <= MARKET_CLOSE_ET)

def _sleep_until_next_session():
    time.sleep(300)

# ===================== HTTP (shared) =====================

SESS = requests.Session()
SESS.headers.update({
    "Authorization": f"Bearer {TRADIER_TOKEN}",
    "Accept": "application/json",
    "User-Agent": "ParityScanner/1.9"
})

def _get(url: str, params: Dict = None):
    max_tries = 6
    backoff = 0.4
    for _ in range(max_tries):
        try:
            r = SESS.get(url, params=params, timeout=20)
        except requests.RequestException:
            time.sleep(backoff); backoff = min(backoff * 1.7, 8.0); continue
        if r.status_code == 200:
            try: return r.json()
            except Exception: return {}
        if r.status_code in (429, 503, 502, 504, 500):
            retry_after = r.headers.get("Retry-After")
            sleep_s = float(retry_after) if retry_after else backoff
            time.sleep(sleep_s); backoff = min(backoff * 1.7, 12.0); continue
        if r.status_code == 401:
            raise SystemExit("Tradier auth failed (401). Check TRADIER_TOKEN.")
        time.sleep(min(backoff, 2.0)); backoff = min(backoff * 1.5, 6.0)
    return {}

# ===================== DATA STRUCTS =====================

@dataclass
class Hit:
    timestamp: str
    base: str
    expiry: str
    strike: float
    spot: float
    call_mid: float
    put_mid: float
    call_spread: float
    put_spread: float
    gap: float                  # (call - put - (spot - PV(K))) $/sh; + => conversion
    strategy: str               # "conversion" or "reversal"
    spread_frac: float          # max(call_spread/mid, put_spread/mid)
    cap_cash_ct: float          # spot*100
    cap_margin_ct: float        # spot*100*margin_fraction
    net_per_ct: float           # per-contract net after haircuts
    dte_days: int               # days to expiry (for prob model)

# ===================== HELPERS =====================

def now_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def days_to(exp: str) -> int:
    try:
        d = dt.datetime.strptime(exp, "%Y-%m-%d").date()
        return (d - dt.date.today()).days
    except Exception:
        return 9999

def discount_strike(K: float, dte_days: int) -> float:
    T = max(0.0, dte_days) / 365.0
    return K * math.exp(-RISK_FREE * T)

def mid(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    try:
        b = float(bid); a = float(ask)
        if b <= 0 or a <= 0:
            return None
        return (a + b) / 2.0
    except Exception:
        return None

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def per_contract_cap(spot: float, use_margin: bool) -> float:
    eff = (MARGIN_FRACTION * MARGIN_SAFETY) if use_margin else 1.0
    return float(spot) * 100.0 * eff

def contracts_fit(budget_usd: float, cap_per_ct: float) -> int:
    if cap_per_ct <= 0 or budget_usd <= 0:
        return 0
    return max(0, int(budget_usd // cap_per_ct))

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(x)))

# ----- Borrow helpers -----
def get_borrow_rate(sym: str) -> Optional[float]:
    """Return annualized borrow rate (e.g., 0.02 for 2%/yr), or None if unknown."""
    try:
        r = PER_SYMBOL_BORROW.get((sym or "").upper())
        return float(r) if r is not None else None
    except Exception:
        return None

def norm_borrow_rate(br: Optional[float]) -> float:
    """Safe number for math: if borrow is unknown and unknowns allowed, use 0.0."""
    return 0.0 if br is None else float(br)

# ===================== TRADIER CALLS =====================

def list_expiries(symbol: str) -> List[str]:
    url = f"{TRADIER_HOST}/v1/markets/options/expirations"
    js = _get(url, {"symbol": symbol, "includeAllRoots": "true", "strikes": "false"})
    exps = js.get("expirations", {}).get("date", [])
    if isinstance(exps, str):
        exps = [exps]
    return exps or []

def option_chain(symbol: str, expiration: str) -> List[Dict]:
    url = f"{TRADIER_HOST}/v1/markets/options/chains"
    js = _get(url, {"symbol": symbol, "expiration": expiration, "greeks": "false"})
    rows = js.get("options", {}).get("option", [])
    if isinstance(rows, dict):
        rows = [rows]
    return rows or []

def quote(symbol: str) -> Optional[float]:
    url   = f"{TRADIER_HOST}/v1/markets/quotes"
    js    = _get(url, {"symbols": symbol})
    q     = js.get("quotes", {}).get("quote")
    if isinstance(q, list):
        q = q[0] if q else None
    if not q:
        return None
    last = q.get("last")
    bid  = q.get("bid"); ask = q.get("ask")
    if bid and ask and bid > 0 and ask > 0:
        return (float(bid) + float(ask)) / 2.0
    return safe_float(last, default=None)

# ===================== DYNAMIC TIERING =====================

_dynamic_tier_map: Dict[str, int] = {}
_dynamic_tier_ts: float = 0.0

def _tiering_collect_metrics(sym: str) -> Optional[Dict[str, float]]:
    spot = quote(sym)
    if not spot:
        return None

    exps = list_expiries(sym)
    if not exps:
        return None
    exps = sorted([e for e in exps if days_to(e) <= TIERING_MAX_DTE], key=lambda x: days_to(x))[:TIERING_MAX_EXPIRIES]
    if not exps:
        return None

    k_lo = spot * (1 - TIERING_STRIKE_BAND)
    k_hi = spot * (1 + TIERING_STRIKE_BAND)

    spread_fracs: List[float] = []
    vols: List[float] = []
    ois: List[float] = []
    obs = 0

    for exp in exps:
        rows = option_chain(sym, exp)
        if not rows:
            continue
        for r in rows:
            K = safe_float(r.get("strike"), 0.0)
            if K <= 0 or K < k_lo or K > k_hi:
                continue
            typ = str(r.get("option_type", r.get("type", ""))).lower()
            if typ not in ("call", "put"):
                continue
            bid = safe_float(r.get("bid"), 0.0)
            ask = safe_float(r.get("ask"), 0.0)
            if bid <= 0 or ask <= 0 or ask <= bid:
                continue
            m = (bid + ask) / 2.0
            if m <= 0:
                continue
            spread = ask - bid
            spread_frac = spread / m
            spread_fracs.append(spread_frac)

            vols.append(safe_float(r.get("volume"), 0.0))
            ois.append(safe_float(r.get("open_interest"), 0.0))
            obs += 1

    if obs < TIERING_MIN_OBS or not spread_fracs:
        return None

    avg_spread_frac = sum(spread_fracs) / len(spread_fracs)
    med_vol = median(vols) if vols else 0.0
    med_oi  = median(ois)  if ois  else 0.0

    return {
        "avg_spread_frac": float(avg_spread_frac),
        "med_volume": float(med_vol),
        "med_open_interest": float(med_oi),
        "observations": obs,
    }

def _tiering_score(metrics: Dict[str, float]) -> float:
    sf = metrics.get("avg_spread_frac", 1.0)
    vol = metrics.get("med_volume", 0.0)
    oi  = metrics.get("med_open_interest", 0.0)

    # Spread: sf=0.00→1.0, sf=0.30→0.0
    spread_score = max(0.0, min(1.0, (0.30 - sf) / 0.30))

    def squash(x: float, k: float = 2000.0) -> float:
        return max(0.0, min(1.0, x / (x + k)))

    vol_score = squash(vol, k=2000.0)
    oi_score  = squash(oi,  k=3000.0)

    w_spread, w_vol, w_oi = 0.50, 0.30, 0.20
    score = w_spread * spread_score + w_vol * vol_score + w_oi * oi_score
    return float(max(0.0, min(1.0, score)))

def _score_to_tier(score: float) -> int:
    if score >= 0.80: return 1
    if score >= 0.60: return 2
    if score >= 0.40: return 3
    return 4

def rebuild_dynamic_tiers(symbols: List[str]) -> Dict[str, int]:
    tier_map: Dict[str, int] = {}
    for sym in symbols:
        try:
            m = _tiering_collect_metrics(sym)
            if m:
                s = _tiering_score(m)
                tier_map[sym.upper()] = _score_to_tier(s)
            else:
                tier_map[sym.upper()] = TIER_MAP.get(sym.upper(), 4)
        except Exception:
            tier_map[sym.upper()] = TIER_MAP.get(sym.upper(), 4)
        time.sleep(0.10)
    return tier_map

def maybe_refresh_dynamic_tiers():
    global _dynamic_tier_map, _dynamic_tier_ts
    if not DYNAMIC_TIERING:
        return
    now = time.time()
    if (now - _dynamic_tier_ts) < TIER_CACHE_TTL_SEC and _dynamic_tier_map:
        return
    universe = (TIERING_UNIVERSE[:MAX_UNDERLYINGS]
                if TIERING_UNIVERSE else SEED_UNDERLYINGS[:MAX_UNDERLYINGS])
    print(f"{now_str()} — Refreshing dynamic tiers for {len(universe)} symbols…")
    _dynamic_tier_map = rebuild_dynamic_tiers(universe)
    _dynamic_tier_ts = now
    for a in TIER1_ANCHORS:
        _dynamic_tier_map[a.upper()] = 1
    buckets = {1:0,2:0,3:0,4:0}
    for _, t in _dynamic_tier_map.items():
        buckets[t] = buckets.get(t,0)+1
    print(f"{now_str()} — Tier counts: T1={buckets[1]} T2={buckets[2]} T3={buckets[3]} T4={buckets[4]}")

def tier_for(symbol: str) -> int:
    sym = (symbol or "").upper()
    if sym in {s.upper() for s in TIER1_ANCHORS}:
        return 1
    if DYNAMIC_TIERING and _dynamic_tier_map:
        return _dynamic_tier_map.get(sym, TIER_MAP.get(sym, 4))
    return TIER_MAP.get(sym, 4)

# ===================== FILL PROBABILITY & SIZING =====================

def fill_probability(h: Hit) -> float:
    tier = tier_for(h.base)
    base = TIER_BASE_PROB.get(tier, 0.40)

    edge_dollars   = abs(h.gap) * 100.0
    spread_dollars = (h.call_spread + h.put_spread) * 100.0
    edge_vs_spread = clamp(edge_dollars / max(spread_dollars, 1e-6), 0.0, 2.0) / 2.0

    dte_norm = clamp(1.0 - (min(h.dte_days, 10) / 20.0), 0.5, 1.0)

    spread_score = clamp(1.0 - (h.spread_frac / max(SPREAD_MAX, 1e-9)), 0.0, 1.0)

    w = FILL_WEIGHTS
    prob = (
        w["tier"]           * base +
        w["edge_vs_spread"] * edge_vs_spread +
        w["dte"]            * dte_norm +
        w["spread_frac"]    * spread_score
    )
    return clamp(prob)

def contracts_fit_dual(h: Hit) -> Tuple[int, float, float]:
    if not USE_DUAL_CAPS:
        cap = h.cap_margin_ct if USE_MARGIN_FOR_SIZE else h.cap_cash_ct
        n   = contracts_fit(ACCOUNT_BUDGET_USD, cap)
        used_cash = cap * n if not USE_MARGIN_FOR_SIZE else 0.0
        used_marg = cap * n if USE_MARGIN_FOR_SIZE else 0.0
        return n, used_cash, used_marg

    if h.strategy == "conversion":
        n = contracts_fit(CASH_CAP_USD, h.cap_cash_ct)
        return n, h.cap_cash_ct * n, 0.0
    else:
        n = contracts_fit(MARGIN_CAP_USD, h.cap_margin_ct)
        return n, 0.0, h.cap_margin_ct * n

# ===================== CORE LOGIC =====================

def compute_hits_for_symbol(sym: str) -> List[Hit]:
    hits: List[Hit] = []
    spot = quote(sym)
    if not spot:
        return hits

    exps = list_expiries(sym)[:MAX_EXPIRIES]
    exps = [e for e in exps if days_to(e) <= MAX_DTE]
    if not exps:
        return hits

    k_lo = spot * (1 - BAND_PCT)
    k_hi = spot * (1 + BAND_PCT)

    for exp in exps:
        rows = option_chain(sym, exp)
        if not rows:
            continue

        by_k: Dict[float, Dict[str, Dict]] = {}
        for r in rows:
            K = safe_float(r.get("strike"))
            if K <= 0 or K < k_lo or K > k_hi:
                continue

            typ = str(r.get("option_type", r.get("type", ""))).lower()
            b = safe_float(r.get("bid"), 0.0)
            a = safe_float(r.get("ask"), 0.0)
            m = mid(b, a)
            if not m:
                continue

            # Basic liquidity
            oi  = safe_float(r.get("open_interest"), 0.0)
            vol = safe_float(r.get("volume"), 0.0)
            if oi < MIN_OI or vol < MIN_VOL:
                log_skip(sym, exp, K, reason="liquidity_oi_vol", oi=oi, vol=vol)
                continue

            by_k.setdefault(K, {})
            by_k[K][typ] = {"bid": b, "ask": a, "mid": m}

        dte = days_to(exp)
        for K, pair in by_k.items():
            if "call" not in pair or "put" not in pair:
                continue
            c = pair["call"]; p = pair["put"]
            call_mid = c["mid"];  put_mid = p["mid"]

            call_spread = max(0.0, c["ask"] - c["bid"])
            put_spread  = max(0.0, p["ask"] - p["bid"])
            call_spread_frac = call_spread / call_mid if call_mid > 0 else 1.0
            put_spread_frac  = put_spread  / put_mid  if put_mid  > 0 else 1.0
            spread_frac = max(call_spread_frac, put_spread_frac)
            if spread_frac > SPREAD_MAX:
                log_skip(sym, exp, K, reason="spread_wide", spread_frac=spread_frac)
                continue

            pvK = discount_strike(K, dte)
            gap = call_mid - put_mid - (spot - pvK)   # $/share
            if abs(gap) < PARITY_THRESH:
                log_skip(sym, exp, K, reason="thin_edge", gap=gap)
                continue

            # --- slippage/fees ---
            slip_cost  = SLIP_FRAC_OF_SPREAD * (call_spread + put_spread) * 100.0
            stock_slip = STOCK_SLIP_CENTS * 100.0
            fees       = OPTION_FEE_PER_CT * 2 + STOCK_FEE_PER_TRADE

            # --- borrow handling ---
            borrow_rate = get_borrow_rate(sym)  # may be None

            # Require borrow if configured (reversals only)
            if borrow_rate is None and REQUIRE_BORROW_FOR_REVERSAL and gap < 0:
                log_skip(sym, exp, K, reason="borrow_unknown", borrow_required=True)
                continue

            # If known and capped (reversals only)
            if (borrow_rate is not None) and (BORROW_RATE_CAP is not None) and (gap < 0):
                if borrow_rate > BORROW_RATE_CAP:
                    log_skip(sym, exp, K, reason="borrow_too_high", borrow_rate=borrow_rate)
                    continue

            br_eff  = norm_borrow_rate(borrow_rate)
            carry_T = max(0, dte) / 365.0
            borrow_cost_per_ct = (br_eff * carry_T * spot * 100.0) if gap < 0 else 0.0

            net_per_ct = (gap * 100.0) - slip_cost - stock_slip - fees - borrow_cost_per_ct

            hits.append(Hit(
                timestamp=now_str(),
                base=sym,
                expiry=exp,
                strike=K,
                spot=spot,
                call_mid=call_mid,
                put_mid=put_mid,
                call_spread=call_spread,
                put_spread=put_spread,
                gap=gap,
                strategy="conversion" if gap >= 0 else "reversal",
                spread_frac=spread_frac,
                cap_cash_ct=per_contract_cap(spot, use_margin=False),
                cap_margin_ct=per_contract_cap(spot, use_margin=True),
                net_per_ct=net_per_ct,
                dte_days=dte
            ))
    return hits

def discover_universe() -> List[str]:
    base = TIERING_UNIVERSE if TIERING_UNIVERSE else SEED_UNDERLYINGS
    seen = set()
    universe: List[str] = []
    # anchors first
    for s in TIER1_ANCHORS:
        u = s.upper()
        if u not in seen:
            universe.append(u); seen.add(u)
    # then the rest
    for s in base:
        u = s.upper()
        if u not in seen:
            universe.append(u); seen.add(u)
        if len(universe) >= MAX_UNDERLYINGS:
            break
    return universe[:MAX_UNDERLYINGS]

# ===================== ENRICHMENT, PRIORITY & PRINT =====================

def enrich_hits_with_metrics(hits: List[Hit], budget_usd: float, use_margin_for_size: bool) -> List[Hit]:
    if not hits:
        return hits

    kept: List[Hit] = []
    for h in hits:
        n_ct, cash_used, marg_used = contracts_fit_dual(h)
        setattr(h, "contracts_fit", n_ct)
        setattr(h, "cash_used", cash_used)
        setattr(h, "margin_used", marg_used)

        total  = h.net_per_ct * n_ct
        tier   = tier_for(h.base)
        tscore = TIER_SAFETY_SCORE.get(tier, 0.40)

        prob = fill_probability(h)
        setattr(h, "fill_prob", prob)

        denom = (cash_used + marg_used) if USE_DUAL_CAPS else budget_usd
        roi_pct = (total / denom * 100.0) if denom > 0 else 0.0

        setattr(h, "total_net", total)
        setattr(h, "roi_pct", roi_pct)
        setattr(h, "tier_num", tier)
        setattr(h, "tier_safety", tscore)

        # Gates
        if tier not in TARGET_TIERS:
            continue
        if ONLY_PROFITABLE and h.net_per_ct <= 0:
            continue
        if prob < MIN_FILL_PROB:
            continue
        if (MIN_ROI_BPS > 0) and (roi_pct * 100.0 < MIN_ROI_BPS):
            continue
        if (MIN_TOTAL_NET_USD > 0) and (total < MIN_TOTAL_NET_USD):
            continue
        if n_ct <= 0:
            continue

        kept.append(h)

    if not kept:
        return kept

    max_roi = max((getattr(h, "roi_pct", 0.0) for h in kept), default=0.0); max_roi = max(max_roi, 1e-9)
    max_net = max((getattr(h, "total_net", 0.0) for h in kept), default=0.0); max_net = max(max_net, 1e-9)

    w_roi, w_net, w_tier = PRIORITY_WEIGHTS["roi"], PRIORITY_WEIGHTS["total_net"], PRIORITY_WEIGHTS["tier"]
    for h in kept:
        roi_norm = getattr(h, "roi_pct", 0.0) / max_roi
        net_norm = getattr(h, "total_net", 0.0) / max_net
        tier_sc  = getattr(h, "tier_safety", 0.0)
        priority = (w_roi * roi_norm) + (w_net * net_norm) + (w_tier * tier_sc)
        setattr(h, "priority_score", priority)

    hits_sorted_by_pri = sorted(kept, key=lambda x: getattr(x, "priority_score", 0.0), reverse=True)
    for i, h in enumerate(hits_sorted_by_pri):
        setattr(h, "pick", "P1" if i == 0 else "")
        setattr(h, "highlight", "Y" if i == 0 else "")
    return kept

def format_hit_line(h: Hit) -> str:
    n_ct   = getattr(h, "contracts_fit", 0)
    total  = getattr(h, "total_net", 0.0)
    roi    = getattr(h, "roi_pct", 0.0)
    tier   = getattr(h, "tier_num", tier_for(h.base))
    prob   = getattr(h, "fill_prob", 0.0)
    pick   = getattr(h, "pick", "")

    if VERBOSE_OUTPUT:
        cap_used_str = f"cash_used=${getattr(h,'cash_used',0.0):,.0f}, margin_used=${getattr(h,'margin_used',0.0):,.0f}"
        return (f"{h.timestamp} — {h.base} {h.expiry} K={h.strike:.2f} | spot={h.spot:.2f} "
                f"| gap={h.gap:.2f}/sh → {h.strategy} | spread={h.spread_frac*100:.1f}% "
                f"| contracts={n_ct} | net/ct=${h.net_per_ct:.2f} total_net=${total:.2f} "
                f"| ROI={roi:.2f}% | Tier={tier} | Prob={prob:.3f} | {cap_used_str} {pick}")
    else:
        if DISPLAY_ROI_IN_BPS:
            roi_bps   = roi * 100.0
            naive_ann = roi * (365.0 / max(1, ASSUMED_CYCLE_DAYS))
            return (f"{h.timestamp} | {h.base} {h.expiry} K={h.strike:.2f} "
                    f"| gap=${h.gap:.2f}/sh | contracts={n_ct} "
                    f"| total_net=${total:.2f} | ROI={roi_bps:.1f} bps (~{naive_ann:.2f}%/yr) "
                    f"| Tier={tier} | Prob={prob:.3f} {pick}")
        else:
            return (f"{h.timestamp} | {h.base} {h.expiry} K={h.strike:.2f} "
                    f"| gap=${h.gap:.2f}/sh | contracts={n_ct} "
                    f"| total_net=${total:.2f} | ROI={roi:.2f}% | Tier={tier} | Prob={prob:.3f} {pick}")

# ===================== CSV (top hits) =====================

def write_hits_csv(rows: List[Hit]):
    if not WRITE_CSV or not rows:
        return
    newfile = not os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if newfile:
            w.writerow([
                "timestamp","underlying","expiry","strike","call_mid","put_mid","spot",
                "parity_gap","strategy","spread_frac",
                "call_spread","put_spread",
                "cap_cash_per_ct","cap_margin_per_ct","net_usd_per_ct",
                "tier","contracts_fit","total_net","roi_pct",
                "fill_prob","priority_score",
                "cash_used","margin_used","highlight","pick"
            ])
        for h in rows:
            w.writerow([
                h.timestamp, h.base, h.expiry, f"{h.strike:.2f}", f"{h.call_mid:.4f}", f"{h.put_mid:.4f}", f"{h.spot:.4f}",
                f"{h.gap:.4f}", h.strategy, f"{h.spread_frac:.4f}",
                f"{h.call_spread:.4f}", f"{h.put_spread:.4f}",
                f"{h.cap_cash_ct:.2f}", f"{h.cap_margin_ct:.2f}", f"{h.net_per_ct:.2f}",
                getattr(h, "tier_num", tier_for(h.base)),
                getattr(h, "contracts_fit", 0),
                f"{getattr(h, 'total_net', 0.0):.2f}",
                f"{getattr(h, 'roi_pct', 0.0):.2f}",
                f"{getattr(h, 'fill_prob', 0.0):.3f}",
                f"{getattr(h, 'priority_score', 0.0):.4f}",
                f"{getattr(h, 'cash_used', 0.0):.2f}",
                f"{getattr(h, 'margin_used', 0.0):.2f}",
                getattr(h, "highlight", ""),
                getattr(h, "pick", "")
            ])

# ===================== MAIN LOOP =====================

def scan_once():
    maybe_refresh_dynamic_tiers()

    syms = discover_universe()
    all_hits: List[Hit] = []
    for sym in syms:
        try:
            all_hits.extend(compute_hits_for_symbol(sym))
        except Exception as e:
            import traceback
            print("scan_once error:", e)
            traceback.print_exc()

    all_hits = enrich_hits_with_metrics(all_hits, ACCOUNT_BUDGET_USD, USE_MARGIN_FOR_SIZE)
    all_hits.sort(key=lambda h: getattr(h, "priority_score", 0.0), reverse=True)

    cap_line = (f"caps: cash ${CASH_CAP_USD:,.0f}, margin ${MARGIN_CAP_USD:,.0f}"
                if USE_DUAL_CAPS else
                f"budget=${ACCOUNT_BUDGET_USD:,.0f}; margin {'on' if USE_MARGIN_FOR_SIZE else 'off'}")

    print(
        f"{now_str()} — Top {min(TOP_N, len(all_hits))} parity dislocations "
        f"(host={'sandbox' if USE_SANDBOX else 'prod'}; DTE ≤ {MAX_DTE}d; "
        f"band ±{int(BAND_PCT*100)}%; spread ≤{int(SPREAD_MAX*100)}%; "
        f"{cap_line}; tiers={TARGET_TIERS}; min_prob={MIN_FILL_PROB:.2f}; "
        f"dynamic_tiering={'ON' if DYNAMIC_TIERING else 'OFF'} (TTL={TIER_CACHE_TTL_SEC//60}m))"
    )
    if all_hits:
        for h in all_hits[:TOP_N]:
            print(format_hit_line(h))
    else:
        print(f"{now_str()} — No qualified dislocations (after filters).")

    write_hits_csv(all_hits[:TOP_N])
    write_audit_csv()

def main():
    maybe_refresh_dynamic_tiers()

    cap_line = (f"caps: cash ${CASH_CAP_USD:,.0f}, margin ${MARGIN_CAP_USD:,.0f}"
                if USE_DUAL_CAPS else
                f"budget=${ACCOUNT_BUDGET_USD:,.0f} | margin={'on' if USE_MARGIN_FOR_SIZE else 'off'}")
    print(
        f"🚀 Options parity scanner | refresh {REFRESH_SECONDS}s | band ±{int(BAND_PCT*100)}% "
        f"| spread≤{int(SPREAD_MAX*100)}% | {'SANDBOX' if USE_SANDBOX else 'PRODUCTION'} | {cap_line} "
        f"| DTE ≤ {MAX_DTE}d | slip={int(SLIP_FRAC_OF_SPREAD*100)}% of spread/leg, stock {STOCK_SLIP_CENTS:.02f}$/sh "
        f"| ROI display={'bps' if DISPLAY_ROI_IN_BPS else '%'} | tiers={TARGET_TIERS} | min_prob={MIN_FILL_PROB:.2f} "
        f"| dynamic_tiering={'ON' if DYNAMIC_TIERING else 'OFF'} (TTL={TIER_CACHE_TTL_SEC//60}m) "
        f"| anchors={len(TIER1_ANCHORS)} Tier-1 pinned | margin_safety×{MARGIN_SAFETY:.2f}"
    )
    try:
        while True:
            if MARKET_HOURS_ONLY and not _is_market_session(_now_et()):
                _sleep_until_next_session()
                continue

            try:
                scan_once()
            except Exception as e:
                import traceback
                print("scan_once error:", e)
                traceback.print_exc()

            time.sleep(REFRESH_SECONDS)
    except KeyboardInterrupt:
        print("Bye!")

if __name__ == "__main__":
    main()
