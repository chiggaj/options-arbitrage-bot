#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze options parity scanner output (options_parity_hits.csv) and
tag each hit with a Tier (1–4) + Priority score.

What this script does:
- Robustly loads the CSV (sniffs delimiter, skips malformed rows)
- If no "net" column exists, derives net_usd = parity_gap × 100 (per-contract $)
- Adds `tier` for each underlying via TIER_MAP and `priority_score`
- Writes:
  - analysis_report.xlsx (with Tier sheets)
  - charts/ PNGs
  - profitable_only.csv
  - options_parity_hits_tiered.csv (original rows + tier + priority_score)

Usage:
    python analyze_parity_csv.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_IN = "options_parity_hits.csv"
CSV_OUT_TIERED = "options_parity_hits_tiered.csv"
EXCEL_OUT = "analysis_report.xlsx"
CHART_DIR = "charts"

# ------------------------ Tier config ------------------------
# You can edit this mapping as you like.
TIER_MAP = {
    # Tier 1 — Ultra-liquid ETFs & mega caps
    "SPY": 1, "QQQ": 1, "AAPL": 1, "MSFT": 1,
    # Tier 2 — Very attractive
    "NVDA": 2, "AMZN": 2, "META": 2,
    # Tier 3 — Attractive but spikier
    "TSLA": 3, "AMD": 3, "GOOGL": 3,
    # Tier 4 — Optional / opportunistic
    "IWM": 4, "DIA": 4, "NFLX": 4, "BABA": 4, "INTC": 4,
}

# Weight per tier (higher = more priority)
TIER_WEIGHT = {1: 1.00, 2: 0.85, 3: 0.70, 4: 0.50}

# Quality factor from the signal itself (parity gap),
# squashed to avoid giant outliers dominating priority.
def _gap_quality(parity_gap_abs: pd.Series) -> pd.Series:
    # Simple diminishing returns: q = min(1.0, |gap| / 1.00)
    # i.e., $1.00/share gap or more scores full 1.0; smaller gaps proportionally less.
    return np.minimum(1.0, parity_gap_abs.fillna(0.0).astype(float) / 1.00)

# ------------------------ helpers ------------------------

def _clean_numeric(s):
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).replace("$", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return np.nan

# ------------------------ loader ------------------------

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input CSV not found at {os.path.abspath(path)}")

    df = None
    # 1) Try fast C-engine
    try:
        df = pd.read_csv(path)
    except Exception:
        print("Standard CSV parse failed, retrying with tolerant settings…")

    # 2) If not parsed (or one column), use python engine and sniff delimiter
    if df is None or df.shape[1] == 1:
        try:
            df = pd.read_csv(
                path,
                engine="python",
                sep=None,                # sniff
                quotechar='"',
                escapechar="\\",
                on_bad_lines="skip",
                encoding_errors="ignore"
            )
            if df.shape[1] == 1:
                print("Delimiter sniffing produced 1 column; forcing sep='|'…")
                df = pd.read_csv(
                    path, engine="python", sep="|",
                    on_bad_lines="skip", encoding_errors="ignore"
                )
        except Exception:
            print("Tolerant parse failed; forcing sep=',' with python engine…")
            df = pd.read_csv(
                path, engine="python", sep=",",
                quotechar='"', escapechar="\\",
                on_bad_lines="skip", encoding_errors="ignore"
            )

    # Normalize columns
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    print("Columns detected:", list(df.columns))

    # Parse timestamp (best effort)
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        except Exception:
            pass

    # Coerce numerics where present
    numeric_cols = [
        "dte_days","strike","spot",
        "call_bid","call_ask","put_bid","put_ask","call_mid","put_mid",
        "parity_gap","pnl_per_contract_gross",
        "capital_cash_per_ct","capital_margin_per_ct",
        "budget_usd",
        "cash_contracts","cash_gross","cash_slip","cash_fees","cash_carry","cash_net","cash_roi_pct",
        "margin_contracts","margin_gross","margin_slip","margin_fees","margin_carry","margin_net","margin_roi_pct",
        "slippage_frac_options","stock_slippage_cents","option_fee_per_contract","stock_fee_per_trade",
        "margin_rate","short_borrow_rate","margin_fraction",
        "min_net_usd","min_roi_pct",
        # common variants
        "net","net_$","net_usd","net_per_contract","net_per_ct","net_total","net_usd_ct",
        "gross","slip","fees","carry"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(_clean_numeric)

    # Infer side from parity_gap sign if not present
    if "side" not in df.columns and "parity_gap" in df.columns:
        df["side"] = np.where(df["parity_gap"] >= 0, "conversion", "reversal")

    # --- Choose/Create net_usd ---
    net_candidates_in_order = [
        "margin_net", "cash_net",
        "net_usd", "net_$", "net_total", "net",
        "net_usd_ct", "net_per_contract", "net_per_ct"
    ]
    chosen = None
    for c in net_candidates_in_order:
        if c in df.columns and df[c].notna().any():
            chosen = c
            break

    if chosen is not None:
        # Scale per-contract metrics by contracts if available
        if chosen in ("net_usd_ct", "net_per_contract", "net_per_ct"):
            contracts_col = None
            for k in ("margin_contracts", "cash_contracts", "contracts", "num_contracts"):
                if k in df.columns:
                    contracts_col = k
                    break
            if contracts_col:
                df["net_usd"] = df[chosen] * df[contracts_col]
            else:
                df["net_usd"] = df[chosen]
    else:
        # No net column -> derive from parity_gap × 100 (per contract)
        if "parity_gap" in df.columns:
            df["net_usd"] = df["parity_gap"].astype(float) * 100.0
            print("Derived net_usd from parity_gap × 100 (per-contract $).")
        else:
            df["net_usd"] = np.nan

    # DTE buckets
    if "dte_days" in df.columns:
        bins = [-1,0,1,2,3,7,30,365]
        labels = ["0d","1d","2d","3d","<1w","<1m","long"]
        df["dte_bucket"] = pd.cut(df["dte_days"].fillna(-1), bins=bins, labels=labels)

    # Ensure key categoricals
    if "underlying" not in df.columns:
        df["underlying"] = "UNKNOWN"
    if "expiry" not in df.columns:
        df["expiry"] = ""

    return df

# ------------------------ tier & priority ------------------------

def tag_tier_and_priority(df: pd.DataFrame) -> pd.DataFrame:
    # Map to Tier
    df["tier"] = df.get("underlying", "UNKNOWN").map(lambda x: TIER_MAP.get(str(x).upper(), 4))

    # Compute basic priority score = TierWeight * GapQuality
    # (You can extend with liquidity/volume depth if those cols exist.)
    gap_abs = df.get("parity_gap", pd.Series([np.nan]*len(df))).abs()
    q = _gap_quality(gap_abs)
    w = df["tier"].map(TIER_WEIGHT).fillna(0.5)
    df["priority_score"] = (w * q).round(4)

    return df

# ------------------------ charts ------------------------

def ensure_chart_dir():
    os.makedirs(CHART_DIR, exist_ok=True)

def chart_hist_net(df: pd.DataFrame):
    ensure_chart_dir()
    plt.figure()
    clean = df["net_usd"].dropna()
    plt.hist(clean, bins=50)
    plt.title("Net Profit Per Hit (Histogram)")
    plt.xlabel("Net $ per opportunity (per contract)")
    plt.ylabel("Count")
    out = os.path.join(CHART_DIR, "hist_net_usd.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out

def chart_top_underlyings(df: pd.DataFrame, topn: int = 12):
    ensure_chart_dir()
    grp = df.groupby("underlying")["net_usd"].sum().sort_values(ascending=False).head(topn)
    plt.figure()
    grp.plot(kind="bar")
    plt.title(f"Top {topn} Underlyings by Total Net $")
    plt.xlabel("Underlying")
    plt.ylabel("Total Net $ (per contract basis)")
    out = os.path.join(CHART_DIR, "top_underlyings_total_net.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out

def chart_cum_pnl(df: pd.DataFrame):
    ensure_chart_dir()
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        ts_df = df.dropna(subset=["timestamp"]).sort_values("timestamp").copy()
        ts_df["cum_net"] = ts_df["net_usd"].fillna(0).cumsum()
        plt.figure()
        plt.plot(ts_df["timestamp"], ts_df["cum_net"])
        plt.title("Cumulative Net Profit Over Time (per contract basis)")
        plt.xlabel("Time")
        plt.ylabel("Cumulative Net $")
        out = os.path.join(CHART_DIR, "cumulative_net_over_time.png")
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        return out
    return None

def chart_gap_vs_net(df: pd.DataFrame):
    ensure_chart_dir()
    if "parity_gap" in df.columns:
        plt.figure()
        plt.scatter(df["parity_gap"], df["net_usd"])
        plt.title("Parity Gap ($/sh) vs Net $ (per contract)")
        plt.xlabel("Parity Gap ($/share)")
        plt.ylabel("Net $ per contract")
        out = os.path.join(CHART_DIR, "gap_vs_net.png")
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        return out
    return None

# ------------------------ excel report ------------------------

def make_excel_report(df: pd.DataFrame, path: str):
    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        df.to_excel(xw, sheet_name="raw", index=False)

        prof = df[df["net_usd"] > 0].copy()
        prof.to_excel(xw, sheet_name="profitable_only", index=False)

        summary = pd.DataFrame({
            "total_rows": [len(df)],
            "profitable_rows": [len(prof)],
            "win_rate_%": [100.0 * len(prof) / max(1, len(df))],
            "total_net_$": [df["net_usd"].sum()],
            "avg_net_per_hit_$": [df["net_usd"].mean()],
            "median_net_per_hit_$": [df["net_usd"].median()],
        })
        summary.to_excel(xw, sheet_name="summary", index=False)

        by_ul = (df.groupby("underlying")
                 .agg(total_net_usd=("net_usd","sum"),
                      avg_net_usd=("net_usd","mean"),
                      median_net_usd=("net_usd","median"),
                      hits=("net_usd","count"))
                 .sort_values("total_net_usd", ascending=False))
        by_ul.to_excel(xw, sheet_name="by_underlying")

        if "side" in df.columns:
            by_side = (df.groupby("side")
                       .agg(total_net_usd=("net_usd","sum"),
                            avg_net_usd=("net_usd","mean"),
                            hits=("net_usd","count"))
                       .sort_values("total_net_usd", ascending=False))
            by_side.to_excel(xw, sheet_name="by_side")

        if "dte_bucket" in df.columns:
            by_dte = (df.groupby("dte_bucket")
                      .agg(total_net_usd=("net_usd","sum"),
                           avg_net_usd=("net_usd","mean"),
                           hits=("net_usd","count"))
                      .sort_values("total_net_usd", ascending=False))
            by_dte.to_excel(xw, sheet_name="by_dte")

        # New: by Tier and Top by priority
        if "tier" in df.columns:
            by_tier = (df.groupby("tier")
                       .agg(total_net_usd=("net_usd","sum"),
                            avg_net_usd=("net_usd","mean"),
                            hits=("net_usd","count"),
                            avg_priority=("priority_score","mean"))
                       .sort_values("total_net_usd", ascending=False))
            by_tier.to_excel(xw, sheet_name="by_tier")

            # Top 100 by priority_score
            top_priority = df.sort_values("priority_score", ascending=False).head(100)
            top_priority.to_excel(xw, sheet_name="top100_priority", index=False)

# ------------------------ main ------------------------

def main():
    print("Loading:", CSV_IN)
    df = load_data(CSV_IN)

    # Tag Tier & Priority
    df = tag_tier_and_priority(df)

    total = len(df)
    prof = (df["net_usd"] > 0).sum()
    print(f"Rows: {total:,} | Profitable: {prof:,} ({(100.0*prof/max(1,total)):.1f}%)")
    print(f"Total Net $ (per-contract basis): {df['net_usd'].sum():,.2f}")
    print(f"Avg Net $/hit: {df['net_usd'].mean():.2f} | Median: {df['net_usd'].median():.2f}")

    # Save enriched CSV with tiers
    df.to_csv(CSV_OUT_TIERED, index=False)
    print("Wrote tiered CSV:", CSV_OUT_TIERED)

    # Save profitable-only CSV
    df_prof = df[df["net_usd"] > 0].copy()
    df_prof.to_csv("profitable_only.csv", index=False)
    print("Wrote profitable_only.csv")

    # Charts
    os.makedirs(CHART_DIR, exist_ok=True)
    p1 = chart_hist_net(df)
    p2 = chart_top_underlyings(df)
    p3 = chart_cum_pnl(df)
    p4 = chart_gap_vs_net(df)
    print("Charts saved to:", CHART_DIR)

    # Excel report
    make_excel_report(df, EXCEL_OUT)
    print("Wrote Excel report:", EXCEL_OUT)

if __name__ == "__main__":
    main()
