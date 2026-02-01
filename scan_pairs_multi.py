#!/usr/bin/env python3
"""
WORKFLOW 1 – MULTI-BASKET PAIR SCANNER (RETAIL)

Cel:
- wygenerować WIĘCEJ sensownych par (8–15+), nie tylko 1–2
- luźniejsze progi, ale nadal kontrola jakości
- brak stabilizacji / histerezy (to jest Workflow 2)

Kryteria (świadomie poluzowane):
- min wspólnych obserwacji: 400
- Pearson corr (returns, 252): >= 0.70
- Engle-Granger p-value: <= 0.10
- Half-life: <= 60 dni
- Avg |Z| (60): >= 1.5
"""

import os
import itertools
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from datetime import datetime

# ================= CONFIG =================
START_DATE = "2018-01-01"
MIN_SAMPLE = 400

CORR_MIN = 0.70
COINT_P_MAX = 0.10
HALF_LIFE_MAX = 60
AVG_Z_MIN = 1.5

BASKETS = {
    "tech_core": "data/tickers_tech_core.txt",
    "semis": "data/tickers_semis.txt",
    "software": "data/tickers_software.txt",
    "financials": "data/tickers_financials.txt",
    "healthcare": "data/tickers_healthcare.txt",
    "discretionary": "data/tickers_discretionary.txt",
}

OUT_DIR = "results_workflow1"
# ==========================================


def load_tickers(path):
    with open(path, "r") as f:
        return sorted(list(set([x.strip() for x in f if x.strip()])))


def download_prices(tickers):
    df = yf.download(
        tickers,
        start=START_DATE,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    prices = {}
    for t in tickers:
        try:
            if isinstance(df.columns, pd.MultiIndex):
                prices[t] = df[t]["Close"]
            else:
                prices[t] = df["Close"]
        except Exception:
            continue

    return pd.DataFrame(prices).dropna(how="all")


def log_returns(df):
    return np.log(df / df.shift(1)).dropna()


def half_life(spread):
    s = spread.dropna()
    if len(s) < 50:
        return np.nan

    lag = s.shift(1).iloc[1:]
    delta = s.diff().iloc[1:]

    model = OLS(delta, add_constant(lag)).fit()
    rho = model.params[1]

    if abs(rho) < 1e-6 or (1 + rho) <= 0:
        return np.nan

    return -np.log(2) / np.log(1 + rho)


def scan_basket(name, tickers):
    print(f"\n=== SCANNING BASKET: {name} ({len(tickers)} tickers) ===")

    prices = download_prices(tickers)
    if prices.shape[1] < 2:
        return pd.DataFrame()

    rets = log_returns(prices)
    results = []

    for y, x in itertools.combinations(prices.columns, 2):
        px = prices[[y, x]].dropna()
        if len(px) < MIN_SAMPLE:
            continue

        r = rets[[y, x]].dropna()
        if len(r) < MIN_SAMPLE:
            continue

        corr = r[y].rolling(252).corr(r[x]).iloc[-1]
        if corr < CORR_MIN:
            continue

        score, pval, _ = coint(px[y], px[x])
        if pval > COINT_P_MAX:
            continue

        model = OLS(px[y], add_constant(px[x])).fit()
        beta = model.params[1]
        spread = px[y] - beta * px[x]

        hl = half_life(spread)
        if np.isnan(hl) or hl > HALF_LIFE_MAX:
            continue

        z60 = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
        avg_z = z60.abs().mean()

        if avg_z < AVG_Z_MIN:
            continue

        results.append({
            "basket": name,
            "y": y,
            "x": x,
            "corr_252": round(corr, 3),
            "coint_p": round(pval, 4),
            "beta": round(beta, 3),
            "half_life": round(hl, 1),
            "avg_abs_z60": round(avg_z, 2),
            "obs": len(px),
        })

    return pd.DataFrame(results)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = []

    for basket, path in BASKETS.items():
        tickers = load_tickers(path)
        df = scan_basket(basket, tickers)

        if not df.empty:
            out_path = f"{OUT_DIR}/pairs_{basket}.csv"
            df.sort_values("avg_abs_z60", ascending=False).to_csv(out_path, index=False)
            all_results.append(df)
            print(f"Saved: {out_path} ({len(df)} pairs)")
        else:
            print("No pairs found.")

    if all_results:
        summary = pd.concat(all_results).sort_values(
            ["avg_abs_z60", "half_life"],
            ascending=[False, True]
        )
        summary_path = f"{OUT_DIR}/pairs_summary_all_baskets.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\n=== SUMMARY SAVED: {summary_path} ({len(summary)} pairs) ===")
    else:
        print("\nNo pairs found in any basket.")


if __name__ == "__main__":
    main()
