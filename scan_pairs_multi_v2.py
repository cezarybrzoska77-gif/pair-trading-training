#!/usr/bin/env python3
"""
WORKFLOW 1 v2 — STRUCTURAL PAIR SCANNER (SOFT)

Cel:
- Zawsze generować 8–15 sensownych par
- Bez entry, bez twardej kointegracji
- Soft ADF / Half-life jako scoring
"""

import argparse
import itertools
import os
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

# -------------------- helpers --------------------

def download_prices(tickers, start):
    df = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        threads=False,
        group_by="ticker"
    )
    prices = {}
    for t in tickers:
        try:
            s = df[t]["Close"].dropna()
            if len(s) > 500:
                prices[t] = s
        except Exception:
            continue
    return prices


def log_returns(s):
    return np.log(s).diff().dropna()


def beta_price(y, x):
    df = pd.concat([y, x], axis=1).dropna()
    X = sm.add_constant(df.iloc[:, 1])
    model = sm.OLS(df.iloc[:, 0], X).fit()
    return model.params.iloc[1]


def half_life(spread):
    s = spread.dropna()
    if len(s) < 100:
        return np.nan
    lag = s.shift(1).dropna()
    delta = s.diff().dropna()
    lag = lag.loc[delta.index]
    X = sm.add_constant(lag)
    model = sm.OLS(delta, X).fit()
    rho = model.params.iloc[1]
    if abs(rho) < 1e-4 or (1 + rho) <= 0:
        return np.nan
    return -np.log(2) / np.log(1 + rho)


def score_corr(c):
    if c < 0.65:
        return 0
    if c >= 0.75:
        return 25
    return 25 * (c - 0.65) / 0.10


def score_adf(p):
    if p <= 0.05:
        return 20
    if p >= 0.15:
        return 0
    return 20 * (0.15 - p) / 0.10


def score_hl(hl):
    if np.isnan(hl):
        return 0
    if hl <= 20:
        return 20
    if hl >= 60:
        return 0
    return 20 * (60 - hl) / 40


# -------------------- main --------------------

def scan_universe(name, tickers, start, out_dir):
    prices = download_prices(tickers, start)
    pairs = []

    for a, b in itertools.combinations(prices.keys(), 2):
        s1, s2 = prices[a], prices[b]
        df = pd.concat([s1, s2], axis=1).dropna()
        if len(df) < 500:
            continue

        r1 = log_returns(df.iloc[:, 0])
        r2 = log_returns(df.iloc[:, 1])
        corr = r1.corr(r2)
        if corr < 0.65:
            continue

        beta = beta_price(df.iloc[:, 0], df.iloc[:, 1])
        if beta < 0.5 or beta > 2.0:
            continue

        spread = df.iloc[:, 0] - beta * df.iloc[:, 1]
        try:
            adf_p = adfuller(spread.dropna())[1]
        except Exception:
            adf_p = 1.0

        hl = half_life(spread)
        avg_z = abs((spread - spread.rolling(60).mean()) /
                    spread.rolling(60).std()).mean()

        vol_ratio = spread.std() / (r1.std() + r2.std())

        sc_corr = score_corr(corr)
        sc_adf = score_adf(adf_p)
        sc_hl = score_hl(hl)
        sc_vol = 20 if 0.3 <= vol_ratio <= 3.0 else 0

        score_total = sc_corr + sc_adf + sc_hl + sc_vol

        pairs.append({
            "y": a,
            "x": b,
            "corr_252": round(corr, 3),
            "beta": round(beta, 3),
            "adf_p": round(adf_p, 4),
            "half_life": round(hl, 1) if not np.isnan(hl) else np.nan,
            "avg_abs_z60": round(avg_z, 2),
            "vol_ratio": round(vol_ratio, 2),
            "score_total": round(score_total, 1)
        })

    if not pairs:
        return

    df_out = pd.DataFrame(pairs).sort_values("score_total", ascending=False)
    os.makedirs(out_dir, exist_ok=True)
    df_out.to_csv(f"{out_dir}/{name}_pairs_ranked.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True)
    ap.add_argument("--start-date", default="2018-01-01")
    args = ap.parse_args()

    tickers = [
        t.strip() for t in
        open(f"data/tickers_{args.universe}.txt").readlines()
        if t.strip()
    ]

    scan_universe(
        args.universe,
        tickers,
        args.start_date,
        f"results/{args.universe}"
    )


if __name__ == "__main__":
    main()
