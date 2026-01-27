#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_pair.py
---------------
Analiza pary KLAC–LRCX:

- Pobranie danych
- Obliczenie hedge ratio (beta)
- Spread = KLAC - beta*LRCX
- Rolling Z-score (20/40/60 dni)
- Volatility filter: sigma/mean
- ADF stationarity test
- Generowanie sygnałów long/short
- Zapis CSV z wynikami

Uruchomienie:
    python analyze_pair.py --start-date 2018-01-01
"""

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import os


def print_info(msg: str):
    print(f"[INFO] {msg}", flush=True)


def download_prices(tickers, start_date):
    print_info(f"Pobieranie danych: {tickers}")
    df = yf.download(tickers, start=start_date, progress=False, group_by="column")
    if df.empty:
        raise RuntimeError("Brak danych do analizy.")
    prices = df["Adj Close"].dropna()
    return prices


def hedge_ratio(y, x):
    """
    y = KLAC, x = LRCX
    Zwraca alpha, beta, spread
    """
    df = pd.concat([y, x], axis=1).dropna()
    df.columns = ["y", "x"]
    X = sm.add_constant(df["x"])
    model = sm.OLS(df["y"], X).fit()
    alpha = float(model.params["const"])
    beta = float(model.params["x"])
    spread = df["y"] - (alpha + beta * df["x"])
    return alpha, beta, spread


def zscore(series, window):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def adf_test(series):
    """
    Zwraca p-value testu ADF.
    """
    series = series.dropna()
    result = adfuller(series)
    return result[1]  # p-value


def analyze_pair(start_date="2018-01-01", out_csv="results/klac_lrcx_analysis.csv"):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # 1) Pobierz dane
    tickers = ["KLAC", "LRCX"]
    prices = download_prices(tickers, start_date)
    klac = prices["KLAC"]
    lrcx = prices["LRCX"]

    # 2) Hedge ratio i spread
    alpha, beta, spread = hedge_ratio(klac, lrcx)
    print_info(f"Hedge Ratio (beta) = {beta:.4f}")

    # 3) Z-score
    z20 = zscore(spread, 20)
    z40 = zscore(spread, 40)
    z60 = zscore(spread, 60)

    # 4) Volatility filter: sigma/mean
    sigma = spread.rolling(60).std()
    mean = spread.rolling(60).mean()
    vol_ratio = sigma / mean.abs()

    # 5) ADF stationarity
    pvalue_adf = adf_test(spread)
    print_info(f"ADF p-value = {pvalue_adf:.4f}")

    # 6) Sygnały tradingowe
    # Long spread (rozszerzony)
    long_signal = (z20 < -2) & (z40 < -1.5)

    # Short spread (zawężony)
    short_signal = (z20 > 2) & (z40 > 1.5)

    df = pd.DataFrame({
        "KLAC": klac,
        "LRCX": lrcx,
        "Spread": spread,
        "Z20": z20,
        "Z40": z40,
        "Z60": z60,
        "VolRatio": vol_ratio,
        "LongSignal": long_signal,
        "ShortSignal": short_signal
    })

    df.to_csv(out_csv)
    print_info(f"Zapisano analizę -> {out_csv}")
    print_info("Gotowe.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--out-csv", default="results/klac_lrcx_analysis.csv")
    args = parser.parse_args()

    analyze_pair(start_date=args.start_date, out_csv=args.out_csv)